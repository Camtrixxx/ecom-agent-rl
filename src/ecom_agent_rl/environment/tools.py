"""暴露给模型的工具集，以及「这一步合法吗」的守卫。

两个与参考实现不同的决定：

1. **守卫读结构化 `actions`，不解析渲染文本。** 参考实现用正则从中文渲染文本里抠
   `可点击的按钮: [...]` 和 `^\\d+\\|(asin)\\|`，措辞一改守卫就静默失效。环境本来
   就在 `observation_state["actions"]` 里给了权威列表，直接用。
2. **不暴露 `think`。** 参考实现挂了个 `think` 工具，又在描述和 system prompt 里
   反复叮嘱模型别调用它——因为它一旦被调用就会白吃一步，还会把 `latest_observation`
   覆盖成模型自己的碎话，导致后续所有页面相关动作被守卫拒掉、三次即判死。
   不给这个工具，问题就不存在。

`select_option` 带上 `axis`：环境的 `click[value]` 只按值匹配，模型选错规格轴时
无法察觉。`available_options` 里有轴→值的映射，要求模型说清楚选的是哪个轴，就能
在守卫层挡住跨轴误选。
"""

from __future__ import annotations

import re
from typing import Any, Mapping

# 环境的商品 id 是 8-12 位数字，不是真的 Amazon ASIN，但字段名沿用 asin。
PRODUCT_ID = re.compile(r"^\d{8,12}$")

# 无参工具 -> 它对应的 click 目标。环境的 actions 全是小写，click 匹配也按小写走。
_CLICK_TOOLS: dict[str, str] = {
    "view_description": "description",
    "view_features": "features",
    "view_reviews": "reviews",
    "view_attributes": "attributes",
    "next_page": "next >",
    "prev_page": "< prev",
    "back_to_search": "back to search",
    "buy_now": "buy now",
}


def _schema(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties or {},
                "required": required or [],
                "additionalProperties": False,
            },
        },
    }


TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema(
        "search_products",
        "发起一次搜索。仅当最新 observation 显示「搜索是否可用: True」时可用。"
        "query 要简短，只放品类加最有区分度的品牌、型号、功能或规格；"
        "不要整段复制用户需求，也不要重复已经用过的查询。",
        {"query": {"type": "string"}},
        ["query"],
    ),
    _schema(
        "open_product",
        "打开最新 observation 当前页列出的某个候选商品，核验它的价格、品类与规格。"
        "asin 必须原样取自最新 observation，不能用历史页面里的。",
        {"asin": {"type": "string"}},
        ["asin"],
    ),
    _schema(
        "select_option",
        "为当前商品的某个规格轴选一个值。axis 与 value 都必须来自最新 observation 的"
        "「可选规格」，同一轴只选一个值。选完要按新的实际价格重新判断预算。",
        {"axis": {"type": "string"}, "value": {"type": "string"}},
        ["axis", "value"],
    ),
    _schema("view_description", "查看当前商品的 Description 子页；无参数，传 {}。"),
    _schema("view_features", "查看当前商品的 Features 子页；无参数，传 {}。"),
    _schema("view_reviews", "查看当前商品的 Reviews 子页，仅用于辅助判断使用体验，"
                           "不能用来确认型号、规格或价格；无参数，传 {}。"),
    _schema("view_attributes", "查看当前商品的 Attributes 子页；无参数，传 {}。"),
    _schema("next_page", "当前页没有合适候选时翻到下一页；无参数，传 {}。"),
    _schema("prev_page", "返回上一页（在信息子页时用它退回商品详情）；无参数，传 {}。"),
    _schema("back_to_search", "回到搜索页重新查询；无参数，传 {}。"),
    _schema(
        "buy_now",
        "不可撤销的终止动作。仅当品类正确、完整规格下的实际价格在预算内，"
        "且在已核验候选中综合最优时才调用；无参数，传 {}。",
    ),
    _schema(
        "finish_without_purchase",
        "主动结束且不购买——这不算成功。仅当做过多次实质不同的搜索、核验过多个候选，"
        "仍没有可接受商品且没有明显值得继续核验的候选时调用。",
        {"reason": {"type": "string", "enum": ["no_suitable_product"]}},
        ["reason"],
    ),
]

TOOL_NAMES = frozenset(
    schema["function"]["name"] for schema in TOOL_SCHEMAS
)


class RejectedAction(Exception):
    """守卫拒绝了这次调用。`reason` 会回给模型，让它改一个合法动作。"""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


def _actions(state: Mapping[str, Any]) -> list[str]:
    return [str(a) for a in (state.get("actions") or [])]


def _product_ids(state: Mapping[str, Any]) -> list[str]:
    return [a for a in _actions(state) if PRODUCT_ID.match(a)]


def _legal_hint(state: Mapping[str, Any]) -> str:
    """拒绝时告诉模型现在到底能做什么，否则它只会原样重试。"""
    buttons = [a for a in _actions(state) if not PRODUCT_ID.match(a)]
    parts = [f"可点击按钮={buttons}"]
    products = _product_ids(state)
    if products:
        parts.append(f"本页 asin={products}")
    parts.append(f"搜索可用={bool(state.get('search_available'))}")
    options = state.get("available_options") or {}
    if options:
        parts.append(f"可选规格={options}")
    return "；".join(parts)


def check(name: str, arguments: Mapping[str, Any], state: Mapping[str, Any] | None) -> None:
    """在把动作交给环境之前判定它是否合法，不合法就 raise RejectedAction。

    守卫必须是权威的：环境对无法匹配的 `click[...]` 是**静默 no-op**——返回 reward 0、
    done False、页面不变、没有任何报错（实测确认）。若守卫放过一个非法动作，模型会
    白吃一步且完全收不到反馈。
    """
    if name not in TOOL_NAMES:
        raise RejectedAction("unknown_tool", f"没有名为 {name!r} 的工具")
    if state is None:
        raise RejectedAction("missing_observation", "还没有可用的 observation")

    schema = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == name)
    allowed = set(schema["function"]["parameters"]["properties"])
    extra = sorted(set(arguments) - allowed)
    if extra:
        raise RejectedAction(
            "extra_arguments", f"{name} 不接受参数 {extra}，只接受 {sorted(allowed)}"
        )
    missing = sorted(set(schema["function"]["parameters"]["required"]) - set(arguments))
    if missing:
        raise RejectedAction("missing_arguments", f"{name} 缺少必填参数 {missing}")

    hint = _legal_hint(state)

    if name == "search_products":
        if not state.get("search_available"):
            raise RejectedAction(
                "search_unavailable", f"当前页面不能搜索，先 back_to_search。{hint}"
            )
        if not str(arguments["query"]).strip():
            raise RejectedAction("empty_query", "query 不能为空")
        return

    if name == "finish_without_purchase":
        if arguments["reason"] != "no_suitable_product":
            raise RejectedAction(
                "invalid_finish_reason", "reason 只能是 'no_suitable_product'"
            )
        return

    if name == "open_product":
        asin = str(arguments["asin"])
        if asin not in _product_ids(state):
            raise RejectedAction(
                "asin_not_on_page",
                f"asin {asin} 不在当前页。只能打开最新 observation 列出的商品。{hint}",
            )
        return

    if name == "select_option":
        axis, value = str(arguments["axis"]), str(arguments["value"])
        available = state.get("available_options") or {}
        if axis not in available:
            raise RejectedAction(
                "unknown_option_axis",
                f"当前商品没有规格轴 {axis!r}。{hint}",
            )
        values = [str(v) for v in available[axis]]
        if value not in values:
            raise RejectedAction(
                "value_not_in_axis",
                f"{value!r} 不是规格轴 {axis!r} 的可选值，可选 {values}",
            )
        # 环境按值匹配，同名值出现在两个轴上时无法区分；这种商品直接拒掉更安全。
        others = [a for a, vs in available.items() if a != axis and value in [str(v) for v in vs]]
        if others:
            raise RejectedAction(
                "ambiguous_option_value",
                f"值 {value!r} 同时属于规格轴 {others}，环境只按值匹配，无法确定选的是哪一轴",
            )
        if value not in _actions(state):
            raise RejectedAction(
                "option_not_clickable",
                f"{value!r} 当前不可点击（可能已选过或需先退回商品详情）。{hint}",
            )
        return

    target = _CLICK_TOOLS[name]
    if target not in _actions(state):
        raise RejectedAction(
            "button_not_available",
            f"当前页面没有 {target!r} 按钮，{name} 不可用。{hint}",
        )


def to_env_action(name: str, arguments: Mapping[str, Any]) -> str:
    """把工具调用转成环境接受的动作串：`search[q]` / `click[v]` / `finish[r]`。"""
    if name == "search_products":
        return f"search[{arguments['query']}]"
    if name == "finish_without_purchase":
        return f"finish[{arguments['reason']}]"
    if name == "open_product":
        return f"click[{arguments['asin']}]"
    if name == "select_option":
        return f"click[{arguments['value']}]"
    return f"click[{_CLICK_TOOLS[name]}]"
