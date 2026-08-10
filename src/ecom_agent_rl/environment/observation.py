"""把环境返回的结构化状态渲染成模型可见的文本，并挡住答案字段。

环境每次 reset / interact 都会回一大坨字段，其中不少是**答案或答案的推论**：

- `goal_options` —— 目标商品的规格，reset 时就给（实测：`["香草奶昔运动衣 绿", "XS：胸围31cm..."]`）
- `goal` / `purchase` / `reward_detail` —— 终局才非空，但含 gold asin 与逐维打分
- `progress.credited_evidence_added` —— **每一步**都给，含
  `constraint:<asin>:budget:fail` 这类逐条约束判定。这是奖励函数的中间结果，
  模型看到就等于拿到了评分器
- `instruction_simple` / `user_persona` / `reason_key` —— 任务的另一种表述与人设，
  参考实现没用，但同样不该进 prompt

所以这里用**白名单**：只有显式许可的字段能进入模型可见面，其余一律拦掉。遇到
陌生字段直接报错——环境升级新增了字段时，我们要被吵醒，而不是静默泄漏。
"""

from __future__ import annotations

import json
from typing import Any, Mapping

OBSERVATION_VERSION = "shopping-observation-v2"

# 模型可见的 observation_state 字段。渲染器只读这些。
_STATE_FIELDS = frozenset(
    {
        "observation_version",
        "page_type",
        "search_available",
        "actions",
        # search_results
        "query",
        "normalized_query",
        "page",
        "total_pages",
        "total_results",
        "rank_start",
        "rank_end",
        "products",
        # product_detail / information_subpage
        "product",
        "selected_options",
        "available_options",
        "selected_price",
        "subpage",
        "content",
    }
)

# 答案或答案的推论。出现即剥离，并记进审计字段。
FORBIDDEN_FIELDS = frozenset(
    {
        "goal",
        "goal_options",
        "purchase",
        "reward",
        "reward_detail",
        "progress",
        "termination_reason",
        "reward_valid",
        "target_asin",
        "answer",
    }
)

# 与答案无关的协议字段。不进 prompt，但留着做审计和调试。
_METADATA_FIELDS = frozenset(
    {
        "message",
        "env_idx",
        "idx",
        "done",
        "over",
        "environment_version",
        "instruction",
        "instruction_simple",
        "user_persona",
        "reason_key",
        "observation_state",
    }
)

# 搜索结果页固定 20 条。这是断言而不是截断上限：不允许因为放不下就少给模型一个候选，
# 否则守卫认为合法的 asin 模型压根没见过。
PAGE_SIZE = 20


class ObservationError(RuntimeError):
    """结构化状态不符合预期，渲染无法安全进行。"""


def split_env_payload(raw: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """把环境返回拆成「可用」与「被拦下的答案字段」两份。

    白名单之外的陌生字段直接报错：环境加了新字段时必须有人来判断它是不是泄漏，
    默认放过的代价太大。
    """
    unknown = set(raw) - _STATE_FIELDS - FORBIDDEN_FIELDS - _METADATA_FIELDS
    if unknown:
        raise ObservationError(
            f"环境返回了未登记的字段 {sorted(unknown)}。"
            "先判断它是否含答案信息，再加进 observation.py 的白名单或黑名单。"
        )
    blocked = {key: raw[key] for key in sorted(FORBIDDEN_FIELDS & set(raw))}
    allowed = {key: value for key, value in raw.items() if key not in FORBIDDEN_FIELDS}
    return allowed, blocked


def validate_state(state: Any) -> dict[str, Any]:
    """校验结构化状态：版本对得上、没有答案字段、页大小没超。"""
    if not isinstance(state, Mapping):
        raise ObservationError(f"observation_state 不是对象: {type(state).__name__}")
    version = state.get("observation_version")
    if version != OBSERVATION_VERSION:
        raise ObservationError(
            f"observation_state 版本是 {version!r}，期望 {OBSERVATION_VERSION!r}"
        )
    leaked = FORBIDDEN_FIELDS & set(state)
    if leaked:
        raise ObservationError(f"observation_state 含答案字段: {sorted(leaked)}")
    products = state.get("products")
    if isinstance(products, list) and len(products) > PAGE_SIZE:
        raise ObservationError(
            f"搜索页有 {len(products)} 条商品，超过约定的每页 {PAGE_SIZE} 条"
        )
    return dict(state)


def _fmt(value: Any) -> str:
    if value is None or value == "":
        return "未知"
    return str(value)


def _render_search_results(state: Mapping[str, Any]) -> list[str]:
    lines = [
        f"查询: {_fmt(state.get('query'))}",
        f"第 {_fmt(state.get('page'))}/{_fmt(state.get('total_pages'))} 页，"
        f"共 {_fmt(state.get('total_results'))} 条结果"
        f"（本页第 {_fmt(state.get('rank_start'))}-{_fmt(state.get('rank_end'))} 条）",
        "",
        "格式: 排名|asin|价格|品牌|品类|关键属性|标题",
    ]
    for product in state.get("products") or []:
        attrs = "/".join(str(a) for a in product.get("key_attributes") or [])
        lines.append(
            "|".join(
                (
                    _fmt(product.get("rank")),
                    _fmt(product.get("asin")),
                    _fmt(product.get("price")),
                    _fmt(product.get("brand")),
                    _fmt(product.get("category")),
                    attrs,
                    _fmt(product.get("title")),
                )
            )
        )
    return lines


def _render_product(state: Mapping[str, Any]) -> list[str]:
    product = state.get("product") or {}
    # 选了规格之后 selected_price 才是能下单的真实价格，基础 price 可能是区间。
    price = state.get("selected_price", product.get("price"))
    lines = [
        f"asin: {_fmt(product.get('asin'))}",
        f"标题: {_fmt(product.get('title'))}",
        f"品牌: {_fmt(product.get('brand'))}",
        f"品类: {_fmt(product.get('category'))}",
        f"价格: {_fmt(price)}",
        f"关键属性: {'/'.join(str(a) for a in product.get('key_attributes') or []) or '未知'}",
    ]
    available = state.get("available_options") or {}
    if available:
        lines.append("可选规格:")
        for axis, values in available.items():
            lines.append(f"  {axis}: {json.dumps(values, ensure_ascii=False)}")
    selected = state.get("selected_options") or {}
    lines.append(
        f"已选规格: {json.dumps(selected, ensure_ascii=False) if selected else '（无）'}"
    )
    if available:
        missing = [axis for axis in available if axis not in selected]
        lines.append(f"未选规格轴: {json.dumps(missing, ensure_ascii=False) if missing else '（已选全）'}")
    return lines


def render(state: Mapping[str, Any]) -> str:
    """把结构化状态渲染成模型可见文本。

    渲染只是展示层，动作合法性由 guard 直接读 `actions` 判定，不从这段文本反解。
    参考实现把守卫建在正则解析渲染文本上，改一句措辞就会静默破坏守卫。
    """
    state = validate_state(state)
    page_type = str(state.get("page_type") or "unknown")
    header = {
        "search_home": "【搜索首页】",
        "search_results": "【搜索结果】",
        "product_detail": "【商品详情】",
        "information_subpage": "【商品信息子页】",
        "terminal": "【已结束】",
    }.get(page_type, f"【{page_type}】")

    lines = [header]
    if page_type == "search_results":
        lines.extend(_render_search_results(state))
    elif page_type in {"product_detail", "information_subpage"}:
        lines.extend(_render_product(state))
        if page_type == "information_subpage":
            lines.append(f"子页: {_fmt(state.get('subpage'))}")
            lines.append("内容:")
            lines.append(_fmt(state.get("content")))
    elif page_type == "search_home":
        lines.append("还没有搜索过，先用 search_products 发起一次查询。")

    lines.append("")
    lines.append(f"搜索是否可用: {bool(state.get('search_available'))}")
    lines.append(
        "可用动作: " + json.dumps(list(state.get("actions") or []), ensure_ascii=False)
    )
    return "\n".join(lines)
