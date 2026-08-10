"""动作守卫的不变量。

守卫必须是权威的，理由是实测出来的：环境对匹配不上的 `click[...]` 是**静默 no-op**
——返回 reward 0、done False、页面不变、不报错。守卫放过一个非法动作，模型就白吃
一步且完全收不到任何反馈。所以「守卫该拒的都拒了」这件事只能靠测试保证。
"""

from __future__ import annotations

import pytest

from ecom_agent_rl.environment.tools import (
    TOOL_NAMES,
    TOOL_SCHEMAS,
    RejectedAction,
    check,
    to_env_action,
)
from conftest import SEARCH_HOME, detail_state, search_state

def test_think_tool_is_not_exposed():
    """参考实现暴露了 think 又反复叮嘱模型别调用：它白吃一步，还会把
    latest_observation 覆盖成模型自己的话，导致后续动作全被拒、三次即判死。
    不给这个工具，问题就不存在。"""
    assert "think" not in TOOL_NAMES


def test_every_schema_forbids_extra_properties():
    for schema in TOOL_SCHEMAS:
        params = schema["function"]["parameters"]
        assert params["additionalProperties"] is False, schema["function"]["name"]


def test_search_is_rejected_when_the_page_has_no_search_bar():
    with pytest.raises(RejectedAction) as exc:
        check("search_products", {"query": "狗狗衣服"}, search_state())
    assert exc.value.reason == "search_unavailable"


def test_search_is_allowed_on_the_search_home():
    check("search_products", {"query": "狗狗衣服"}, SEARCH_HOME)


def test_empty_query_is_rejected():
    with pytest.raises(RejectedAction) as exc:
        check("search_products", {"query": "   "}, SEARCH_HOME)
    assert exc.value.reason == "empty_query"


def test_opening_an_asin_that_is_not_on_the_current_page_is_rejected():
    """只能点最新 observation 里的 asin，历史页面的不行——这是最常见的越界。"""
    with pytest.raises(RejectedAction) as exc:
        check("open_product", {"asin": "111111111111"}, search_state())
    assert exc.value.reason == "asin_not_on_page"


def test_opening_an_asin_on_the_current_page_is_allowed():
    state = search_state()
    check("open_product", {"asin": state["products"][0]["asin"]}, state)


def test_buy_now_is_rejected_when_the_button_is_absent():
    with pytest.raises(RejectedAction) as exc:
        check("buy_now", {}, search_state())
    assert exc.value.reason == "button_not_available"


def test_buy_now_is_allowed_on_a_detail_page():
    check("buy_now", {}, detail_state())


def test_view_attributes_is_rejected_when_the_product_has_no_attributes_subpage():
    """实测有商品只有 description/features/reviews 三个按钮，没有 attributes。"""
    with pytest.raises(RejectedAction) as exc:
        check("view_attributes", {}, detail_state())
    assert exc.value.reason == "button_not_available"


def test_select_option_rejects_an_axis_the_product_does_not_have():
    with pytest.raises(RejectedAction) as exc:
        check("select_option", {"axis": "颜色", "value": "l"}, detail_state())
    assert exc.value.reason == "unknown_option_axis"


def test_select_option_rejects_a_value_from_another_axis():
    """带上 axis 的意义就在这里：环境只按值匹配，跨轴误选它察觉不到。"""
    state = detail_state()
    state["available_options"] = {"尺码": ["l", "xs"], "颜色分类": ["蓝黄", "橙白"]}
    state["actions"] = state["actions"] + ["蓝黄", "橙白"]
    with pytest.raises(RejectedAction) as exc:
        check("select_option", {"axis": "尺码", "value": "蓝黄"}, state)
    assert exc.value.reason == "value_not_in_axis"


def test_select_option_rejects_a_value_shared_by_two_axes():
    """同名值出现在两轴时环境无法区分，拒掉比蒙一个安全。"""
    state = detail_state()
    state["available_options"] = {"尺码": ["l", "标准"], "款式": ["加厚", "标准"]}
    state["actions"] = state["actions"] + ["标准", "加厚"]
    with pytest.raises(RejectedAction) as exc:
        check("select_option", {"axis": "尺码", "value": "标准"}, state)
    assert exc.value.reason == "ambiguous_option_value"


def test_select_option_rejects_a_value_that_is_not_clickable_now():
    state = detail_state()
    state["actions"] = [a for a in state["actions"] if a != "xs"]
    with pytest.raises(RejectedAction) as exc:
        check("select_option", {"axis": "尺码", "value": "xs"}, state)
    assert exc.value.reason == "option_not_clickable"


def test_extra_arguments_are_rejected():
    """无参工具被塞参数是模型的常见错误，schema 拦不住需要守卫兜。"""
    with pytest.raises(RejectedAction) as exc:
        check("buy_now", {"asin": "900000000000"}, detail_state())
    assert exc.value.reason == "extra_arguments"


def test_missing_required_arguments_are_rejected():
    with pytest.raises(RejectedAction) as exc:
        check("open_product", {}, search_state())
    assert exc.value.reason == "missing_arguments"


def test_an_unknown_tool_name_is_rejected():
    with pytest.raises(RejectedAction) as exc:
        check("click_anything", {}, detail_state())
    assert exc.value.reason == "unknown_tool"


def test_finish_requires_the_enumerated_reason():
    with pytest.raises(RejectedAction) as exc:
        check("finish_without_purchase", {"reason": "too_expensive"}, search_state())
    assert exc.value.reason == "invalid_finish_reason"


def test_rejection_message_tells_the_model_what_is_legal_now():
    """不告诉模型现在能做什么，它只会原样重试，三次就判死。"""
    with pytest.raises(RejectedAction) as exc:
        check("open_product", {"asin": "111111111111"}, search_state())
    message = exc.value.message
    assert "本页 asin" in message
    assert search_state()["products"][0]["asin"] in message


@pytest.mark.parametrize(
    ("name", "arguments", "expected"),
    [
        ("search_products", {"query": "狗狗衣服"}, "search[狗狗衣服]"),
        ("open_product", {"asin": "900000000000"}, "click[900000000000]"),
        ("select_option", {"axis": "尺码", "value": "l"}, "click[l]"),
        ("view_description", {}, "click[description]"),
        ("next_page", {}, "click[next >]"),
        ("prev_page", {}, "click[< prev]"),
        ("back_to_search", {}, "click[back to search]"),
        ("buy_now", {}, "click[buy now]"),
        ("finish_without_purchase", {"reason": "no_suitable_product"}, "finish[no_suitable_product]"),
    ],
)
def test_action_strings_match_what_the_environment_accepts(name, arguments, expected):
    """环境的 actions 全是小写，click 也按小写匹配（实测确认）。"""
    assert to_env_action(name, arguments) == expected


def test_every_tool_maps_to_an_action_string():
    """新增工具时忘了加映射，会在 rollout 里才炸。"""
    samples = {
        "search_products": {"query": "q"},
        "open_product": {"asin": "900000000000"},
        "select_option": {"axis": "尺码", "value": "l"},
        "finish_without_purchase": {"reason": "no_suitable_product"},
    }
    for name in TOOL_NAMES:
        assert to_env_action(name, samples.get(name, {}))
