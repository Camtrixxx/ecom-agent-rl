"""模型可见面的不变量：答案字段一个都不许漏。

环境每步都会回一批答案或答案的推论（`goal_options` 在 reset 就给、`progress` 里有
逐条约束判定）。这些字段进了 prompt，后面所有指标就都不可信了，而且这种泄漏不会
报错、只会让分数变好看，所以必须靠测试钉住。
"""

from __future__ import annotations

import pytest

from ecom_agent_rl.environment.observation import (
    FORBIDDEN_FIELDS,
    OBSERVATION_VERSION,
    ObservationError,
    render,
    split_env_payload,
    validate_state,
)
from conftest import detail_state, search_state


@pytest.mark.parametrize("field", sorted(FORBIDDEN_FIELDS))
def test_every_forbidden_field_is_stripped_from_the_payload(field):
    """逐个字段验证，而不是只测一两个——漏一个就是漏了答案。"""
    raw = {"instruction": "买点东西", "observation_state": search_state(), field: "SECRET"}
    allowed, blocked = split_env_payload(raw)
    assert field not in allowed, f"{field} 没有被剥离"
    assert blocked[field] == "SECRET", f"{field} 没有进审计记录"


def test_reset_payload_from_the_real_environment_is_stripped():
    """实测 reset 会回 goal_options（就是目标商品的规格）和 user_persona。"""
    raw = {
        "instruction": "想给狗狗买件衣服，预算 35",
        "instruction_simple": "推荐香草奶昔色狗狗衣服",
        "goal_options": ["香草奶昔运动衣 绿", "XS：胸围31cm（建议1-3斤）"],
        "user_persona": "养狗的年轻人",
        "reason_key": "k",
        "message": "Task 7 started",
        "env_idx": 1,
        "idx": 7,
        "environment_version": "shopsimulator-environment-v2.1",
        "observation_state": {
            "observation_version": OBSERVATION_VERSION,
            "page_type": "search_home",
            "search_available": True,
            "actions": ["search"],
        },
    }
    allowed, blocked = split_env_payload(raw)
    assert "goal_options" not in allowed
    assert blocked["goal_options"] == ["香草奶昔运动衣 绿", "XS：胸围31cm（建议1-3斤）"]


def test_progress_is_stripped_because_it_leaks_per_step_constraint_verdicts():
    """`progress` 每一步都给，里面有 `constraint:<asin>:budget:fail` —— 等于把评分器给了模型。"""
    raw = {
        "observation_state": detail_state(),
        "progress": {
            "credited_evidence_added": [
                "product:892874612717",
                "constraint:892874612717:budget:fail",
            ]
        },
    }
    allowed, blocked = split_env_payload(raw)
    assert "progress" not in allowed
    assert "budget:fail" in str(blocked["progress"])


def test_unknown_fields_are_rejected_rather_than_passed_through():
    """环境升级加了新字段时要有人来判断它是否含答案，不能默认放过。"""
    with pytest.raises(ObservationError, match="未登记的字段"):
        split_env_payload({"observation_state": detail_state(), "gold_answer_v2": "x"})


def test_validate_rejects_a_state_carrying_answer_fields():
    state = detail_state() | {"goal": {"asin": "900000000000"}}
    with pytest.raises(ObservationError, match="答案字段"):
        validate_state(state)


def test_validate_rejects_an_unexpected_observation_version():
    with pytest.raises(ObservationError, match="版本"):
        validate_state(detail_state() | {"observation_version": "shopping-observation-v3"})


def test_validate_rejects_an_oversized_search_page():
    """页大小是断言不是截断：不允许因为放不下就少给模型一个候选。"""
    with pytest.raises(ObservationError, match="每页"):
        validate_state(search_state(count=21))


def test_rendered_search_page_shows_every_actionable_asin():
    """守卫按 `actions` 判合法；渲染少写一个 asin 就会出现「合法但模型没见过」。"""
    state = search_state(count=5)
    text = render(state)
    for product in state["products"]:
        assert product["asin"] in text


def test_rendered_detail_page_prefers_the_selected_variant_price():
    """选完规格后能下单的是 selected_price，基础 price 可能只是区间。"""
    state = detail_state() | {"product": dict(detail_state()["product"], price="108.0 to 260.0")}
    state["selected_price"] = 139.0
    text = render(state)
    assert "价格: 139.0" in text
    assert "108.0 to 260.0" not in text


def test_rendered_detail_page_lists_unselected_option_axes():
    """未选满规格就不该买，模型需要知道还差哪几轴。"""
    text = render(detail_state())
    assert "未选规格轴" in text and "尺码" in text


def test_render_refuses_a_state_with_answer_fields():
    """渲染是最后一道关口，也要自己校验，不能假设调用方已经过滤过。"""
    with pytest.raises(ObservationError):
        render(detail_state() | {"reward_detail": {"brand": 1.0}})
