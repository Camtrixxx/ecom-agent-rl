"""SFT 筛选器的测试。

重点钉三件事：接受口径可切换（方案 C 的两份数据靠它）、被拒动作被剔除且剩下的
observation 序列仍连续、答案字段不会漏进训练样本。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ecom_agent_rl.data.sft import (
    DEFAULT_MAX_REJECTIONS,
    GOLD_ONLY,
    SUCCESS_TYPES,
    Report,
    acceptance_reasons,
    build_row,
    select,
    training_messages,
)


def trajectory(
    *,
    task_id: int = 1,
    status: str = "done",
    reward_type: str = "gold_purchase",
    reward_valid: bool = True,
    rejection_count: int = 0,
    steps: list[dict[str, Any]] | None = None,
    messages: list[dict[str, Any]] | None = None,
    rejections: list[dict[str, Any]] | None = None,
    error: Any = None,
    done: bool = True,
) -> dict[str, Any]:
    if steps is None:
        steps = [
            {"step": 0, "tool_name": "search_products", "arguments": {"query": "x"},
             "observation": "【搜索结果】..."},
            {"step": 1, "tool_name": "buy_now", "arguments": {}, "observation": "终局"},
        ]
    if messages is None:
        messages = [
            {"role": "system", "content": "prompt"},
            {"role": "user", "content": "任务"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search_products", "arguments": '{"query": "x"}'}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "【搜索结果】..."},
        ]
    return {
        "trajectory_id": f"t{task_id}",
        "task_id": task_id,
        "status": status,
        "done": done,
        "error": error,
        "steps": steps,
        "messages": messages,
        "rejections": rejections or [],
        "rejection_count": rejection_count,
        "audit": {"terminal": {
            "reward_valid": reward_valid,
            "reward_detail": {"reward_type": reward_type},
        }},
    }


# --- 接受口径 -----------------------------------------------------------------

def test_a_clean_gold_trajectory_is_accepted():
    ok, reasons = acceptance_reasons(trajectory())
    assert ok, reasons


def test_valid_alternative_is_accepted_under_the_evaluation_criterion():
    """我们的评测口径含 valid_alternative（reward 0.55），训练口径默认与它一致。"""
    ok, _ = acceptance_reasons(
        trajectory(reward_type="valid_alternative_purchase"), accept_types=SUCCESS_TYPES
    )
    assert ok


def test_valid_alternative_is_rejected_under_the_gold_only_criterion():
    """参考实现的口径。同一批轨迹筛两份，靠的就是这个参数。"""
    ok, reasons = acceptance_reasons(
        trajectory(reward_type="valid_alternative_purchase"), accept_types=GOLD_ONLY
    )
    assert not ok
    assert any("valid_alternative_purchase" in r for r in reasons)


@pytest.mark.parametrize("reward_type", [
    "partial_alternative_purchase", "wrong_purchase", "repeat_loop",
    "max_steps", "early_abstain", "graceful_stop", "reward_unverifiable",
])
def test_non_success_reward_types_are_always_rejected(reward_type: str):
    ok, _ = acceptance_reasons(trajectory(reward_type=reward_type))
    assert not ok


def test_an_untrustworthy_reward_is_rejected_even_when_gold():
    """reward_valid=False 意味着环境自己都不确信这个判定。"""
    ok, reasons = acceptance_reasons(trajectory(reward_valid=False))
    assert not ok
    assert "reward_not_valid" in reasons


def test_all_reasons_are_reported_not_just_the_first():
    """调口径时需要知道「放宽某一项能多收多少」，所以原因要全。"""
    ok, reasons = acceptance_reasons(
        trajectory(status="no_tool_call", done=False, reward_type="repeat_loop",
                   reward_valid=False, rejection_count=5)
    )
    assert not ok
    assert len(reasons) >= 4, reasons


def test_a_trajectory_without_a_buy_step_is_rejected():
    ok, reasons = acceptance_reasons(trajectory(steps=[
        {"step": 0, "tool_name": "search_products", "arguments": {}, "observation": "x"}]))
    assert not ok
    assert "missing_buy" in reasons


def test_the_rejection_threshold_can_be_tightened_or_relaxed():
    dirty = trajectory(rejection_count=4)
    assert not acceptance_reasons(dirty)[0], "默认阈值 3，4 次应被拒"
    assert acceptance_reasons(dirty, max_rejections=4)[0]
    assert not acceptance_reasons(trajectory(rejection_count=1), max_rejections=0)[0]


def test_the_default_threshold_sits_at_the_measured_plateau():
    """实测阈值曲线（150 题）：0→3.3%, 1→19.3%, 2→34.7%, 3→38.0%, 4+→38.0%。

    平台在 3：阈值 3 时「仅因 rejections 被丢」的轨迹为 0 条，再放宽没有余量；
    阈值 2 会白丢 5 条全是 gold_purchase 的轨迹。这条钉住默认值——改动前该重跑
    扫描，而不是凭直觉挪。
    """
    assert DEFAULT_MAX_REJECTIONS == 3
    for count in (0, 1, 2, 3):
        assert acceptance_reasons(trajectory(rejection_count=count))[0], count
    assert not acceptance_reasons(trajectory(rejection_count=4))[0]


def test_multiple_tool_calls_in_one_turn_are_rejected():
    """并发点击是在一个已经变了的页面上盲点，不能进训练集。"""
    msgs = trajectory()["messages"]
    msgs[2]["tool_calls"].append(
        {"id": "c2", "type": "function",
         "function": {"name": "open_product", "arguments": "{}"}})
    ok, reasons = acceptance_reasons(trajectory(messages=msgs))
    assert not ok
    assert any("multiple_tool_calls" in r for r in reasons)


def test_an_errored_trajectory_is_rejected():
    ok, reasons = acceptance_reasons(trajectory(error="LLMError: boom"))
    assert not ok
    assert "has_error" in reasons


# --- 被拒动作的剔除 -----------------------------------------------------------

def rejected_pair_trajectory() -> dict[str, Any]:
    """教师第 2 步点了当前页不允许点的东西，被拒，第 3 步改对了。"""
    messages = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "任务"},
        {"role": "assistant", "content": "先搜索", "tool_calls": [
            {"id": "ok1", "type": "function",
             "function": {"name": "search_products", "arguments": '{"query": "x"}'}}]},
        {"role": "tool", "tool_call_id": "ok1", "content": "【搜索结果】第1页"},
        {"role": "assistant", "content": "再搜一次", "tool_calls": [
            {"id": "bad1", "type": "function",
             "function": {"name": "search_products", "arguments": '{"query": "y"}'}}]},
        {"role": "tool", "tool_call_id": "bad1", "content": "动作被拒绝，未执行。当前页面不能搜索"},
        {"role": "assistant", "content": "改为打开商品", "tool_calls": [
            {"id": "ok2", "type": "function",
             "function": {"name": "open_product", "arguments": '{"asin": "123"}'}}]},
        {"role": "tool", "tool_call_id": "ok2", "content": "【商品详情】"},
    ]
    return trajectory(
        messages=messages,
        rejection_count=1,
        rejections=[{"at_step": 1, "reason": "search_unavailable",
                     "tool_call": {"id": "bad1"}}],
    )


def test_rejected_actions_and_their_replies_are_removed():
    out = training_messages(rejected_pair_trajectory())
    ids = [m.get("tool_call_id") for m in out if m["role"] == "tool"]
    assert "bad1" not in ids, "被拒动作的错误回复不能留在训练集"
    call_ids = [c["id"] for m in out if m.get("tool_calls") for c in m["tool_calls"]]
    assert call_ids == ["ok1", "ok2"], "只保留成功路径"


def test_the_observation_sequence_stays_contiguous_after_removal():
    """guard 拒绝时动作根本没执行、环境状态没变，所以删掉它不会断开 observation。

    这条不变量来自 agent.py：tools.check 在本地对 state 判定，抛 RejectedAction
    后直接 append 消息并 continue，从不调用环境。若哪天 guard 改成先执行再判定，
    这个测试仍会过但语义就错了——所以 agent.py 那段的注释也要一起看。
    """
    out = training_messages(rejected_pair_trajectory())
    roles = [m["role"] for m in out]
    # assistant 与 tool 必须严格交替，不能出现连续两个 assistant。
    for a, b in zip(roles, roles[1:]):
        assert not (a == "assistant" and b == "assistant"), roles
    observations = [m["content"] for m in out if m["role"] == "tool"]
    assert observations == ["【搜索结果】第1页", "【商品详情】"]
    assert all("被拒绝" not in o for o in observations)


def test_teacher_reasoning_is_stripped():
    """学生学动作，不学教师的思维链——不同模型的 reasoning 格式也不通用。"""
    msgs = trajectory()["messages"]
    msgs[2]["reasoning_content"] = "让我想想……"
    out = training_messages(trajectory(messages=msgs))
    assert all("reasoning_content" not in m for m in out)
    assert json.dumps(out, ensure_ascii=False).find("让我想想") == -1


def test_the_training_row_carries_no_answer_fields():
    """audit 里的 gold / reward_detail 绝不能漏进训练样本。"""
    row = build_row(trajectory(), tools=[{"type": "function",
                                          "function": {"name": "search_products"}}])
    blob = json.dumps(row, ensure_ascii=False)
    for leak in ("reward_detail", "reward_type", "reward_valid", "audit", "gold"):
        assert leak not in blob, f"训练样本泄漏了 {leak}"


def test_the_training_row_keeps_the_full_episode_as_one_sample():
    row = build_row(trajectory(), tools=[])
    assert row["task_id"] == 1
    assert [m["role"] for m in row["messages"]] == ["system", "user", "assistant", "tool"]


# --- 批筛与审计 ---------------------------------------------------------------

def test_select_dedupes_by_task_keeping_the_first():
    verdicts, report = select([trajectory(task_id=7), trajectory(task_id=7)])
    assert [v.accepted for v in verdicts] == [True, False]
    assert verdicts[1].reasons == ("duplicate_task",)
    assert report.accepted == 1
    assert report.duplicate_tasks == 1


def test_select_excludes_held_out_tasks():
    _, report = select([trajectory(task_id=9)], held_out_task_ids=frozenset({9}))
    assert report.accepted == 0
    assert report.held_out == 1


def test_dedupe_and_held_out_do_not_distort_the_acceptance_rate_numerator():
    """质量判据先跑，切分策略后跑：报告要能分开这两种丢弃。"""
    _, report = select(
        [trajectory(task_id=1), trajectory(task_id=1), trajectory(task_id=2, reward_type="repeat_loop")]
    )
    assert report.total == 3
    assert report.accepted == 1
    assert report.duplicate_tasks == 1
    assert any("repeat_loop" in r for r in report.reason_counts)


def test_the_report_records_the_reward_type_distribution_of_every_trajectory():
    """包括被丢弃的——判断该放宽哪一边靠的就是这个分布。"""
    _, report = select([
        trajectory(task_id=1),
        trajectory(task_id=2, reward_type="repeat_loop"),
        trajectory(task_id=3, status="no_tool_call", reward_type=""),
    ])
    assert report.reward_types["gold_purchase"] == 1
    assert report.reward_types["repeat_loop"] == 1
    assert report.reward_types["no_terminal:no_tool_call"] == 1


def test_the_acceptance_rate_is_reported():
    _, report = select([trajectory(task_id=i) for i in range(4)]
                       + [trajectory(task_id=9, reward_type="wrong_purchase")])
    assert report.total == 5
    assert report.accepted == 4
    assert report.acceptance_rate == pytest.approx(0.8)
    assert report.to_dict()["acceptance_rate"] == 0.8


def test_an_empty_batch_does_not_divide_by_zero():
    _, report = select([])
    assert report.acceptance_rate == 0.0
