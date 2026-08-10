"""指标层的测试。

重点钉住三件容易被改回去的事：终局标签取自 audit 而不是猜、置信区间按题聚合、
配对比较只用共同 task_id。
"""

from __future__ import annotations

import json
import math
from typing import Any

import pytest

from ecom_agent_rl.evaluation.metrics import (
    REWARD_VALUES,
    SUCCESS_TYPES,
    Outcome,
    bootstrap_ci,
    format_summary,
    load_outcomes,
    load_strata,
    outcome_from_record,
    paired_comparison,
    stratified,
    summarize,
)


_UNSET = object()


def record(
    task_id: int,
    reward_type: str | None = "gold_purchase",
    *,
    attempt: int = 0,
    status: str = "done",
    reward: Any = _UNSET,
    reward_valid: bool = True,
    env_steps: int = 5,
    rejections: int = 0,
) -> dict:
    """造一条轨迹记录，形状与 Trajectory.as_record() 一致。"""
    audit: dict = {}
    if reward_type is not None:
        audit["terminal"] = {
            "reward_detail": {
                "reward_type": reward_type,
                "reward_valid": reward_valid,
                "reward_version": "shopsimulator-reward-v3",
            }
        }
    return {
        "trajectory_id": f"t{task_id}-{attempt}",
        "task_id": task_id,
        "attempt": attempt,
        "status": status,
        "done": status == "done",
        # reward=None 是「环境明确没给分」，与不传 reward（按终局类型取默认值）不同。
        "reward": REWARD_VALUES.get(reward_type or "", 0.0) if reward is _UNSET else reward,
        "env_steps": env_steps,
        "rejection_count": rejections,
        "messages": [],
        "steps": [],
        "rejections": [],
        "audit": audit,
        "error": None,
    }


def outcomes(*specs: tuple[int, str]) -> list[Outcome]:
    return [outcome_from_record(record(task_id, kind)) for task_id, kind in specs]


# --- 终局标签的来源 ---------------------------------------------------------


def test_reward_type_comes_from_audit_not_from_status():
    """status=done 只说明回合结束了，没说买对了。标签必须来自环境。"""
    outcome = outcome_from_record(record(1, "wrong_purchase", status="done"))
    assert outcome.reward_type == "wrong_purchase"
    assert not outcome.success
    assert outcome.wrong_purchase


def test_missing_terminal_falls_back_to_a_labelled_status():
    """回合没正常结束时环境不给标签，兜底值要能在报告里认出来是哪种。"""
    outcome = outcome_from_record(record(1, None, status="no_tool_call"))
    assert outcome.reward_type == "no_terminal:no_tool_call"
    assert not outcome.success
    assert not outcome.wrong_purchase


@pytest.mark.parametrize("kind", sorted(REWARD_VALUES))
def test_only_gold_and_valid_alternative_count_as_success(kind: str):
    outcome = outcome_from_record(record(1, kind))
    assert outcome.success is (kind in SUCCESS_TYPES)


def test_partial_alternative_purchase_is_not_success():
    """硬门过了但软匹配没满——买到的不是用户要的东西，不能算成功。"""
    assert not outcome_from_record(record(1, "partial_alternative_purchase")).success


def test_only_gold_purchase_counts_as_gold():
    assert outcome_from_record(record(1, "gold_purchase")).gold
    assert not outcome_from_record(record(1, "valid_alternative_purchase")).gold


def test_invalid_reward_is_flagged_and_dropped_from_the_main_metrics():
    records = [
        record(1, "gold_purchase"),
        record(2, "reward_unverifiable", reward_valid=False),
    ]
    summary = summarize([outcome_from_record(r) for r in records])
    assert summary["dropped_invalid_reward"] == 1
    # 只剩 task 1，成功率 1.0；若把不可信那条算进去会变成 0.5。
    assert summary["success_rate"]["mean"] == 1.0
    # 但终局类型分布仍然报全部，否则读者看不到被丢了什么。
    assert summary["reward_types"]["reward_unverifiable"] == 1


# --- bootstrap ------------------------------------------------------------


def test_bootstrap_ci_brackets_the_mean():
    values = [1.0] * 30 + [0.0] * 70
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo < 0.3 < hi
    assert 0.0 <= lo and hi <= 1.0


def test_bootstrap_ci_stays_inside_zero_one_for_a_low_rate():
    """成功率 0.02、50 题：正态近似会给出负的下界，bootstrap 不会。"""
    values = [1.0] + [0.0] * 49
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo >= 0.0
    assert hi > 0.0


def test_bootstrap_ci_is_deterministic_given_a_seed():
    values = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)
    assert bootstrap_ci(values, seed=7) != bootstrap_ci(values, seed=8)


def test_bootstrap_ci_on_a_single_sample_is_a_zero_width_interval():
    assert bootstrap_ci([0.4]) == (0.4, 0.4)


def test_bootstrap_ci_on_no_samples_is_nan():
    lo, hi = bootstrap_ci([])
    assert math.isnan(lo) and math.isnan(hi)


def test_a_wider_confidence_level_gives_a_wider_interval():
    values = [1.0] * 20 + [0.0] * 30
    narrow = bootstrap_ci(values, confidence=0.80, seed=3)
    wide = bootstrap_ci(values, confidence=0.99, seed=3)
    assert wide[0] <= narrow[0] and narrow[1] <= wide[1]


# --- 按题聚合 -------------------------------------------------------------


def test_k_samples_of_one_task_do_not_count_as_k_independent_samples():
    """一题采 8 次，与 8 题各采 1 次，成功率相同但独立观测数不同。

    构造成一半成功一半失败，这样区间才有宽度可比。按题聚合时前者只有 1 个观测、
    区间零宽；若实现把 k×N 条当独立样本，前者会得到和后者一样的非零宽区间。
    """
    one_task = [
        outcome_from_record(
            record(1, "gold_purchase" if i % 2 else "wrong_purchase", attempt=i)
        )
        for i in range(8)
    ]
    eight_tasks = [
        outcome_from_record(record(i, "gold_purchase" if i % 2 else "wrong_purchase"))
        for i in range(8)
    ]

    single, spread = summarize(one_task), summarize(eight_tasks)
    assert single["tasks"] == 1 and spread["tasks"] == 8
    assert single["attempts_per_task"] == 8.0
    assert single["success_rate"]["mean"] == spread["success_rate"]["mean"] == 0.5

    def width(summary: dict) -> float:
        item = summary["success_rate"]
        return item["ci_high"] - item["ci_low"]

    # 1 题就是 1 个观测，bootstrap 只能重采出同一个值。
    assert width(single) == 0.0
    assert width(spread) > 0.0


def test_per_task_averaging_weights_tasks_equally():
    """题 1 采 4 次全对，题 2 采 1 次错。按题平均是 0.5，按轨迹平均会是 0.8。"""
    records = [record(1, "gold_purchase", attempt=i) for i in range(4)]
    records.append(record(2, "wrong_purchase"))
    summary = summarize([outcome_from_record(r) for r in records])
    assert summary["success_rate"]["mean"] == 0.5


def test_partial_success_within_a_task_lands_between_zero_and_one():
    records = [
        record(1, "gold_purchase", attempt=0),
        record(1, "wrong_purchase", attempt=1),
    ]
    summary = summarize([outcome_from_record(r) for r in records])
    assert summary["success_rate"]["mean"] == 0.5
    assert summary["wrong_purchase_rate"]["mean"] == 0.5


# --- summarize 的其余输出 --------------------------------------------------


def test_summarize_on_no_trajectories_does_not_divide_by_zero():
    assert summarize([]) == {"trajectories": 0, "tasks": 0}


def test_mean_reward_is_none_when_no_trajectory_carries_a_reward():
    records = [record(1, None, status="max_steps", reward=None)]
    summary = summarize([outcome_from_record(r) for r in records])
    assert summary["mean_reward"] is None
    # 但成功率照样能算——没 reward 就是没成功。
    assert summary["success_rate"]["mean"] == 0.0


def test_reward_types_are_sorted_by_frequency():
    records = [record(i, "wrong_purchase") for i in range(5)]
    records += [record(10 + i, "gold_purchase") for i in range(2)]
    summary = summarize([outcome_from_record(r) for r in records])
    assert list(summary["reward_types"]) == ["wrong_purchase", "gold_purchase"]


def test_statuses_are_reported_separately_from_reward_types():
    """基础设施失败和模型表现差要能分开看。"""
    records = [
        record(1, "gold_purchase", status="done"),
        record(2, None, status="rejection_limit"),
        record(3, None, status="no_tool_call"),
    ]
    summary = summarize([outcome_from_record(r) for r in records])
    assert summary["statuses"]["done"] == 1
    assert set(summary["statuses"]) == {"done", "rejection_limit", "no_tool_call"}


# --- 分层 -----------------------------------------------------------------


def test_stratified_splits_by_the_given_map():
    items = outcomes(*[(i, "gold_purchase") for i in range(12)],
                     *[(100 + i, "wrong_purchase") for i in range(12)])
    strata = {i: "1-2" for i in range(12)} | {100 + i: "8+" for i in range(12)}
    report = stratified(items, strata)
    assert report["1-2"]["success_rate"]["mean"] == 1.0
    assert report["8+"]["success_rate"]["mean"] == 0.0


def test_small_buckets_are_reported_but_flagged_underpowered():
    """3 题的 100% 成功率必须带警告，否则会被当成真的。"""
    items = outcomes((1, "gold_purchase"), (2, "gold_purchase"), (3, "gold_purchase"))
    report = stratified(items, {1: "tiny", 2: "tiny", 3: "tiny"}, min_tasks=10)
    assert report["tiny"]["success_rate"]["mean"] == 1.0
    assert report["tiny"]["underpowered"] is True


def test_tasks_missing_from_the_strata_map_land_in_unknown():
    report = stratified(outcomes((1, "gold_purchase")), {})
    assert "unknown" in report


def test_load_strata_reads_difficulty_and_domain_from_the_pool(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text(
        json.dumps({"task_id": 7, "difficulty": "3-4", "domain": "Supplies"}) + "\n"
        + json.dumps({"task_id": 9, "difficulty": "8+", "domain": "Beauty"}) + "\n",
        encoding="utf-8",
    )
    assert load_strata(path, "difficulty") == {7: "3-4", 9: "8+"}
    assert load_strata(path, "domain") == {7: "Supplies", 9: "Beauty"}


def test_load_strata_names_the_available_fields_when_asked_for_a_missing_one(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text(json.dumps({"task_id": 7, "difficulty": "3-4"}) + "\n", encoding="utf-8")
    with pytest.raises(KeyError, match="difficulty"):
        load_strata(path, "domain")


def test_load_strata_works_on_the_real_evaluation_pool():
    from pathlib import Path

    pool = Path(__file__).resolve().parents[1] / "data" / "task_pools" / "evaluation.jsonl"
    if not pool.exists():
        pytest.skip("任务池未生成")
    for field in ("difficulty", "domain"):
        strata = load_strata(pool, field)
        assert len(strata) == 500
        assert all(isinstance(k, int) and v for k, v in strata.items())


# --- 配对比较 -------------------------------------------------------------


def test_paired_comparison_measures_the_delta_on_shared_tasks():
    baseline = outcomes(*[(i, "wrong_purchase") for i in range(20)])
    treatment = outcomes(*[(i, "gold_purchase") for i in range(20)])
    result = paired_comparison(baseline, treatment)
    assert result["paired_tasks"] == 20
    assert result["baseline_success_rate"] == 0.0
    assert result["treatment_success_rate"] == 1.0
    assert result["delta"] == 1.0
    assert result["significant"] is True


def test_paired_comparison_ignores_tasks_present_on_only_one_side():
    baseline = outcomes((1, "wrong_purchase"), (2, "wrong_purchase"))
    treatment = outcomes((2, "gold_purchase"), (3, "gold_purchase"))
    result = paired_comparison(baseline, treatment)
    assert result["paired_tasks"] == 1
    assert result["baseline_only_tasks"] == 1
    assert result["treatment_only_tasks"] == 1


def test_paired_comparison_with_no_shared_tasks_says_so_instead_of_crashing():
    result = paired_comparison(outcomes((1, "gold_purchase")), outcomes((2, "gold_purchase")))
    assert result["paired_tasks"] == 0
    assert "note" in result


def test_a_no_op_change_is_not_reported_as_significant():
    """两侧完全一样时区间必须跨 0，否则我们会把噪声当成改进。"""
    both = [(i, "gold_purchase" if i % 3 else "wrong_purchase") for i in range(30)]
    result = paired_comparison(outcomes(*both), outcomes(*both))
    assert result["delta"] == 0.0
    assert result["significant"] is False
    assert result["ci_low"] <= 0.0 <= result["ci_high"]


def test_a_small_mixed_improvement_is_not_called_significant():
    """30 题里净赢 1 题——区间该跨 0。"""
    baseline = [(i, "wrong_purchase") for i in range(30)]
    treatment = [(i, "wrong_purchase") for i in range(29)] + [(29, "gold_purchase")]
    result = paired_comparison(outcomes(*baseline), outcomes(*treatment))
    assert result["delta"] == pytest.approx(1 / 30, abs=1e-4)
    assert result["significant"] is False


def test_paired_comparison_drops_invalid_reward_trajectories():
    baseline = [outcome_from_record(record(1, "wrong_purchase"))]
    treatment = [
        outcome_from_record(record(1, "gold_purchase")),
        outcome_from_record(record(1, "reward_unverifiable", attempt=1, reward_valid=False)),
    ]
    result = paired_comparison(baseline, treatment)
    # 不可信那条被丢掉，剩下 1/1 成功，而不是 1/2。
    assert result["treatment_success_rate"] == 1.0


def test_a_regression_is_also_flagged_significant():
    baseline = outcomes(*[(i, "gold_purchase") for i in range(20)])
    treatment = outcomes(*[(i, "wrong_purchase") for i in range(20)])
    result = paired_comparison(baseline, treatment)
    assert result["delta"] == -1.0
    assert result["significant"] is True


# --- IO 与格式化 ----------------------------------------------------------


def test_load_outcomes_reads_a_trajectory_file(tmp_path):
    path = tmp_path / "traj.jsonl"
    path.write_text(
        json.dumps(record(1, "gold_purchase"), ensure_ascii=False) + "\n"
        + "\n"  # 空行要能跳过
        + json.dumps(record(2, "wrong_purchase"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    loaded = load_outcomes(path)
    assert [o.task_id for o in loaded] == [1, 2]
    assert loaded[0].success and not loaded[1].success


def test_format_summary_shows_the_main_rates_and_intervals():
    items = outcomes(*[(i, "gold_purchase") for i in range(6)],
                     *[(10 + i, "wrong_purchase") for i in range(4)])
    text = format_summary(summarize(items))
    assert "成功率" in text and "错买率" in text
    assert "95% CI" in text
    assert "gold_purchase" in text and "wrong_purchase" in text
    # 百分比要加起来是 100
    assert "60.0%" in text and "40.0%" in text


def test_format_summary_on_no_trajectories():
    assert format_summary(summarize([])) == "没有轨迹"


def test_format_summary_reports_dropped_trajectories():
    items = [
        outcome_from_record(record(1, "gold_purchase")),
        outcome_from_record(record(2, "reward_unverifiable", reward_valid=False)),
    ]
    assert "不可信" in format_summary(summarize(items))


def test_format_summary_handles_a_missing_mean_reward():
    items = [outcome_from_record(record(1, None, status="max_steps", reward=None))]
    text = format_summary(summarize(items))
    assert "平均 reward" in text  # 显示为占位而不是抛异常
