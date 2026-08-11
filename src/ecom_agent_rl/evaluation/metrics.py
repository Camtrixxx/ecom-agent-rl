"""从轨迹算指标，带分层与置信区间。

设计上的三个决定：

1. **主指标是成功率，不是平均 reward。** Reward v3 的取值从 -0.85 到 1.0，且
   `partial_alternative_purchase` 是连续的，平均 reward 会把「买错了」和「没买」
   混成一个数。成功率（`gold_purchase` + `valid_alternative_purchase`）语义清楚，
   `wrong_purchase` 率单独报——它是最该压下去的失败模式。

2. **置信区间用 bootstrap，不用正态近似。** 500 题、成功率可能低到 0.1 的量级下，
   Wald 区间会给出越界的下界。分层报告里某些桶只有几十条，更需要 bootstrap。

3. **每题多次采样时先按题内取平均，再对题求 bootstrap。** 同一题的 k 次采样不独立，
   直接把 k×N 条当独立样本会把区间算窄。

`reward_detail.reward_type` 是环境给的权威终局标签，比我们自己从 messages 猜要可靠，
它在 `audit.terminal` 里（不在模型可见面）。
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Reward v3 的终局类型与取值，抄自 web_agent_site/engine/reward.py 的 DEFAULT_REWARDS。
# 这里重复一遍是为了让指标代码能独立解释轨迹，改环境时测试会发现不一致。
REWARD_VALUES: dict[str, float] = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_alternative_purchase": 0.25,  # 实际是 min(0.25, -0.30 + 0.55*score)，上界
    "reward_unverifiable": 0.0,
    "graceful_stop": -0.15,
    "early_abstain": -0.35,
    "max_steps": -0.50,
    "repeat_loop": -0.65,
    "wrong_purchase": -0.85,
}

# 算成功的终局。partial 不算：它意味着硬门过了但软匹配没满，买到的不是用户要的东西。
SUCCESS_TYPES = frozenset({"gold_purchase", "valid_alternative_purchase"})


@dataclass(frozen=True)
class Outcome:
    """一条轨迹的评测结论。"""

    task_id: int
    attempt: int
    status: str
    reward_type: str
    reward: float | None
    env_steps: int
    rejections: int
    reward_valid: bool

    @property
    def success(self) -> bool:
        return self.reward_type in SUCCESS_TYPES

    @property
    def gold(self) -> bool:
        return self.reward_type == "gold_purchase"

    @property
    def wrong_purchase(self) -> bool:
        return self.reward_type == "wrong_purchase"


def outcome_from_record(record: Mapping[str, Any]) -> Outcome:
    """从轨迹记录抽出评测结论。

    `reward_type` 是环境的权威标签。回合没正常结束（模型不调工具、被拒判死、跑满
    步数）时环境没给标签，用轨迹 status 兜底，这些都算失败，但要能在报告里和
    「环境判定的失败」区分开。

    优先读轨迹顶层，回落到 `audit.terminal.reward_detail`：顶层字段是后加的，
    `outputs/rollouts/` 里已有的 baseline / SFT 轨迹没有它，两条路必须给出同一个
    结论，否则同一份数据换个读法就换个数。
    """
    audit = record.get("audit") or {}
    terminal = audit.get("terminal") or {}
    detail = terminal.get("reward_detail") or {}
    reward_type = str(
        record.get("reward_type")
        or detail.get("reward_type")
        or f"no_terminal:{record.get('status')}"
    )
    return Outcome(
        task_id=int(record["task_id"]),
        attempt=int(record.get("attempt", 0)),
        status=str(record.get("status") or "unknown"),
        reward_type=reward_type,
        reward=(
            float(record["reward"]) if isinstance(record.get("reward"), (int, float)) else None
        ),
        env_steps=int(record.get("env_steps") or 0),
        rejections=int(record.get("rejection_count") or 0),
        # reward_valid=False 表示环境自己认为这次打分不可信，不该计入主指标。
        reward_valid=bool(
            record.get("reward_valid", detail.get("reward_valid", True))
        ),
    )


def load_outcomes(path: Path) -> list[Outcome]:
    outcomes: list[Outcome] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                outcomes.append(outcome_from_record(json.loads(line)))
    return outcomes


def load_strata(path: Path, field: str) -> dict[int, str]:
    """从任务池读 task_id → 分层名。

    分层轴不重算：`build_task_pools.py` 已经把 `difficulty`（attributes 桶）和
    `domain` 写进池文件，重算一遍等于给同一件事留两份实现。
    """
    strata: dict[int, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if field not in record:
                raise KeyError(
                    f"任务池 {path.name} 没有字段 {field!r}，可用字段: {sorted(record)}"
                )
            strata[int(record["task_id"])] = str(record[field])
    return strata


def bootstrap_ci(
    values: Sequence[float], *, confidence: float = 0.95, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """对均值做 percentile bootstrap。

    样本少于 2 条时区间无意义，直接返回该点本身——报告里会显示成零宽区间，
    比返回 NaN 更容易读。
    """
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        return (float(values[0]), float(values[0]))
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(resamples):
        means.append(sum(values[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = (1.0 - confidence) / 2.0
    return (
        means[max(0, int(lo * resamples) - 1)],
        means[min(resamples - 1, int((1.0 - lo) * resamples))],
    )


def _per_task_means(outcomes: Iterable[Outcome], value: Any) -> list[float]:
    """先按题内取平均，再返回每题一个数。

    同一题的 k 次采样不独立，把 k×N 条当独立样本会低估区间宽度。
    """
    grouped: dict[int, list[float]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.task_id].append(float(value(outcome)))
    return [sum(v) / len(v) for v in grouped.values()]


def summarize(
    outcomes: Sequence[Outcome], *, confidence: float = 0.95, seed: int = 0
) -> dict[str, Any]:
    """主指标 + 置信区间 + 终局类型分布。"""
    if not outcomes:
        return {"trajectories": 0, "tasks": 0}

    valid = [o for o in outcomes if o.reward_valid]
    dropped = len(outcomes) - len(valid)

    def ci(value: Any, source: Sequence[Outcome] = valid) -> dict[str, Any]:
        per_task = _per_task_means(source, value)
        lo, hi = bootstrap_ci(per_task, confidence=confidence, seed=seed)
        mean = sum(per_task) / len(per_task) if per_task else float("nan")
        return {"mean": round(mean, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)}

    rewarded = [o for o in valid if o.reward is not None]
    types: dict[str, int] = defaultdict(int)
    for outcome in outcomes:
        types[outcome.reward_type] += 1

    return {
        "trajectories": len(outcomes),
        "tasks": len({o.task_id for o in outcomes}),
        "attempts_per_task": round(len(outcomes) / len({o.task_id for o in outcomes}), 2),
        "dropped_invalid_reward": dropped,
        "success_rate": ci(lambda o: o.success),
        "gold_rate": ci(lambda o: o.gold),
        "wrong_purchase_rate": ci(lambda o: o.wrong_purchase),
        # 只对真的有分的轨迹算平均。没分的（跑满步数、模型不调工具）不该被当成 0——
        # 0 在 Reward v3 里是 reward_unverifiable 的取值，有具体含义。
        "mean_reward": ci(lambda o: o.reward, rewarded) if rewarded else None,
        "mean_env_steps": ci(lambda o: o.env_steps),
        "mean_rejections": ci(lambda o: o.rejections),
        "reward_types": dict(sorted(types.items(), key=lambda kv: -kv[1])),
        "statuses": dict(
            sorted(
                ((s, sum(1 for o in outcomes if o.status == s))
                 for s in {o.status for o in outcomes}),
                key=lambda kv: -kv[1],
            )
        ),
    }


def stratified(
    outcomes: Sequence[Outcome],
    strata: Mapping[int, str],
    *,
    confidence: float = 0.95,
    seed: int = 0,
    min_tasks: int = 10,
) -> dict[str, Any]:
    """按给定分层（难度桶或品类）分别汇总。

    `min_tasks` 以下的桶照样报，但标 `underpowered`——不标出来，读者会把 3 题的
    100% 成功率当成真的。
    """
    grouped: dict[str, list[Outcome]] = defaultdict(list)
    for outcome in outcomes:
        grouped[strata.get(outcome.task_id, "unknown")].append(outcome)

    report: dict[str, Any] = {}
    for name in sorted(grouped):
        summary = summarize(grouped[name], confidence=confidence, seed=seed)
        summary["underpowered"] = summary["tasks"] < min_tasks
        report[name] = summary
    return report


def paired_comparison(
    baseline: Sequence[Outcome],
    treatment: Sequence[Outcome],
    *,
    confidence: float = 0.95,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """按 task_id 配对比较成功率。

    配对而非独立两样本：两侧跑的是同一批题，配对能消掉题目难度的方差，这是同样
    题量下能拿到的最紧区间。只比较两边都有的 task_id。
    """
    def by_task(outcomes: Sequence[Outcome]) -> dict[int, list[Outcome]]:
        grouped: dict[int, list[Outcome]] = defaultdict(list)
        for outcome in outcomes:
            if outcome.reward_valid:
                grouped[outcome.task_id].append(outcome)
        return grouped

    left, right = by_task(baseline), by_task(treatment)
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"paired_tasks": 0, "note": "两侧没有共同的 task_id，无法配对比较"}

    def rate(items: Sequence[Outcome]) -> float:
        return sum(1.0 for o in items if o.success) / len(items)

    deltas = [rate(right[t]) - rate(left[t]) for t in shared]
    lo, hi = bootstrap_ci(deltas, confidence=confidence, resamples=resamples, seed=seed)
    mean = sum(deltas) / len(deltas)
    return {
        "paired_tasks": len(shared),
        "baseline_success_rate": round(
            sum(rate(left[t]) for t in shared) / len(shared), 4
        ),
        "treatment_success_rate": round(
            sum(rate(right[t]) for t in shared) / len(shared), 4
        ),
        "delta": round(mean, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        # 区间不跨 0 才算这个改动真的有效果。
        "significant": lo > 0.0 or hi < 0.0,
        "baseline_only_tasks": len(set(left) - set(right)),
        "treatment_only_tasks": len(set(right) - set(left)),
    }


def format_summary(summary: Mapping[str, Any]) -> str:
    """给人看的一段文本。"""
    if not summary.get("trajectories"):
        return "没有轨迹"

    def line(label: str, key: str) -> str:
        item = summary.get(key)
        if not item:
            return f"{label:<16} —"
        return (
            f"{label:<16} {item['mean']:>8.4f}  "
            f"[{item['ci_low']:.4f}, {item['ci_high']:.4f}]"
        )

    lines = [
        f"轨迹 {summary['trajectories']} 条，题目 {summary['tasks']} 个"
        f"（每题 {summary['attempts_per_task']} 次）",
    ]
    if summary.get("dropped_invalid_reward"):
        lines.append(f"环境标记 reward 不可信而排除: {summary['dropped_invalid_reward']} 条")
    lines.append("")
    lines.append(f"{'指标':<16} {'均值':>8}  {'95% CI':^18}")
    lines.append("-" * 46)
    for label, key in (
        ("成功率", "success_rate"),
        ("gold 率", "gold_rate"),
        ("错买率", "wrong_purchase_rate"),
        ("平均 reward", "mean_reward"),
        ("平均步数", "mean_env_steps"),
        ("平均被拒次数", "mean_rejections"),
    ):
        lines.append(line(label, key))
    lines.append("")
    lines.append("终局类型:")
    total = summary["trajectories"]
    for name, count in summary["reward_types"].items():
        lines.append(f"  {name:<34} {count:>5}  {count / total * 100:>5.1f}%")
    return "\n".join(lines)
