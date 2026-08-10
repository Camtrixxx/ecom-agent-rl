"""把教师轨迹筛成 SFT 训练集。

筛选是纯白名单 AND 逻辑，没有 reward 数值阈值：一条轨迹要么每项都过，要么丢弃并
记下全部拒绝原因。用「全部原因」而不是「第一个原因」是为了能回答「如果放宽某一项
能多收多少」——调口径时这个分布比总数有用得多。

## 与参考实现的两处关键差别

**不需要逐步重放校验。** 参考实现的接受率 0.41 主要不是被 reward_type 卡掉的：raw
里 `gold_purchase` 有 1742 条而 accepted 只有 1026，差额 716 全是 `guard_violation`
——它把违规动作和正常动作混存在 `steps` 里，必须逐步重算 guard 才能找出来。我们的
rollout 层在写盘时就把被拒动作单独存进 `rejections`，`steps` 里只有执行成功的动作，
所以 `rejection_count` 直接就是答案，无需重放。这也意味着我们不会因为重放逻辑与采集
时的 guard 版本不一致而误杀轨迹。

**接受口径可配置。** 参考实现硬编码只收 `gold_purchase`，而我们的评测口径是
`gold_purchase` + `valid_alternative_purchase`（见 evaluation/metrics.py 的
SUCCESS_TYPES）。两者不一致会导致「训练时告诉模型只有买到 gold 才算对，评测时却给
0.55 分」。哪个口径更好是实证问题，所以做成参数，两份都产出、当 ablation 对照。

## 为什么拒绝原因要分组记

`rejection_count > 0` 与 `reward_type != gold_purchase` 是两个独立的失败轴。前者是
「过程脏」（教师点了当前页不允许点的东西，虽然最后买对了），后者是「结果错」。分开
统计才能判断该放宽哪一边。
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# 评测口径（evaluation/metrics.py 的 SUCCESS_TYPES）。训练与评测同口径的默认值。
SUCCESS_TYPES = frozenset({"gold_purchase", "valid_alternative_purchase"})

# 参考实现的口径：只收满分轨迹。用于 ablation 对照。
GOLD_ONLY = frozenset({"gold_purchase"})

# 允许的被拒动作次数上限。
#
# 实测（教师试采）：阈值 0 时接受率 0%，放到 2 跳到 36.4%，再往上到 12 完全不变
# ——所有成功轨迹的被拒次数都 ≤ 2。所以这不是在噪声里挑阈值，2 是曲线的平台起点。
# 参考实现的 41% 接受率卡的是同一件事（它 716 条 guard_violation），量级也吻合。
#
# 为什么不设 0：教师买对了商品、只是中途点错一两下，把这类全丢会让接受率归零。
# 为什么不设很大：被拒次数多说明教师在乱点，那条轨迹的中间推理不值得学。
DEFAULT_MAX_REJECTIONS = 2

# 训练样本里只保留这些字段。教师的 reasoning_content 等一律剥掉——学生学的是动作，
# 不是教师的思维链（而且不同模型的 reasoning 格式不通用）。
_MESSAGE_FIELDS = frozenset({"role", "content", "tool_calls", "tool_call_id", "name"})


@dataclass(frozen=True)
class Verdict:
    """一条轨迹的筛选结论。`reasons` 为空即接受。"""

    trajectory_id: str
    task_id: int
    accepted: bool
    reasons: tuple[str, ...] = ()
    reward_type: str = ""
    steps: int = 0
    rejection_count: int = 0


@dataclass
class Report:
    """一批轨迹的筛选审计。"""

    total: int = 0
    accepted: int = 0
    # 每条被丢弃的轨迹会给所有原因各记一次，所以总和 > 被丢弃数。
    reason_counts: Counter = field(default_factory=Counter)
    reward_types: Counter = field(default_factory=Counter)
    duplicate_tasks: int = 0
    held_out: int = 0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.total if self.total else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "duplicate_tasks_excluded": self.duplicate_tasks,
            "held_out_excluded": self.held_out,
            "reason_counts": dict(self.reason_counts.most_common()),
            "reward_types": dict(self.reward_types.most_common()),
        }


def _reward_detail(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    """终局判定在 audit 里——评测隔离把答案字段在写盘前就剥离到那儿了。"""
    terminal = (trajectory.get("audit") or {}).get("terminal") or {}
    return terminal.get("reward_detail") or {}


def _terminal(trajectory: Mapping[str, Any]) -> Mapping[str, Any]:
    return (trajectory.get("audit") or {}).get("terminal") or {}


def acceptance_reasons(
    trajectory: Mapping[str, Any],
    *,
    accept_types: frozenset[str] = SUCCESS_TYPES,
    max_rejections: int = DEFAULT_MAX_REJECTIONS,
    require_purchase: bool = True,
) -> tuple[bool, list[str]]:
    """返回 (是否接受, 全部拒绝原因)。

    `max_rejections` 是过程洁癖的旋钮。0 表示全程无被拒动作——实测这会把教师数据
    筛空（见 DEFAULT_MAX_REJECTIONS 的说明）。被拒动作本身不会进训练样本
    （`training_messages` 会剔除），所以放宽它留下的是「教师走过弯路但删掉了弯路」
    的示范，而非教会学生违规。
    """
    reasons: list[str] = []
    steps = trajectory.get("steps") or []
    detail = _reward_detail(trajectory)
    terminal = _terminal(trajectory)

    if trajectory.get("error"):
        reasons.append("has_error")
    if trajectory.get("status") != "done":
        reasons.append(f"status_not_done:{trajectory.get('status')}")
    if trajectory.get("done") is not True:
        reasons.append("trajectory_not_done")

    reward_type = str(detail.get("reward_type") or "")
    if not reward_type:
        reasons.append("no_reward_type")
    elif reward_type not in accept_types:
        reasons.append(f"reward_type_not_accepted:{reward_type}")

    # reward_valid 是环境对「这条 reward 能不能信」的自评。不可信的轨迹即使
    # reward_type 好看也不能用——它可能是判定器自己没算出来。
    if terminal.get("reward_valid") is not True:
        reasons.append("reward_not_valid")

    if require_purchase and not any(
        step.get("tool_name") == "buy_now" for step in steps
    ):
        reasons.append("missing_buy")

    count = int(trajectory.get("rejection_count") or 0)
    if count > max_rejections:
        reasons.append(f"rejections:{count}")

    # 每个 assistant 回合只能有一个 tool_call：多个调用是基于同一个旧 observation
    # 生成的，训练时会教模型并发点击一个已经变了的页面。rollout 层已经只执行第一个
    # 并把其余记进 dropped_tool_calls，这里确认一下没有漏网的。
    for index, message in enumerate(trajectory.get("messages") or []):
        if message.get("role") == "assistant" and len(message.get("tool_calls") or []) > 1:
            reasons.append(f"message_{index}:multiple_tool_calls")
    if any(step.get("dropped_tool_calls") for step in steps):
        reasons.append("dropped_tool_calls")

    if not steps:
        reasons.append("no_steps")

    return not reasons, reasons


def _sanitize_message(message: Mapping[str, Any]) -> dict[str, Any]:
    """字段白名单，并把 tool_call 压到训练需要的最小形状。"""
    out = {k: v for k, v in message.items() if k in _MESSAGE_FIELDS}
    calls = out.get("tool_calls")
    if calls:
        out["tool_calls"] = [
            {
                "id": c.get("id"),
                "type": "function",
                "function": {
                    "name": (c.get("function") or {}).get("name"),
                    "arguments": (c.get("function") or {}).get("arguments"),
                },
            }
            for c in calls
        ]
    return out


def _rejected_call_ids(trajectory: Mapping[str, Any]) -> set[str]:
    ids = set()
    for rejection in trajectory.get("rejections") or []:
        call_id = (rejection.get("tool_call") or {}).get("id")
        if call_id:
            ids.add(str(call_id))
    return ids


def training_messages(trajectory: Mapping[str, Any]) -> list[dict[str, Any]]:
    """剔掉被拒绝的动作及其错误回复，只留成功路径。

    学生不该学教师的违规尝试——那些动作在当时的页面上是非法的，模仿它们只会
    制造 rejection。被拒动作改变了后续 observation 这件事无需担心：guard 拒绝时
    环境状态没变（动作根本没执行）。
    """
    rejected = _rejected_call_ids(trajectory)
    out: list[dict[str, Any]] = []
    skip_tool_ids: set[str] = set()
    for message in trajectory.get("messages") or []:
        calls = message.get("tool_calls") or []
        if message.get("role") == "assistant" and calls:
            if any(str(c.get("id")) in rejected for c in calls):
                # 这条 assistant 消息的工具调用被拒了，它和它的错误回复都不进训练集。
                skip_tool_ids.update(str(c.get("id")) for c in calls)
                continue
        if message.get("role") == "tool" and str(message.get("tool_call_id")) in skip_tool_ids:
            continue
        out.append(_sanitize_message(message))
    return out


def build_row(trajectory: Mapping[str, Any], tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """一条轨迹 → 一个多轮训练样本。

    整条 episode 作一个样本而不是按步拆开：拆开会丢掉「第 5 步为什么这么选」依赖
    前 4 步 observation 的信息，而这正是长程任务要学的东西。
    """
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "task_id": trajectory.get("task_id"),
        "messages": training_messages(trajectory),
        "tools": [dict(t) for t in tools],
    }


def select(
    trajectories: Iterable[Mapping[str, Any]],
    *,
    accept_types: frozenset[str] = SUCCESS_TYPES,
    max_rejections: int = DEFAULT_MAX_REJECTIONS,
    held_out_task_ids: frozenset[int] = frozenset(),
    dedupe_by_task: bool = True,
) -> tuple[list[Verdict], Report]:
    """筛一批轨迹。返回 (每条的结论, 审计报告)。

    去重与 held-out 排除放在质量判据**之后**：先按质量筛，再从合格者里挑，这样
    报告里的接受率反映的是教师的真实质量，不被切分策略污染。
    """
    verdicts: list[Verdict] = []
    report = Report()
    seen_tasks: set[int] = set()

    for trajectory in trajectories:
        report.total += 1
        detail = _reward_detail(trajectory)
        reward_type = str(detail.get("reward_type") or f"no_terminal:{trajectory.get('status')}")
        report.reward_types[reward_type] += 1

        ok, reasons = acceptance_reasons(
            trajectory, accept_types=accept_types, max_rejections=max_rejections
        )
        task_id = int(trajectory.get("task_id", -1))

        if ok:
            if task_id in held_out_task_ids:
                ok, reasons = False, ["held_out_task"]
                report.held_out += 1
            elif dedupe_by_task and task_id in seen_tasks:
                ok, reasons = False, ["duplicate_task"]
                report.duplicate_tasks += 1
            else:
                seen_tasks.add(task_id)

        if ok:
            report.accepted += 1
        else:
            for reason in reasons:
                report.reason_counts[reason] += 1

        verdicts.append(
            Verdict(
                trajectory_id=str(trajectory.get("trajectory_id") or ""),
                task_id=task_id,
                accepted=ok,
                reasons=tuple(reasons),
                reward_type=reward_type,
                steps=len(trajectory.get("steps") or []),
                rejection_count=int(trajectory.get("rejection_count") or 0),
            )
        )

    return verdicts, report


def read_trajectories(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
