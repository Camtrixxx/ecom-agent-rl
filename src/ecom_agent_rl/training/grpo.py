"""GRPO 的纯逻辑层：回报整形、组内优势、把轨迹渲染成带优势的训练样本。

放在这里而不是脚本里，是因为这三件事都能在没有 GPU、没有环境、没有模型的情况下
单测，而它们恰恰是最容易静默算错的部分。

## 1. 回报整形：三类结局要分开处理，不能都当成 0

轨迹里的 `reward` 有三种缺失方式，混为一谈会让训练学到相反的东西：

| 情况 | `reward_valid` | `reward` | 这里怎么处理 |
|---|---|---|---|
| 正常终局 | True | 数值 | 直接用 |
| 环境判不了（`reward_unverifiable`） | **False** | None | **剔除**，不进梯度也不进基线 |
| 没走到终局（不发工具调用、撞上下文上限…） | True | None | 给 `invalid_reward` 地板分 |

中间那行是关键。`reward_unverifiable` 在环境里取值 0.0，若照搬这个 0.0，它在一个
平均回报为负的组里会变成**正优势**——模型会被奖励去触发「判不了」。这不是策略失败
而是数据问题，正确的做法是从组里拿掉。上游为此把顶层 `reward` 置成 None，让「把
0.0 当成真值」在数值上就不可能发生；这里承接同一个约定。

最后一行相反：没走到终局是模型自己把回合浪费掉了，是**策略失败**，必须罚。默认罚
`-1.0`，严格低于环境里最差的合法结局（`wrong_purchase` −0.85），语义是「连一个合法
结局都没产出，比买错还差」。这一档在 SFT 权重上只占 ~0.5% 的回合，取值不敏感，但
不能不给——不给（当成缺失丢掉）等于告诉模型「不发动作就不会被罚」。

## 2. 优势只减组均值，不除标准差

标准 GRPO 是 `(r - mean) / (std + eps)`。这里**不除 std**（Dr. GRPO 的修正）：除以
std 会给「组内方差小」的题更大的梯度权重，而组内方差小恰恰意味着这题没什么可学的
（全组同一个结局）。除法把它放大成了最陡的方向，是一个纯粹的偏置。

组内有效样本 < 2 的组整组丢弃：一个样本算不出基线，`r - r = 0` 不是「没有优势」而是
「没有信息」。全组优势都为 0 的组也丢弃——梯度恒为 0，白跑一次前反向。

## 3. 渲染失败的轨迹仍然参与基线

超长（渲染后 > `max_length`）的轨迹进不了梯度，但**它的回报仍然计入组均值**。这是
有意的：基线是对这道题期望回报的估计，它该用上全部样本；某条样本没有梯度项，不代表
它不该影响别的样本的优势。

反过来说，被丢掉的长轨迹是有系统性的——`repeat_loop` 的回合步数中位数明显更高，也就是
被丢掉的多是负样本。这会削弱「别绕圈」这个方向的梯度，所以丢弃率必须被报出来
（`RolloutStats.dropped_too_long`），不能静默。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .sft_dataset import IGNORE_INDEX, RenderStats, build_example

# 没走到终局的默认回报。低于环境最差的合法结局（wrong_purchase −0.85）。
DEFAULT_INVALID_REWARD = -1.0


@dataclass
class GroupStats:
    """一轮采样的审计。每一项都对应一种「样本没进梯度」的原因。"""

    trajectories: int = 0
    unverifiable: int = 0          # reward_valid=False，剔除
    no_terminal: int = 0           # 没走到终局，吃 invalid_reward
    groups: int = 0
    groups_too_small: int = 0      # 有效样本 < 2
    groups_no_variance: int = 0    # 全组同一个回报
    kept: int = 0                  # 最终进梯度的轨迹数
    rewards: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        rewards = self.rewards
        return {
            "trajectories": self.trajectories,
            "unverifiable": self.unverifiable,
            "no_terminal": self.no_terminal,
            "groups": self.groups,
            "groups_too_small": self.groups_too_small,
            "groups_no_variance": self.groups_no_variance,
            "kept": self.kept,
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        }


def shape_reward(
    record: Mapping[str, Any], *, invalid_reward: float = DEFAULT_INVALID_REWARD
) -> float | None:
    """一条轨迹 → 一个标量回报。返回 None 表示这条不参与训练也不进基线。

    判定顺序不能换：`reward_valid` 要先看。环境判不了的回合顶层 `reward` 是 None，
    先看 `reward` 会把它误判成「没走到终局」而罚地板分——罚的是环境的问题不是模型的。
    """
    if not record.get("reward_valid", True):
        return None
    reward = record.get("reward")
    if reward is None:
        return float(invalid_reward)
    return float(reward)


def group_advantages(
    records: Sequence[Mapping[str, Any]],
    *,
    invalid_reward: float = DEFAULT_INVALID_REWARD,
    stats: GroupStats | None = None,
) -> list[tuple[Mapping[str, Any], float]]:
    """按 `task_id` 分组，返回 [(轨迹, 优势)]，已剔除无信息的组。

    优势 = 回报 − 组均值（不除 std，见模块 docstring）。
    """
    stats = stats if stats is not None else GroupStats()
    groups: dict[Any, list[tuple[Mapping[str, Any], float]]] = defaultdict(list)

    for record in records:
        stats.trajectories += 1
        reward = shape_reward(record, invalid_reward=invalid_reward)
        if reward is None:
            stats.unverifiable += 1
            continue
        if record.get("reward") is None:
            stats.no_terminal += 1
        groups[record.get("task_id")].append((record, reward))

    out: list[tuple[Mapping[str, Any], float]] = []
    for members in groups.values():
        stats.groups += 1
        if len(members) < 2:
            stats.groups_too_small += 1
            continue
        rewards = [reward for _, reward in members]
        baseline = sum(rewards) / len(rewards)
        advantages = [reward - baseline for reward in rewards]
        if all(advantage == 0.0 for advantage in advantages):
            stats.groups_no_variance += 1
            continue
        stats.rewards.extend(rewards)
        for (record, _), advantage in zip(members, advantages):
            out.append((record, advantage))
    return out


def build_examples(
    scored: Iterable[tuple[Mapping[str, Any], float]],
    tokenizer: Any,
    *,
    tools: Sequence[Mapping[str, Any]],
    max_length: int = 32768,
    stats: GroupStats | None = None,
    render_stats: RenderStats | None = None,
) -> list[dict[str, Any]]:
    """把打好优势的轨迹渲染成训练样本。

    渲染完全复用 SFT 那套：assistant 的可训练区间是拿两次 chat template 渲染做差
    得到的，不硬编码 `<|im_start|>assistant`。这里的 `labels` 不再是「要拟合的目标」
    而是「哪些 token 是模型自己的动作」——GRPO 的 mask 和 SFT 的 mask 恰好是同一个东西。

    轨迹记录本身不带 `tools`（`Trajectory.as_record` 不存 schema），必须由调用方传入
    采样时用的那一份，否则渲染出的前缀和模型当时看到的不是同一个。
    """
    stats = stats if stats is not None else GroupStats()
    render_stats = render_stats if render_stats is not None else RenderStats()

    examples: list[dict[str, Any]] = []
    for record, advantage in scored:
        # `total` 由 SFT 的 build_dataset 一次性赋值，这条路径不经过它。不自己加就恒为 0，
        # 审计行会显示「渲染 0 条、丢 0 条」——丢弃率算出来是 0/0，正好把该看见的东西藏起来。
        render_stats.total += 1
        row = dict(record)
        row["tools"] = list(tools)
        example = build_example(
            row, tokenizer, max_length=max_length, stats=render_stats
        )
        if example is None:
            continue
        example["advantage"] = float(advantage)
        example["reward_type"] = record.get("reward_type")
        examples.append(example)
    stats.kept = len(examples)
    return examples


def collate(batch: Sequence[Mapping[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    """在 SFT 的 collate 之上补一列每条序列的优势。

    优势是**逐序列**的标量，但 loss 是逐 token 累加的，所以要广播到 token 上。广播
    放在前向里做（用 labels 的 mask），这里只把标量带过去。
    """
    import torch

    from .sft_dataset import collate as sft_collate

    out = sft_collate(batch, pad_token_id=pad_token_id)
    out["advantages"] = torch.tensor(
        [float(item["advantage"]) for item in batch], dtype=torch.float32
    )
    return out


def token_weights(labels: Any, advantages: Any) -> Any:
    """把逐序列的优势广播成逐 token 的权重，非动作 token 记 0。

    `labels` 已经右移对齐（调用方传 `labels[:, 1:]`）。返回值与它同形。
    """
    mask = (labels != IGNORE_INDEX).to(advantages.dtype)
    return mask * advantages.unsqueeze(1)
