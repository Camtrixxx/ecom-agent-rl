"""SFT 训练脚本的分批 / 分卡口径。

这三组测试各自钉住一个已经踩过的坑，退回旧行为时会失败：

- `val_rounds`：验证集不能用 shard()，那会按卡数丢掉最长的样本，让 val loss 跨卡数
  不可比（S1）。
- `steps_in_epoch`：步数公式必须和训练循环同口径，ceil 会多跑一个残 epoch（S3）。
- 两者与 `token_budget_batches` 的联动：真实的 111 批验证集在 6/7/8 卡下必须给出
  同一个 (loss 和, token 和)。

`train_sft.py` 的 torch 依赖都在 main() 里，模块级 import 不会拖起 torch。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_sft import shard, steps_in_epoch, token_budget_batches, val_rounds


def _fake(lengths: list[int]) -> list[dict[str, list[int]]]:
    return [{"input_ids": [0] * n} for n in lengths]


class Test验证集分卡:
    @pytest.mark.parametrize("world", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_每条批恰好被计一次权重(self, world: int) -> None:
        """把权重按批号加起来，每批必须恰好得 1.0——不重不漏。"""
        batches = [[i] for i in range(111)]
        seen: dict[int, float] = {}
        for rank in range(world):
            for indices, weight in val_rounds(batches, rank, world):
                seen[indices[0]] = seen.get(indices[0], 0.0) + weight
        assert len(seen) == 111, "有批从未被任何 rank 前向过"
        assert all(w == pytest.approx(1.0) for w in seen.values()), \
            f"权重不为 1 的批：{ {k: v for k, v in seen.items() if v != 1.0} }"

    @pytest.mark.parametrize("world", [1, 2, 3, 5, 6, 7, 8])
    def test_各rank轮数相同(self, world: int) -> None:
        """轮数不等就意味着 forward 次数不等，FSDP 会卡死在 all-gather 上。"""
        batches = [[i] for i in range(111)]
        counts = {len(val_rounds(batches, r, world)) for r in range(world)}
        assert len(counts) == 1, f"各 rank 轮数不一致：{counts}"

    def test_补齐的那几轮权重为零(self) -> None:
        # 10 批 3 卡：rank0 得 4 批，rank1/rank2 各 3 批 + 1 个零权重占位。
        batches = [[i] for i in range(10)]
        assert [w for _, w in val_rounds(batches, 0, 3)] == [1.0, 1.0, 1.0, 1.0]
        assert [w for _, w in val_rounds(batches, 1, 3)] == [1.0, 1.0, 1.0, 0.0]
        assert [w for _, w in val_rounds(batches, 2, 3)] == [1.0, 1.0, 1.0, 0.0]

    def test_批数少于卡数直接报错而不是静默算错(self) -> None:
        with pytest.raises(ValueError, match="少于"):
            val_rounds([[0], [1]], 0, 4)

    def test_不像shard那样丢掉最长的样本(self) -> None:
        """shard() 丢尾部，而批是按长度排序的——丢的永远是最长的那几条。"""
        batches = [[i] for i in range(111)]
        kept_by_shard = {b[0] for r in range(7) for b in shard(batches, r, 7)}
        kept_by_val = {i[0] for r in range(7) for i, w in val_rounds(batches, r, 7) if w > 0}
        assert len(kept_by_shard) == 105, "shard 的截断行为变了，S1 的前提要重新核"
        assert kept_by_val == set(range(111))
        # 被 shard 丢掉的恰是排序后最靠后的、也就是最长的那 6 批。
        assert set(range(111)) - kept_by_shard == {105, 106, 107, 108, 109, 110}


class Test每epoch步数:
    @pytest.mark.parametrize(
        "per_rank,grad_accum,expected",
        [(108, 4, 27), (92, 4, 23), (81, 4, 20), (12, 4, 3), (3, 4, 0)],
    )
    def test_与训练循环同口径(self, per_rank: int, grad_accum: int, expected: int) -> None:
        """公式必须等于训练循环 range(0, per_rank - ga + 1, ga) 的实际迭代次数。"""
        actual_loop = len(range(0, per_rank - grad_accum + 1, grad_accum))
        assert steps_in_epoch(per_rank, grad_accum) == actual_loop == expected

    def test_八卡下ceil会多算一步(self) -> None:
        """回归钉子：8 卡 per_rank=81，ceil 给 21 而循环只走 20。

        差这一步的后果是 total_steps=63 而 3 个 epoch 只有 60 步，于是多跑一个残
        epoch，余弦在残 epoch 中途才退到 0，--save-each-epoch 还会多写一个目录。
        """
        import math

        assert math.ceil(81 / 4) == 21
        assert steps_in_epoch(81, 4) == 20
        assert steps_in_epoch(81, 4) * 3 == 60


class Test真实验证集的分卡:
    """用真实长度分布跑一遍：111 批的 val 集在 6/7/8 卡下必须得到同一个和。"""

    @staticmethod
    def _lengths() -> list[int]:
        # 近似真实 val 的长度分布（298 条，最长 29238），关键是**长度递增排序后**
        # 尾部几批最长——这正是 shard() 会丢掉的部分。
        return [800 + i * 95 for i in range(298)]

    def test_三种卡数下的loss与token和完全相同(self) -> None:
        examples = _fake(self._lengths())
        batches = token_budget_batches(examples, 32768)
        assert len(batches) >= 8

        def totals(world: int) -> tuple[float, int]:
            loss_sum, token_sum = 0.0, 0
            for rank in range(world):
                for indices, weight in val_rounds(batches, rank, world):
                    # 用长度当 loss 的替身：只要加权求和的结构对，真 loss 也对。
                    loss_sum += sum(len(examples[i]["input_ids"]) for i in indices) * weight
                    token_sum += int(sum(len(examples[i]["input_ids"]) for i in indices) * weight)
            return loss_sum, token_sum

        reference = totals(6)
        for world in (1, 7, 8):
            assert totals(world) == reference, f"world_size={world} 与 6 卡不一致"
        # 而且它等于「所有样本都算一次」——没有任何一条被丢。
        assert reference[1] == sum(self._lengths())

    def test_shard丢的批数等于批数取余卡数(self) -> None:
        """对照组：旧行为丢多少完全由卡数决定，这就是 val loss 不可比的来源。

        不断言「每种卡数都丢了东西」——整除时确实一批不丢，那取决于批数凑巧是多少，
        不是可依赖的性质。可依赖的是这条取余关系，以及它随卡数变化。
        """
        examples = _fake(self._lengths())
        batches = token_budget_batches(examples, 32768)
        dropped_batches = {}
        for world in (6, 7, 8):
            kept = {id(b) for r in range(world) for b in shard(batches, r, world)}
            dropped_batches[world] = len(batches) - len(kept)
            assert dropped_batches[world] == len(batches) % world
        assert len(set(dropped_batches.values())) > 1, \
            "三种卡数丢的批数完全一样，S1 的『不可比』论据要重新核"

        # 而且丢的一定是尾部——批按长度升序，尾部就是最长的样本。
        kept7 = [b for r in range(7) for b in shard(batches, r, 7)]
        tail = batches[len(batches) - len(batches) % 7:]
        assert all(id(b) not in {id(k) for k in kept7} for b in tail)
