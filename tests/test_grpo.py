"""GRPO 逻辑层的测试。

重点钉的是三件「算错了也不会报错、只会让训练学歪」的事：
回报整形对三类结局的区分、优势的基线口径、以及优势到 token 权重的广播。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# 轮次抽题的逻辑住在 scripts/train_grpo.py 里（它只依赖 stdlib，import 不会拖起 torch）。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from ecom_agent_rl.training.grpo import (
    DEFAULT_INVALID_REWARD,
    GroupStats,
    group_advantages,
    shape_reward,
    token_weights,
)
from ecom_agent_rl.training.sft_dataset import IGNORE_INDEX


def record(task_id: int, **kwargs):
    base = {"task_id": task_id, "reward": 0.0, "reward_valid": True, "reward_type": "x"}
    base.update(kwargs)
    return base


class TestShapeReward:
    def test_正常终局直接用环境的数(self):
        assert shape_reward(record(1, reward=0.55)) == 0.55
        assert shape_reward(record(1, reward=-0.85)) == -0.85

    def test_环境判不了的回合返回None而不是0(self):
        """`reward_unverifiable` 在环境里取 0.0。照搬这个 0.0 会让它在一个平均回报
        为负的组里变成正优势——等于奖励模型去触发「判不了」。"""
        assert shape_reward(record(1, reward=None, reward_valid=False)) is None

    def test_判不了时先看reward_valid再看reward(self):
        """两个字段都缺失时的判定顺序：先 reward_valid。反过来会把它误判成
        「没走到终局」而罚地板分——罚的是环境的问题，不是模型的。"""
        assert shape_reward(record(1, reward=None, reward_valid=False)) is None
        assert shape_reward(record(1, reward=None, reward_valid=True)) == DEFAULT_INVALID_REWARD

    def test_没走到终局吃地板分且严格低于最差合法结局(self):
        value = shape_reward(record(1, reward=None))
        assert value == DEFAULT_INVALID_REWARD
        # wrong_purchase 是环境里最差的合法结局
        assert value < -0.85

    def test_地板分可配(self):
        assert shape_reward(record(1, reward=None), invalid_reward=-0.5) == -0.5

    def test_reward_valid缺省视为True(self):
        row = {"task_id": 1, "reward": 1.0}
        assert shape_reward(row) == 1.0


class TestGroupAdvantages:
    def test_优势是回报减组均值(self):
        rows = [record(1, reward=1.0), record(1, reward=0.0), record(1, reward=-0.5)]
        out = group_advantages(rows)
        baseline = 0.5 / 3
        assert [pytest.approx(a) for a in (1.0 - baseline, -baseline, -0.5 - baseline)] == [
            a for _, a in out
        ]

    def test_组内优势之和为零(self):
        rows = [record(1, reward=r) for r in (1.0, 0.55, -0.85, -0.65)]
        assert sum(a for _, a in group_advantages(rows)) == pytest.approx(0.0)

    def test_不除标准差(self):
        """除以 std 会给「组内方差小」的题更大的梯度权重，而方差小恰恰意味着这题
        没什么可学的。两个组只差一个常数缩放时，优势也应该只差同一个缩放。"""
        tight = group_advantages([record(1, reward=0.1), record(1, reward=-0.1)])
        wide = group_advantages([record(2, reward=1.0), record(2, reward=-1.0)])
        assert [a for _, a in tight] == [pytest.approx(a / 10) for _, a in wide]

    def test_按task_id分组而不是混成一个大池(self):
        rows = [
            record(1, reward=1.0), record(1, reward=0.0),
            record(2, reward=-1.0), record(2, reward=-2.0),
        ]
        out = group_advantages(rows)
        assert [a for _, a in out] == [0.5, -0.5, 0.5, -0.5]

    def test_单样本的组整组丢弃(self):
        """一个样本算不出基线：`r - r = 0` 不是「没有优势」而是「没有信息」。"""
        stats = GroupStats()
        out = group_advantages([record(1, reward=1.0)], stats=stats)
        assert out == []
        assert stats.groups_too_small == 1

    def test_全组同回报的组丢弃(self):
        stats = GroupStats()
        rows = [record(1, reward=0.55) for _ in range(4)]
        assert group_advantages(rows, stats=stats) == []
        assert stats.groups_no_variance == 1

    def test_判不了的样本不进基线(self):
        """它既不该有梯度，也不该把组均值往上拉。"""
        stats = GroupStats()
        rows = [
            record(1, reward=1.0), record(1, reward=0.0),
            record(1, reward=None, reward_valid=False),
        ]
        out = group_advantages(rows, stats=stats)
        assert len(out) == 2
        assert [a for _, a in out] == [0.5, -0.5]  # 基线是 0.5，不是 1/3
        assert stats.unverifiable == 1

    def test_没走到终局的样本进基线且吃地板分(self):
        stats = GroupStats()
        rows = [record(1, reward=1.0), record(1, reward=None)]
        out = group_advantages(rows, stats=stats)
        assert [a for _, a in out] == [1.0, -1.0]  # 基线 (1.0 + -1.0)/2 = 0
        assert stats.no_terminal == 1
        assert stats.unverifiable == 0

    def test_审计计数与实际保留一致(self):
        stats = GroupStats()
        rows = [
            record(1, reward=1.0), record(1, reward=-0.85),
            record(2, reward=0.55), record(2, reward=0.55),   # 无方差
            record(3, reward=1.0),                            # 太小
        ]
        out = group_advantages(rows, stats=stats)
        assert stats.trajectories == 5
        assert stats.groups == 3
        assert stats.groups_no_variance == 1
        assert stats.groups_too_small == 1
        assert len(out) == 2
        # `_kept` 只统计真正进了梯度的组
        assert stats.to_dict()["mean_reward_kept"] == pytest.approx((1.0 - 0.85) / 2)

    def test_两个回报口径必须分开(self):
        """`_kept` 丢掉无方差组，而被丢的恰是满分组，于是它系统性偏低。

        这一条钉的是"别再合并成一个 `mean_reward`"：下面这组数据里两个口径相差
        0.29，只看 `_kept` 会把一个大部分题都做对的策略读成平庸。
        """
        stats = GroupStats()
        rows = [
            record(1, reward=1.0), record(1, reward=1.0),      # 全对 → 无方差，被丢
            record(2, reward=1.0), record(2, reward=1.0),      # 全对 → 无方差，被丢
            record(3, reward=1.0), record(3, reward=-0.85),    # 有方差 → 进梯度
        ]
        group_advantages(rows, stats=stats)
        payload = stats.to_dict()
        assert stats.groups_no_variance == 2
        # 全部有效回报：满分组照样算进来
        assert payload["mean_reward_all"] == pytest.approx((1.0 * 5 - 0.85) / 6, abs=1e-4)
        assert payload["mean_reward_kept"] == pytest.approx((1.0 - 0.85) / 2)
        assert payload["mean_reward_all"] > payload["mean_reward_kept"]
        # 旧的有歧义字段不许复活：读到它的代码拿到的是一个会误导的数。
        assert "mean_reward" not in payload


class TestTokenWeights:
    def test_只有动作token拿到权重(self):
        torch = pytest.importorskip("torch")
        labels = torch.tensor([[IGNORE_INDEX, 5, 6, IGNORE_INDEX]])
        advantages = torch.tensor([0.5])
        assert token_weights(labels, advantages).tolist() == [[0.0, 0.5, 0.5, 0.0]]

    def test_每条序列拿自己的优势(self):
        torch = pytest.importorskip("torch")
        labels = torch.tensor([[1, IGNORE_INDEX], [IGNORE_INDEX, 2]])
        advantages = torch.tensor([0.25, -0.75])
        assert token_weights(labels, advantages).tolist() == [[0.25, 0.0], [0.0, -0.75]]

    def test_负优势保持负号(self):
        """符号错了就是在奖励失败的轨迹，而 loss 曲线照样好看。"""
        torch = pytest.importorskip("torch")
        labels = torch.tensor([[7]])
        assert token_weights(labels, torch.tensor([-1.0])).tolist() == [[-1.0]]


class TestLossEquivalence:
    def test_加权CE等于负的加权logprob(self):
        """GRPO 的 loss 写成 `Σ A·CE` 是为了复用 SFT 那段分块 fp32 上采的 CE。
        这一条钉住它确实等于 `-Σ A·log πθ`，即带基线的策略梯度。"""
        torch = pytest.importorskip("torch")
        logits = torch.randn(1, 3, 11)
        labels = torch.tensor([[2, IGNORE_INDEX, 7]])
        advantage = torch.tensor([0.6])

        weights = token_weights(labels, advantage)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 11), labels.reshape(-1),
            ignore_index=IGNORE_INDEX, reduction="none",
        )
        weighted_ce = (ce * weights.reshape(-1)).sum()

        logprobs = torch.log_softmax(logits, dim=-1)
        manual = -0.6 * (logprobs[0, 0, 2] + logprobs[0, 2, 7])
        assert weighted_ce.item() == pytest.approx(manual.item(), rel=1e-5)

    def test_分块求和与整体求和等价(self):
        """vocab 152064 下整体 `.float()` 要 18.2G，必须分块。分块只有在
        逐 token（reduction="none"）再加权求和时才严格等价。"""
        torch = pytest.importorskip("torch")
        logits = torch.randn(2, 8, 17)
        labels = torch.randint(0, 17, (2, 8))
        labels[0, ::2] = IGNORE_INDEX
        weights = token_weights(labels, torch.tensor([0.3, -0.9]))

        flat_logits, flat_labels = logits.reshape(-1, 17), labels.reshape(-1)
        flat_weights = weights.reshape(-1)

        def compute(chunk: int) -> float:
            total = 0.0
            for start in range(0, flat_labels.numel(), chunk):
                piece = torch.nn.functional.cross_entropy(
                    flat_logits[start : start + chunk].float(),
                    flat_labels[start : start + chunk],
                    ignore_index=IGNORE_INDEX, reduction="none",
                )
                total += (piece * flat_weights[start : start + chunk]).sum().item()
            return total

        assert compute(5) == pytest.approx(compute(16), rel=1e-5)


class TestCleanTorchrunEnv:
    def test_剥掉elastic的store开关(self):
        """留着它，vLLM 的 TCPStore 不会启 daemon，只会死等 600s 超时——报的是
        「client socket timed out」，看不出和 accelerate 有任何关系。"""
        from train_grpo import clean_torchrun_env

        out = clean_torchrun_env({"TORCHELASTIC_USE_AGENT_STORE": "True", "PATH": "/x"})
        assert "TORCHELASTIC_USE_AGENT_STORE" not in out
        assert out["PATH"] == "/x"

    def test_剥掉rank与master(self):
        from train_grpo import clean_torchrun_env

        source = {"RANK": "3", "WORLD_SIZE": "6", "MASTER_ADDR": "10.0.0.1",
                  "MASTER_PORT": "29500", "LOCAL_RANK": "3"}
        assert clean_torchrun_env(source) == {}

    def test_不动业务变量(self):
        """代理绕过和 API key 必须原样传下去，否则 vLLM 连不上或起不来。"""
        from train_grpo import clean_torchrun_env

        source = {"no_proxy": "*", "CUDA_VISIBLE_DEVICES": "1", "LD_LIBRARY_PATH": "/c"}
        assert clean_torchrun_env(source) == source

    def test_不修改传进来的环境(self):
        from train_grpo import clean_torchrun_env

        source = {"RANK": "0", "PATH": "/x"}
        clean_torchrun_env(source)
        assert source == {"RANK": "0", "PATH": "/x"}


class TestIterationTasks:
    def test_同一轮内题目不重复(self):
        """重复会让两个「组」共享一道题，基线各算一份，等于给那道题双倍权重。"""
        from train_grpo import iteration_tasks

        picked = iteration_tasks(list(range(100)), iteration=3, count=24, seed=7)
        assert len(picked) == len(set(picked)) == 24

    def test_同轮次同seed可复现(self):
        from train_grpo import iteration_tasks

        pool = list(range(50))
        assert iteration_tasks(pool, 2, 8, 7) == iteration_tasks(pool, 2, 8, 7)

    def test_相邻轮次不重叠(self):
        from train_grpo import iteration_tasks

        pool = list(range(100))
        first = set(iteration_tasks(pool, 0, 24, 7))
        second = set(iteration_tasks(pool, 1, 24, 7))
        assert first & second == set()

    def test_池子用完会绕回(self):
        from train_grpo import iteration_tasks

        picked = iteration_tasks(list(range(10)), iteration=1, count=8, seed=7)
        assert len(picked) == 8
        assert set(picked) <= set(range(10))
