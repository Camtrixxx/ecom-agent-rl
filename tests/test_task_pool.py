"""任务池切分的不变量：零重叠、分层均衡、可复现。

这三条是后续所有指标可信度的前提，所以每条都要有测试钉住，而不是靠跑一次看输出。
"""

from __future__ import annotations

import json
import random

import pytest

from ecom_agent_rl.data.task_pool import (
    DIFFICULTY_BUCKETS,
    Task,
    difficulty_bucket,
    distribution,
    load_tasks,
    split_tasks,
    write_split,
)


def make_tasks(count: int, *, domains: int = 9, seed: int = 0) -> list[Task]:
    rng = random.Random(seed)
    return [
        Task(
            task_id=i,
            asin=f"asin-{i}",
            domain=f"domain-{i % domains}",
            attribute_count=rng.randint(1, 12),
        )
        for i in range(count)
    ]


def test_splits_are_disjoint_and_exact_size():
    tasks = make_tasks(4000)
    sizes = {"sft_train": 1500, "sft_val": 250, "grpo_train": 1200, "eval": 300}
    splits = split_tasks(tasks, sizes, seed=42)

    for name, size in sizes.items():
        assert len(splits[name]) == size, f"{name} 数量不对"

    ids = [t.task_id for items in splits.values() for t in items]
    assert len(ids) == len(set(ids)), "池子之间有重叠"


def test_split_is_reproducible_with_same_seed():
    tasks = make_tasks(2000)
    sizes = {"a": 500, "b": 300}
    first = split_tasks(tasks, sizes, seed=7)
    second = split_tasks(tasks, sizes, seed=7)
    assert {k: [t.task_id for t in v] for k, v in first.items()} == {
        k: [t.task_id for t in v] for k, v in second.items()
    }


def test_different_seed_gives_a_different_split():
    tasks = make_tasks(2000)
    sizes = {"a": 500, "b": 300}
    a = [t.task_id for t in split_tasks(tasks, sizes, seed=1)["a"]]
    b = [t.task_id for t in split_tasks(tasks, sizes, seed=2)["a"]]
    assert a != b


def test_strata_are_proportionally_balanced():
    """各池的难度分布应当接近，否则池间比较会被难度差异污染。"""
    tasks = make_tasks(9000)
    sizes = {"train": 3000, "eval": 500}
    splits = split_tasks(tasks, sizes, seed=42)

    def shares(name: str) -> dict[str, float]:
        dist = distribution(splits[name])["difficulty"]
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}

    train, evaluation = shares("train"), shares("eval")
    for bucket, _, _ in DIFFICULTY_BUCKETS:
        # 500 题的小池子有抽样噪声，容差取 5 个百分点。
        assert abs(train.get(bucket, 0) - evaluation.get(bucket, 0)) < 0.05, (
            f"{bucket} 桶占比差异过大: train={train.get(bucket)} eval={evaluation.get(bucket)}"
        )


def test_every_domain_appears_in_the_eval_split():
    """评测集必须覆盖全部品类，否则分品类报告会有空行。"""
    tasks = make_tasks(9000, domains=9)
    splits = split_tasks(tasks, {"train": 3000, "eval": 500}, seed=42)
    assert len({t.domain for t in splits["eval"]}) == 9


def test_requesting_more_than_the_pool_is_rejected():
    with pytest.raises(ValueError, match="池子只有"):
        split_tasks(make_tasks(100), {"a": 80, "b": 40}, seed=0)


def test_difficulty_buckets_cover_all_counts():
    assert difficulty_bucket(0) == "1-2"  # attributes 为空退化到最易桶
    assert difficulty_bucket(1) == "1-2"
    assert difficulty_bucket(4) == "3-4"
    assert difficulty_bucket(7) == "5-7"
    assert difficulty_bucket(22) == "8+"


def test_write_split_is_stable_and_hashed(tmp_path):
    tasks = make_tasks(50)[:10]
    path = tmp_path / "split.jsonl"
    first = write_split(path, tasks)
    second = write_split(path, tasks)
    assert first == second, "同样输入应得到同样 sha256"

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 10
    record = json.loads(lines[0])
    assert set(record) == {"task_id", "asin", "domain", "attribute_count", "difficulty"}


def test_load_tasks_rejects_multi_instruction_products(tmp_path):
    """零重叠依赖「每商品 1 条 instruction」，前提被破坏必须报错而不是静默切分。"""
    import gzip

    path = tmp_path / "products.json.gz"
    payload = [
        {"asin": "a", "domain_en_short": "Home", "instructions": [{"attributes": ["x"]}]},
        {
            "asin": "b",
            "domain_en_short": "Home",
            "instructions": [{"attributes": ["x"]}, {"attributes": ["y"]}],
        },
    ]
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)

    with pytest.raises(ValueError, match="商品级泄漏"):
        load_tasks(path)
