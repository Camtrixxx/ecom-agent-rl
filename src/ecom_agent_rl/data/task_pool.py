"""把商品池切成互不重叠的 SFT / GRPO / 评测三个任务池。

切分要满足三件事，缺一件后面的结论就不可信：

1. **零重叠**。评测题一旦出现在训练侧，指标就没有意义。
2. **分层均衡**。三池在难度轴（`attributes` 数量）和品类轴（`domain_en_short`）上
   的分布应当一致，否则「SFT 比 baseline 高 3 个点」可能只是它抽到了更简单的题。
3. **可复现**。固定 seed + 记录 sha256 与血缘，换机器能重建同一份切分。

难度轴选 `attributes` 数量的理由见 docs/environment-notes.md：`weighted_preferences`
与 hard_constraints 三个字段恒空，`attributes` 是当前唯一免费且可靠的难度信号。
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

# 商品数据里每个商品恰好 1 条 instruction，task_id 即商品下标。见 environment-notes。
EXPECTED_PRODUCTS = 23421
PRODUCT_SHA256 = "57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f"

SCHEMA_VERSION = "ecom-task-pool-v1"
ENVIRONMENT = "shopsimulator-environment-v2.1"
REWARD = "shopsimulator-reward-v3"

# attributes 数量的分层桶。8 以上样本较少（2,306 条）合并成一桶，
# 否则最难的几档在 500 题的评测集里会只剩个位数，分层报告没有统计意义。
DIFFICULTY_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("1-2", 1, 2),
    ("3-4", 3, 4),
    ("5-7", 5, 7),
    ("8+", 8, 10**9),
)


def difficulty_bucket(attribute_count: int) -> str:
    for name, low, high in DIFFICULTY_BUCKETS:
        if low <= attribute_count <= high:
            return name
    # 0 条 attributes 的任务归入最易桶：core_functions 维度为空，软匹配退化。
    return DIFFICULTY_BUCKETS[0][0]


@dataclass(frozen=True)
class Task:
    """一条任务。`task_id` 是商品在数据文件中的下标，环境按它 reset。"""

    task_id: int
    asin: str
    domain: str
    attribute_count: int

    @property
    def stratum(self) -> tuple[str, str]:
        return (self.domain, difficulty_bucket(self.attribute_count))

    def as_record(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "asin": self.asin,
            "domain": self.domain,
            "attribute_count": self.attribute_count,
            "difficulty": difficulty_bucket(self.attribute_count),
        }


def load_tasks(products_path: Path) -> list[Task]:
    """读商品池并校验「每商品恰好 1 条 instruction」这一前提。

    这个前提是零重叠的基础：若一个商品挂多条任务，仅按 task_id 切分会让同一商品的
    不同任务落进不同池子，形成商品级泄漏——模型在训练时见过该商品，评测时又考它。
    """
    with gzip.open(products_path, "rt", encoding="utf-8") as handle:
        products = json.load(handle)

    tasks: list[Task] = []
    for task_id, item in enumerate(products):
        instructions = item.get("instructions") or []
        if len(instructions) != 1:
            raise ValueError(
                f"task_id {task_id} (asin {item.get('asin')}) 有 {len(instructions)} 条 "
                "instruction，不是 1 条。task_id 与商品不再一一对应，"
                "按 task_id 切分会产生商品级泄漏，切分逻辑需改为按 asin 分组。"
            )
        tasks.append(
            Task(
                task_id=task_id,
                asin=str(item["asin"]),
                domain=str(item.get("domain_en_short") or "unknown"),
                attribute_count=len(instructions[0].get("attributes") or []),
            )
        )
    return tasks


def _stratified_allocate(
    strata: dict[tuple[str, str], list[Task]], sizes: dict[str, int], rng: random.Random
) -> dict[str, list[Task]]:
    """按层比例分配，再把余数补齐到目标数量。

    每层内先洗牌，然后按各池占总量的比例切片——同一层的任务只会进一个池子，
    因此零重叠由构造保证，不依赖事后去重。
    """
    total_requested = sum(sizes.values())
    pool_total = sum(len(items) for items in strata.values())
    if total_requested > pool_total:
        raise ValueError(f"需要 {total_requested} 条任务，池子只有 {pool_total} 条")

    names = list(sizes)
    picked: dict[str, list[Task]] = {name: [] for name in names}
    leftovers: list[Task] = []

    for stratum in sorted(strata):
        items = list(strata[stratum])
        rng.shuffle(items)
        # 该层应分给各池的条数，按池子大小占比等分。floor 之后的余数进 leftovers,
        # 稍后统一补齐，避免每层各自 round 导致总数偏离目标。
        cursor = 0
        for name in names:
            take = int(len(items) * sizes[name] / pool_total)
            picked[name].extend(items[cursor : cursor + take])
            cursor += take
        leftovers.extend(items[cursor:])

    rng.shuffle(leftovers)
    cursor = 0
    for name in names:
        deficit = sizes[name] - len(picked[name])
        if deficit > 0:
            picked[name].extend(leftovers[cursor : cursor + deficit])
            cursor += deficit
    for name in names:
        # 比例分配可能给某池多切了一条，按目标数截断。
        picked[name] = picked[name][: sizes[name]]
        if len(picked[name]) != sizes[name]:
            raise AssertionError(f"{name}: 分到 {len(picked[name])} 条，目标 {sizes[name]}")
    return picked


def split_tasks(
    tasks: Sequence[Task], sizes: dict[str, int], seed: int
) -> dict[str, list[Task]]:
    """把任务分层切成若干互不重叠的池子。`sizes` 的 key 即池子名。"""
    strata: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for task in tasks:
        strata[task.stratum].append(task)

    rng = random.Random(seed)
    splits = _stratified_allocate(strata, sizes, rng)

    seen: dict[int, str] = {}
    for name, items in splits.items():
        for task in items:
            if task.task_id in seen:
                raise AssertionError(
                    f"task_id {task.task_id} 同时出现在 {seen[task.task_id]} 和 {name}"
                )
            seen[task.task_id] = name
    # 每池内按 task_id 排序，让输出文件稳定、便于 diff。
    return {name: sorted(items, key=lambda t: t.task_id) for name, items in splits.items()}


def distribution(tasks: Iterable[Task]) -> dict[str, dict[str, int]]:
    """池子在两个分层轴上的分布，用于核对切分是否均衡。"""
    items = list(tasks)
    return {
        "domain": dict(sorted(Counter(t.domain for t in items).items())),
        "difficulty": dict(
            sorted(
                Counter(difficulty_bucket(t.attribute_count) for t in items).items(),
                key=lambda kv: [b[0] for b in DIFFICULTY_BUCKETS].index(kv[0]),
            )
        ),
    }


def write_split(path: Path, tasks: Sequence[Task]) -> str:
    """写 jsonl 并返回 sha256。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(task.as_record(), ensure_ascii=False, sort_keys=True) + "\n"
        for task in tasks
    )
    path.write_text(payload, encoding="utf-8")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
