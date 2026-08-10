"""并发跑一批 rollout，支持断点续跑。

参考实现是 `for task in tasks: for attempt in ...` 的串行双循环，每步一次阻塞
HTTP。环境侧实测能到 133 episodes/s（32 worker），串行只能吃掉其中 1/30。

并发上限取 `min(pool.capacity, concurrency)`：环境的最优工作点是「并发 = worker 数」
（见 docs/environment-notes.md），再往上加只会让尾延迟爆掉而吞吐不变。所以默认直接
用 worker 数，而不是 `capacity`（后者含 slot 倍数，会超配）。

断点续跑按 `(task_id, attempt)` 去重。结果**边跑边追加**写盘，进程被杀也只丢在飞的
那几条；基础设施类失败（环境挂了、模型服务 502）会中止整批，因为继续跑只是在稳定地
生产垃圾。

基础设施失败**不写进主轨迹文件**，改写同名的 `.failures.jsonl`。写进去会有两个后果：
续跑按 `(task_id, attempt)` 去重，于是这些回合永远不会被重试——"修好之后重跑即可续跑"
就成了空话；而且下游指标层会把它们当成模型的失败来算（实测会出现
`no_terminal:env_error` 混进终局类型分布）。环境挂掉不是模型的表现。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .agent import INFRA_FAILURES, Trajectory, run_episode
from .llm import ChatClient
from ..environment.pool import EnvironmentPool

logger = logging.getLogger(__name__)


def load_task_ids(path: Path) -> list[int]:
    """从任务池 jsonl 读 task_id。"""
    ids: list[int] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(int(json.loads(line)["task_id"]))
    return ids


def completed_attempts(path: Path) -> set[tuple[int, int]]:
    """已经写盘的 (task_id, attempt)，用于续跑时跳过。

    坏行（进程被杀在半行）跳过而不是报错——续跑的目的就是从残缺状态恢复。
    """
    done: set[tuple[int, int]] = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                done.add((int(record["task_id"]), int(record.get("attempt", 0))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return done


class _Writer:
    """把轨迹追加进 jsonl。多线程共写一个文件，必须串行化。"""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, trajectory: Trajectory) -> None:
        line = json.dumps(trajectory.as_record(), ensure_ascii=False)
        with self._lock:
            self._handle.write(line + "\n")
            # 每条都 flush：崩了要能保住已完成的部分，一条轨迹的开销可以忽略。
            self._handle.flush()

    def close(self) -> None:
        with self._lock:
            self._handle.close()


def _plan(
    task_ids: Sequence[int], attempts: int, skip: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    return [
        (task_id, attempt)
        for task_id in task_ids
        for attempt in range(attempts)
        if (task_id, attempt) not in skip
    ]


def run_batch(
    *,
    pool: EnvironmentPool,
    client: ChatClient,
    task_ids: Sequence[int],
    output: Path,
    attempts: int = 1,
    concurrency: int | None = None,
    resume: bool = True,
    on_result: Callable[[Trajectory], None] | None = None,
    **episode_kwargs: Any,
) -> dict[str, Any]:
    """并发跑 `task_ids × attempts` 个回合，边跑边写 `output`。

    返回本次运行的汇总。基础设施失败会中止后续提交，已完成的部分保留在文件里，
    再跑一次同样的命令即可续上。
    """
    skip = completed_attempts(output) if resume else set()
    plan = _plan(task_ids, attempts, skip)
    if concurrency is None:
        # 最优工作点是并发 = worker 数，不是 capacity。
        concurrency = len(pool.urls)
    concurrency = max(1, min(concurrency, len(plan) or 1))

    logger.info(
        "rollout: %d 个回合待跑（跳过已完成 %d 个），并发 %d，输出 %s",
        len(plan), len(skip), concurrency, output,
    )

    writer = _Writer(output)
    # 基础设施失败单独存档：留证据但不占用 (task_id, attempt)，续跑才能重试它们。
    failures = _Writer(output.with_suffix(".failures.jsonl"))
    counts: dict[str, int] = {}
    rewards: list[float] = []
    aborted: str | None = None
    started = time.monotonic()
    lock = threading.Lock()
    stop = threading.Event()

    def worker(item: tuple[int, int]) -> Trajectory | None:
        if stop.is_set():
            return None
        task_id, attempt = item
        return run_episode(
            pool=pool, client=client, task_id=task_id, attempt=attempt, **episode_kwargs
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures: dict[Future[Trajectory | None], tuple[int, int]] = {}
        pending = iter(plan)

        def submit_next() -> bool:
            try:
                item = next(pending)
            except StopIteration:
                return False
            futures[executor.submit(worker, item)] = item
            return True

        for _ in range(concurrency):
            if not submit_next():
                break

        while futures:
            done_futures = [f for f in list(futures) if f.done()]
            if not done_futures:
                time.sleep(0.05)
                continue
            for future in done_futures:
                item = futures.pop(future)
                try:
                    trajectory = future.result()
                except Exception as exc:  # 兜底：run_episode 本该自己收干净
                    logger.exception("rollout worker crashed on %s", item)
                    aborted = f"worker crashed: {type(exc).__name__}: {exc}"
                    stop.set()
                    continue
                if trajectory is None:
                    continue
                (failures if trajectory.infra_failure else writer).write(trajectory)
                with lock:
                    counts[trajectory.status] = counts.get(trajectory.status, 0) + 1
                    if trajectory.reward is not None:
                        rewards.append(trajectory.reward)
                if on_result is not None:
                    on_result(trajectory)
                if trajectory.infra_failure and not stop.is_set():
                    aborted = (
                        f"基础设施失败（{trajectory.status}）：{trajectory.error}。"
                        "修好之后重跑同样的命令即可续跑。"
                    )
                    logger.error("%s", aborted)
                    stop.set()
                if not stop.is_set():
                    submit_next()

    writer.close()
    failures.close()
    elapsed = time.monotonic() - started
    completed = sum(counts.values())
    infra_failed = sum(count for status, count in counts.items() if status in INFRA_FAILURES)
    summary = {
        "planned": len(plan),
        "skipped": len(skip),
        "completed": completed,
        # 写进主文件的条数。基础设施失败不算——它们进了 .failures.jsonl，下次会重试。
        "written": completed - infra_failed,
        "infra_failed": infra_failed,
        "status_counts": dict(sorted(counts.items())),
        "elapsed_seconds": round(elapsed, 1),
        "episodes_per_second": round(completed / elapsed, 2) if elapsed > 0 else None,
        "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else None,
        "rewarded_episodes": len(rewards),
        "aborted": aborted,
        "usage": client.usage.snapshot(),
    }
    return summary
