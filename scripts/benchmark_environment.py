#!/usr/bin/env python3
"""压测 ShopSimulator 环境池的并发吞吐，确定 rollout / 评测 / 教师采集的并发上限。

走 `EnvironmentPool`（与真实调用同一条路径），每个 worker 跑完整生命周期：
reset → interact × steps → release_one。只打空请求测不出会话状态与租约的真实开销。

用法：
    # 起 8 个 worker（默认）
    bash scripts/start_environment.sh

    # 压测：并发扫到远超池子容量，观察吞吐何时不再增长
    python scripts/benchmark_environment.py --workers 8 --concurrency 1 8 16 32 64
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ecom_agent_rl.environment.pool import (  # noqa: E402
    DEFAULT_BASE_PORT,
    EnvironmentPool,
    EnvironmentServiceError,
)

# 环境侧 MAX_HISTORY_LENGTH=42，一次 interact 追加 1 条记录，
# 取远小于上限的步数，避免压测本身触发 over。
DEFAULT_STEPS = 5
TASK_POOL_SIZE = 23421


@dataclass
class Outcome:
    ok: bool
    latencies: list[float] = field(default_factory=list)
    error: str | None = None


def run_episode(pool: EnvironmentPool, task_idx: int, steps: int) -> Outcome:
    outcome = Outcome(ok=False)
    try:
        start = time.perf_counter()
        with pool.episode(task_idx) as episode:
            outcome.latencies.append(time.perf_counter() - start)
            for _ in range(steps):
                start = time.perf_counter()
                step = episode.interact("search[测试查询]")
                outcome.latencies.append(time.perf_counter() - start)
                if step.over:
                    break
        outcome.ok = True
    except EnvironmentServiceError as exc:
        outcome.error = str(exc)
    return outcome


def measure(
    pool: EnvironmentPool, concurrency: int, episodes: int, steps: int, seed: int
) -> dict:
    rng = random.Random(seed)
    tasks = [rng.randrange(TASK_POOL_SIZE) for _ in range(episodes)]
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        outcomes = list(
            executor.map(lambda idx: run_episode(pool, idx, steps), tasks)
        )
    wall = time.perf_counter() - start

    latencies = sorted(value for item in outcomes for value in item.latencies)
    succeeded = sum(1 for item in outcomes if item.ok)
    errors: dict[str, int] = {}
    for item in outcomes:
        if item.error:
            # 归一到错误类别，避免 url/env_idx 把直方图打散。
            key = item.error.split(": ", 1)[-1][:60]
            errors[key] = errors.get(key, 0) + 1

    return {
        "concurrency": concurrency,
        "episodes": episodes,
        "succeeded": succeeded,
        "wall_seconds": round(wall, 2),
        "episodes_per_second": round(succeeded / wall, 3) if wall else 0.0,
        "requests_per_second": round(len(latencies) / wall, 2) if wall else 0.0,
        "latency_ms": {
            "mean": round(statistics.mean(latencies) * 1000, 1) if latencies else None,
            "p50": round(latencies[len(latencies) // 2] * 1000, 1) if latencies else None,
            "p99": round(latencies[int(len(latencies) * 0.99)] * 1000, 1)
            if len(latencies) > 1
            else None,
        },
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=DEFAULT_BASE_PORT)
    parser.add_argument(
        "--workers", type=int, default=8, help="环境进程数，须与启动时一致"
    )
    parser.add_argument("--slots-per-worker", type=int, default=4)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument(
        "--episodes", type=int, default=0, help="每档回合数；默认取 concurrency × 4"
    )
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--ready-timeout", type=float, default=900.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-out")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pool = EnvironmentPool(
        host=args.host,
        base_port=args.base_port,
        workers=args.workers,
        slots_per_worker=args.slots_per_worker,
        timeout=args.timeout,
    )
    print(
        f"workers: {args.workers}   pool capacity: {pool.capacity}   "
        f"steps/episode: {args.steps}"
    )
    print("waiting for all workers...", flush=True)
    pool.wait_until_ready(args.ready_timeout)
    print("ready\n")

    header = f"{'conc':>5} {'ok':>9} {'ep/s':>7} {'req/s':>8} {'p50ms':>8} {'p99ms':>9}"
    print(header)
    print("-" * len(header))

    reports = []
    for concurrency in args.concurrency:
        episodes = args.episodes or concurrency * 4
        report = measure(pool, concurrency, episodes, args.steps, args.seed)
        reports.append(report)
        latency = report["latency_ms"]
        print(
            f"{concurrency:>5} {report['succeeded']:>4}/{episodes:<4} "
            f"{report['episodes_per_second']:>7} {report['requests_per_second']:>8} "
            f"{latency['p50'] or 0:>8} {latency['p99'] or 0:>9}"
            + (f"   errors={report['errors']}" if report["errors"] else ""),
            flush=True,
        )

    best = max(reports, key=lambda item: item["episodes_per_second"])
    print(
        f"\npeak: {best['episodes_per_second']} episodes/s "
        f"at concurrency={best['concurrency']} with {args.workers} workers"
    )
    print(
        "并发超过池子容量后吞吐不再增长即为饱和点；若饱和吞吐 ≈ workers × 5 ep/s，"
        "说明扩展是线性的，继续加 worker 即可。"
    )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
