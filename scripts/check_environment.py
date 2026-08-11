#!/usr/bin/env python3
"""环境池的 slot 体检，必要时回收泄漏的租约。

为什么需要它：服务端的 slot 租约**不会自动回收**，只在收到 `release_one` 时释放。
任何在「服务端已 `slot_pool.acquire()`、客户端还没拿到 env_idx」之间断掉的回合都会
把那个 slot 永久占住，而且不打印任何告警。实测连着跑完一轮采集之后，32 个 slot 漏了
25 个，有效容量掉到 7 —— 表现是「并发明明没超也报 env_error」，而 `env_error` 属于
`INFRA_FAILURES`，会中止整批。

所以顺序是**先量容量再调并发**。一看到 `env_error` 就往下调并发，是在给一个错误的
解释配一个无效的药。长跑前后各跑一次这个脚本是很便宜的保险。

用法：

    python scripts/check_environment.py                 # 只体检，不改动
    python scripts/check_environment.py --reclaim       # 顺手回收漏掉的租约
    python scripts/check_environment.py --json outputs/environment/health.json

退出码：有泄漏（或有 worker 不可用）返回 1，干净返回 0，便于在 shell 里当闸门用。

慢是正常的：满了的 worker 不快速失败，服务端 `MAX_RETRIES=5` ×
`RETRY_DELAY_SECONDS=5` 会把请求挂 25 秒才回错误。脚本跨 worker 并行探测，所以整体
耗时约等于**单个**满 worker 的 25 秒，而不是 25 × worker 数。

`--reclaim` 的适用条件与 `release_all` 一样苛刻：**同端口段上不能有别的 rollout 在
跑**。这个脚本是独立进程，不持有任何租约，所以在它看来全部租约都是孤儿——别人在飞的
回合会被一起放掉。它比 `release_all` 好的地方只有一处：逐个 `release_one` 能报出到底
哪几个编号是租着的，`slot_pool.reset()` 什么都不告诉你。安全性上并没有更好。

（`EnvironmentPool` 自己在 reset 撞墙时会就地回收孤儿，见 `_open_episode`。所以正常
跑 rollout 不需要先跑这个脚本；它是给「想知道现在到底剩多少容量」和「跑之前先清干净」
用的。）
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecom_agent_rl.environment.pool import (  # noqa: E402
    EnvironmentPool,
    EnvironmentServiceError,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=5700)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--env-slots",
        type=int,
        default=4,
        help="服务端每 worker 的 slot 数，须与启动时的 SHOPSIM_ENV_SLOTS 一致；"
        "给小了会把好的 slot 报成不存在，给大了会把不存在的报成泄漏",
    )
    parser.add_argument(
        "--reclaim",
        action="store_true",
        help="回收泄漏的租约。**同端口段上不能有 rollout 在跑**，否则会放掉别人在飞的回合",
    )
    parser.add_argument("--json", type=Path, default=None, help="把结果写成 json")
    parser.add_argument("--ready-timeout", type=float, default=30.0)
    return parser.parse_args()


def measure(pool: EnvironmentPool) -> dict[str, list[int]]:
    """并行量每个 worker 的可用 slot。串行的话 8 个满 worker 要等 200 秒。"""
    urls = pool.urls
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        return dict(zip(urls, executor.map(pool.measure_free_slots, urls)))


def report(free: dict[str, list[int]], env_slots: int) -> dict[str, object]:
    workers = []
    for url, slots in free.items():
        workers.append(
            {
                "url": url,
                "free": len(slots),
                "slots": env_slots,
                "leaked": env_slots - len(slots),
            }
        )
    total_free = sum(w["free"] for w in workers)
    return {
        "workers": workers,
        "total_free": total_free,
        "total_slots": env_slots * len(workers),
        "total_leaked": env_slots * len(workers) - total_free,
    }


def render(summary: dict[str, object], title: str) -> None:
    print(f"\n== {title} ==")
    for worker in summary["workers"]:  # type: ignore[index]
        flag = "  漏 %d" % worker["leaked"] if worker["leaked"] else ""
        print(f"  {worker['url']}  {worker['free']}/{worker['slots']}{flag}")
    print(
        f"  合计 {summary['total_free']}/{summary['total_slots']}"
        f"，泄漏 {summary['total_leaked']}"
    )


def main() -> None:
    args = parse_args()
    pool = EnvironmentPool(
        host=args.host,
        base_port=args.base_port,
        workers=args.workers,
        # 客户端信号量在这个脚本里没有意义（不跑回合），关键是 env_slots 要对：
        # 回收和探测都按它决定索引空间有多大。
        slots_per_worker=args.env_slots,
        env_slots=args.env_slots,
    )
    try:
        pool.wait_until_ready(timeout=args.ready_timeout)
    except EnvironmentServiceError as exc:
        # 端口没监听和「全部 slot 都漏了」在探测结果上长得一样（都是 0 可用），
        # 必须先分开，否则会拿着一份错误的诊断去调并发。
        raise SystemExit(f"环境池不可用，先体检不了：\n{exc}")

    before = report(measure(pool), args.env_slots)
    render(before, "体检")
    result: dict[str, object] = {"before": before, "reclaimed": None, "after": None}

    if args.reclaim and before["total_leaked"]:
        print("\n回收泄漏的租约（要求同端口段上没有 rollout 在跑）……")
        reclaimed = pool.reconcile()
        result["reclaimed"] = {url: slots for url, slots in reclaimed.items() if slots}
        after = report(measure(pool), args.env_slots)
        result["after"] = after
        render(after, "回收后")
        recovered = after["total_free"] - before["total_free"]  # type: ignore[operator]
        print(f"  恢复 {recovered} 个 slot")
    elif args.reclaim:
        print("\n没有泄漏，不需要回收。")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n结果写入 {args.json}")

    final = result["after"] or before
    if final["total_leaked"]:  # type: ignore[index]
        print(
            f"\n仍有 {final['total_leaked']} 个 slot 泄漏。"  # type: ignore[index]
            "并发按可用容量设，不要按 worker × slot 设。"
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
