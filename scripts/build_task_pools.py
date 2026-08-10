#!/usr/bin/env python3
"""切出 SFT / GRPO / 评测三个互不重叠的任务池，并写血缘。

用法：
    python scripts/build_task_pools.py            # 写入 data/
    python scripts/build_task_pools.py --dry-run  # 只看分布，不落盘

产物（data/ 不入 git，靠 metadata.json 的 sha256 复现）：
    data/task_pools/{sft_train,sft_val,grpo_train,grpo_val,evaluation}.jsonl
    data/task_pools/metadata.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecom_agent_rl.data.task_pool import (  # noqa: E402
    ENVIRONMENT,
    EXPECTED_PRODUCTS,
    PRODUCT_SHA256,
    REWARD,
    SCHEMA_VERSION,
    distribution,
    load_tasks,
    split_tasks,
    write_split,
)

DEFAULT_PRODUCTS = ROOT / "third_party/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
DEFAULT_OUT = ROOT / "data/task_pools"

# roadmap 锁定的规模。评测 500 题，比参考实现的 200 题大一档，用于支撑多次采样后的
# 置信区间；三池合计 7,000 条，占 23,421 的 29.9%，留足后续加量空间。
SIZES = {
    "sft_train": 3000,
    "sft_val": 500,
    "grpo_train": 3000,
    "grpo_val": 200,
    "evaluation": 500,
}
DEFAULT_SEED = 20260810


def display_path(path: Path) -> str:
    """血缘里记相对仓库根的路径；`--out` 指到仓库外（比如复现校验）时退回绝对路径。"""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--products", type=Path, default=DEFAULT_PRODUCTS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true", help="只打印分布，不写文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.products.exists():
        raise SystemExit(f"商品数据不存在: {args.products}\n先跑 bash scripts/setup_environment.sh")

    tasks = load_tasks(args.products)
    if len(tasks) != EXPECTED_PRODUCTS:
        print(
            f"warning: 商品数 {len(tasks)} != 预期 {EXPECTED_PRODUCTS}，"
            "数据版本可能变了，血缘里记录实际值",
            file=sys.stderr,
        )

    splits = split_tasks(tasks, SIZES, args.seed)

    print(f"任务池 {len(tasks)} 条，切出 {sum(SIZES.values())} 条（占 "
          f"{sum(SIZES.values()) / len(tasks) * 100:.1f}%）\n")

    overall = distribution(tasks)["difficulty"]
    overall_total = sum(overall.values())
    buckets = list(overall)

    header = f"{'池子':<12}{'条数':>7}  " + "  ".join(f"{b:>7}" for b in buckets)
    print(header)
    print("-" * len(header))
    print(
        f"{'(全池)':<12}{overall_total:>7}  "
        + "  ".join(f"{overall[b] / overall_total * 100:>6.1f}%" for b in buckets)
    )
    for name in SIZES:
        dist = distribution(splits[name])["difficulty"]
        total = sum(dist.values())
        print(
            f"{name:<12}{total:>7}  "
            + "  ".join(f"{dist.get(b, 0) / total * 100:>6.1f}%" for b in buckets)
        )

    domains = {name: len(distribution(splits[name])["domain"]) for name in SIZES}
    print(f"\n品类覆盖（全池 {len(distribution(tasks)['domain'])} 类）："
          + "，".join(f"{k}={v}" for k, v in domains.items()))

    if args.dry_run:
        print("\n--dry-run：未写文件")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    entries = {}
    for name, items in splits.items():
        path = args.out / f"{name}.jsonl"
        entries[name] = {
            "path": display_path(path),
            "tasks": len(items),
            "sha256": write_split(path, items),
            "distribution": distribution(items),
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "environment": ENVIRONMENT,
        "reward": REWARD,
        "provenance": {
            "seed": args.seed,
            "products": display_path(args.products),
            "products_count": len(tasks),
            "products_sha256_expected": PRODUCT_SHA256,
            "stratified_by": ["domain_en_short", "attribute_count_bucket"],
            "disjoint": True,
            "note": "task_id 即商品下标；已校验每商品恰好 1 条 instruction，无商品级泄漏",
        },
        "splits": entries,
    }
    meta_path = args.out / "metadata.json"
    meta_text = json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True)
    meta_path.write_text(meta_text + "\n", encoding="utf-8")

    print(f"\n写入 {args.out}/")
    for name, entry in entries.items():
        print(f"  {name + '.jsonl':<22} {entry['tasks']:>5} 条  {entry['sha256'][:16]}")
    print(f"  {'metadata.json':<22}       {hashlib.sha256(meta_text.encode()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
