#!/usr/bin/env python3
"""统计 Reward v3 各软匹配维度在真实任务池中的激活率。

结论记录在 docs/environment-notes.md。调整奖励权重前后可用本脚本做对照。
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "third_party/ShopSimulator/shop_env"
PRODUCTS = SHOP_ENV / "data/fine_items_eval_train_all.json.gz"

# reward.py 的 DIMENSION_WEIGHTS，复制而非导入：本脚本要能在环境 venv 之外跑。
DIMENSION_WEIGHTS = {
    "brand": 0.35,
    "model": 0.25,
    "core_functions": 0.25,
    "key_options": 0.15,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=3000, help="抽样任务数；0 表示全量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--products", type=Path, default=PRODUCTS)
    parser.add_argument("--json-out", type=Path, help="将结果写入 JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.path.insert(0, str(SHOP_ENV))
    from web_agent_site.engine.constraints import explicit_budget_from_instruction
    from web_agent_site.engine.reward_features import compile_reward_features

    with gzip.open(args.products, "rt", encoding="utf-8") as handle:
        products = json.load(handle)

    if args.sample and args.sample < len(products):
        products = random.Random(args.seed).sample(products, args.sample)

    active = Counter()
    attribute_counts = Counter()
    dimension_hits = Counter()
    total = 0
    budget_explicit = 0
    failures = 0

    for item in products:
        for record in item.get("instructions") or []:
            total += 1
            try:
                features = compile_reward_features(record, item)
            except Exception:
                failures += 1
                continue

            hits = {
                "brand": bool(features["expected_brand"]),
                "model": bool(features["expected_model"]),
                "core_functions": bool(features["expected_core_functions"]),
                "key_options": bool(features["required_options_by_key"]),
            }
            for name, hit in hits.items():
                dimension_hits[name] += int(hit)

            weight = sum(DIMENSION_WEIGHTS[n] for n, hit in hits.items() if hit)
            active[round(weight, 2)] += 1

            attribute_counts[len(record.get("attributes") or [])] += 1
            if explicit_budget_from_instruction(record.get("instruction") or "") is not None:
                budget_explicit += 1

    scored = total - failures
    if not scored:
        raise SystemExit("no instructions scored")

    report = {
        "sampled_instructions": total,
        "scored": scored,
        "compile_failures": failures,
        "dimension_activation": {
            name: {
                "design_weight": DIMENSION_WEIGHTS[name],
                "activation_rate": dimension_hits[name] / scored,
            }
            for name in DIMENSION_WEIGHTS
        },
        "active_weight_distribution": {
            str(weight): count for weight, count in sorted(active.items())
        },
        "explicit_budget_rate": budget_explicit / scored,
        "attribute_count_distribution": {
            str(count): n for count, n in sorted(attribute_counts.items())
        },
    }

    print(f"scored instructions: {scored} (compile failures: {failures})")
    print("\ndimension activation:")
    for name, entry in report["dimension_activation"].items():
        print(
            f"  {name:<15} weight={entry['design_weight']:.2f} "
            f"active={entry['activation_rate'] * 100:5.1f}%"
        )
    print("\nactive_weight distribution:")
    for weight, count in sorted(active.items(), key=lambda kv: -kv[1]):
        print(f"  {weight:.2f}  {count:6d}  {count / scored * 100:5.1f}%")
    print(f"\nexplicit budget: {budget_explicit / scored * 100:.1f}%")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
