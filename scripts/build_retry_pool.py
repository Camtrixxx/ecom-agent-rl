#!/usr/bin/env python3
"""从 verdicts 里挑出零 accepted 的任务，建一个重采池。

为什么只挑零 accepted 的：SFT 数据集按 1 任务 1 轨迹构建，`select()` 也按 task_id
去重。只重采一条可用数据都没有的任务，重采成功也只会给该任务补上第一条轨迹，
不会出现「同任务多条轨迹放大该任务权重」——roadmap 阶段 B 末尾担心的那个问题
在这里结构性地不成立，不是靠去重兜住的。

    python scripts/build_retry_pool.py \
        --pool data/task_pools/sft_train.jsonl \
        --verdicts data/sft/verdicts.jsonl \
        --out data/task_pools/sft_train_retry.jsonl
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/task_pools/sft_train.jsonl")
    ap.add_argument("--verdicts", default="data/sft/verdicts.jsonl")
    ap.add_argument("--out", default="data/task_pools/sft_train_retry.jsonl")
    args = ap.parse_args()

    pool_path, verdicts_path, out_path = (ROOT / args.pool), (ROOT / args.verdicts), (ROOT / args.out)
    pool = read_jsonl(pool_path)
    verdicts = read_jsonl(verdicts_path)

    accepted = {v["task_id"] for v in verdicts if v.get("accepted")}
    judged = {v["task_id"] for v in verdicts}
    retry = [t for t in pool if t["task_id"] not in accepted]

    # 池里有而 verdicts 没判过的任务：采集当时就没跑到它。也该重采，但要单独报——
    # 它和「跑了没通过」是两回事，混在一起会让重采成功率的分母失去意义。
    unjudged = [t for t in retry if t["task_id"] not in judged]

    out_path.write_text(
        "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in retry), encoding="utf-8"
    )

    by_type = collections.Counter(
        v["reward_type"] for v in verdicts if v["task_id"] not in accepted
    )
    # 那 74 条 gold_purchase 是「题做对了、被格式挡下」，重采成功率应当最高，单独记。
    gold_rejections = collections.Counter(
        v.get("rejection_count", 0)
        for v in verdicts
        if v["task_id"] not in accepted and v["reward_type"] == "gold_purchase"
    )

    meta = {
        "schema_version": "ecom-retry-pool-v1",
        "outputs": {
            "retry_pool": {
                "path": str(out_path.relative_to(ROOT)),
                "tasks": len(retry),
                "sha256": sha256_of(out_path),
            }
        },
        "provenance": {
            "source_pool": str(pool_path.relative_to(ROOT)),
            "source_pool_sha256": sha256_of(pool_path),
            "source_pool_tasks": len(pool),
            "verdicts": str(verdicts_path.relative_to(ROOT)),
            "verdicts_sha256": sha256_of(verdicts_path),
            "accepted_tasks": len(accepted),
        },
        "report": {
            "retry_tasks": len(retry),
            "unjudged_tasks": len(unjudged),
            "reward_types": dict(by_type.most_common()),
            "gold_purchase_rejection_counts": {str(k): v for k, v in sorted(gold_rejections.items())},
        },
    }
    meta_path = out_path.with_name(out_path.stem + ".metadata.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"重采池 {len(retry)} 个任务 → {out_path.relative_to(ROOT)}")
    print(f"  其中从未被判过的 {len(unjudged)} 个")
    print(f"  血缘 → {meta_path.relative_to(ROOT)}")
    for k, n in by_type.most_common():
        print(f"    {k:38s} {n}")


if __name__ == "__main__":
    main()
