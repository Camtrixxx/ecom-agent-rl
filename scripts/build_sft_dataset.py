#!/usr/bin/env python3
"""把教师轨迹筛成 SFT 训练集，并写血缘。

用法：
    python scripts/build_sft_dataset.py \
        --trajectories outputs/teacher/sft_train.jsonl \
        --out-dir data/sft

    # 只看审计报告，不落盘（调口径时用这个）
    python scripts/build_sft_dataset.py --trajectories ... --dry-run

产物（data/ 不入 git，靠 metadata.json 的 sha256 复现）：
    <out-dir>/train.jsonl        接受的轨迹，一条 = 一个多轮样本
    <out-dir>/verdicts.jsonl     每条轨迹的筛选结论（含被拒原因），用于回溯
    <out-dir>/metadata.json      血缘 + 审计报告

为什么把 verdicts 也写盘：接受率是个会被反复质疑的数字（"是不是筛太严了"）。
只存最终数据集的话，回答这个问题就得重跑一遍筛选；存下每条的判定和全部原因，
就能直接查"放宽某一项能多收多少"。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecom_agent_rl.data.sft import (  # noqa: E402
    DEFAULT_MAX_REJECTIONS,
    GOLD_ONLY,
    SUCCESS_TYPES,
    build_row,
    read_trajectories,
    select,
)
from ecom_agent_rl.environment.observation import OBSERVATION_VERSION  # noqa: E402
from ecom_agent_rl.environment.tools import TOOL_SCHEMAS  # noqa: E402
from ecom_agent_rl.rollout.prompt import SYSTEM_PROMPT  # noqa: E402

SCHEMA_VERSION = "ecom-sft-dataset-v1"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rel(path: Path) -> str:
    """尽量写相对路径（血缘要能跨机器读），落在仓库外时退回绝对路径。"""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True,
                        help="教师轨迹 jsonl，可给多个（会按顺序合并）")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sft")
    parser.add_argument("--gold-only", action="store_true",
                        help="只收 gold_purchase（参考实现口径）；默认含 valid_alternative")
    parser.add_argument("--max-rejections", type=int, default=DEFAULT_MAX_REJECTIONS,
                        help=f"被拒动作次数上限，默认 {DEFAULT_MAX_REJECTIONS}（实测平台）")
    parser.add_argument("--held-out", type=Path, default=None,
                        help="要排除的任务池 jsonl（防止训练集碰到验证题）")
    parser.add_argument("--dry-run", action="store_true", help="只报告，不落盘")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trajectories = []
    for path in args.trajectories:
        rows = read_trajectories(path)
        print(f"读入 {len(rows):>5} 条  {path}")
        trajectories.extend(rows)

    held_out: frozenset[int] = frozenset()
    if args.held_out:
        held_out = frozenset(
            int(json.loads(line)["task_id"])
            for line in args.held_out.open()
            if line.strip()
        )
        print(f"排除 held-out {len(held_out)} 题  {args.held_out}")

    accept_types = GOLD_ONLY if args.gold_only else SUCCESS_TYPES
    verdicts, report = select(
        trajectories,
        accept_types=accept_types,
        max_rejections=args.max_rejections,
        held_out_task_ids=held_out,
    )

    print(f"\n口径: {'gold-only' if args.gold_only else 'gold + valid_alternative'}"
          f"  被拒上限: {args.max_rejections}")
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))

    if args.dry_run:
        print("\n--dry-run：未落盘")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = args.out_dir / "train.jsonl"
    with train.open("w") as handle:
        for trajectory, verdict in zip(trajectories, verdicts):
            if verdict.accepted:
                row = build_row(trajectory, TOOL_SCHEMAS)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    verdict_path = args.out_dir / "verdicts.jsonl"
    with verdict_path.open("w") as handle:
        for verdict in verdicts:
            handle.write(json.dumps(asdict(verdict), ensure_ascii=False) + "\n")

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "provenance": {
            "trajectories": [str(p) for p in args.trajectories],
            "trajectories_sha256": {
                str(p): sha256_of(p) for p in args.trajectories
            },
            "held_out_pool": str(args.held_out) if args.held_out else None,
            "held_out_tasks": len(held_out),
            # 系统提示词与工具 schema 都进 prompt，换了就是换了一份数据集。
            "system_prompt_sha256": hashlib.sha256(
                SYSTEM_PROMPT.encode()
            ).hexdigest(),
            "tool_schemas_sha256": hashlib.sha256(
                json.dumps(TOOL_SCHEMAS, ensure_ascii=False, sort_keys=True).encode()
            ).hexdigest(),
        },
        "selection": {
            "accept_types": sorted(accept_types),
            "max_rejections": args.max_rejections,
        },
        "report": report.to_dict(),
        "outputs": {
            "train": {
                "path": _rel(train),
                "samples": report.accepted,
                "sha256": sha256_of(train),
            },
            "verdicts": {
                "path": _rel(verdict_path),
                "sha256": sha256_of(verdict_path),
            },
        },
    }
    meta_path = args.out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(f"\n写入 {report.accepted} 个样本 → {train}")
    print(f"血缘 → {meta_path}")


if __name__ == "__main__":
    main()
