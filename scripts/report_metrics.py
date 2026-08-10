#!/usr/bin/env python3
"""把 rollout 轨迹汇总成评测报告。

用法：
    # 单个实验
    python scripts/report_metrics.py --trajectories outputs/rollouts/baseline.jsonl

    # 分层（难度桶 + 品类）
    python scripts/report_metrics.py --trajectories outputs/rollouts/baseline.jsonl \\
        --pool data/task_pools/evaluation.jsonl

    # 和 baseline 配对比较
    python scripts/report_metrics.py --trajectories outputs/rollouts/sft.jsonl \\
        --baseline outputs/rollouts/baseline.jsonl \\
        --pool data/task_pools/evaluation.jsonl

`--json` 把完整报告写盘，文本输出只是给人看的摘要。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecom_agent_rl.evaluation.metrics import (  # noqa: E402
    format_summary,
    load_outcomes,
    load_strata,
    paired_comparison,
    stratified,
    summarize,
)

# 分层轴：两者都由 build_task_pools.py 写进池文件。
STRATA_FIELDS = ("difficulty", "domain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--trajectories", type=Path, required=True, help="待评测的轨迹 jsonl")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="对照轨迹 jsonl，给了就做按 task_id 配对比较")
    parser.add_argument("--pool", type=Path, default=None,
                        help="任务池 jsonl，给了就出分层报告")
    parser.add_argument("--json", type=Path, default=None, help="完整报告写到这个文件")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0, help="bootstrap 随机种子")
    parser.add_argument("--min-tasks", type=int, default=10,
                        help="分层桶少于这个题数就标 underpowered")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.trajectories.exists():
        raise SystemExit(f"轨迹文件不存在: {args.trajectories}")

    outcomes = load_outcomes(args.trajectories)
    if not outcomes:
        raise SystemExit(f"{args.trajectories} 里没有轨迹")

    overall = summarize(outcomes, confidence=args.confidence, seed=args.seed)
    report: dict = {"trajectories_file": str(args.trajectories), "overall": overall}

    print(f"== {args.trajectories.name} ==")
    print(format_summary(overall))

    if args.pool:
        if not args.pool.exists():
            raise SystemExit(f"任务池不存在: {args.pool}")
        # 两个轴都先读出来再打印：否则第一个轴的表已经刷屏了，第二个轴才报缺字段。
        try:
            axes = {field: load_strata(args.pool, field) for field in STRATA_FIELDS}
        except KeyError as exc:
            raise SystemExit(f"分层失败: {exc.args[0]}") from exc

        report["stratified"] = {}
        for field, strata in axes.items():
            buckets = stratified(
                outcomes, strata, confidence=args.confidence,
                seed=args.seed, min_tasks=args.min_tasks,
            )
            report["stratified"][field] = buckets
            print(f"\n== 按 {field} 分层 ==")
            print(f"{'桶':<14} {'题数':>5} {'成功率':>8}  {'95% CI':^18}  {'错买率':>7}")
            print("-" * 62)
            for name, summary in buckets.items():
                success, wrong = summary["success_rate"], summary["wrong_purchase_rate"]
                flag = "  ⚠ 题数不足" if summary["underpowered"] else ""
                print(
                    f"{name:<14} {summary['tasks']:>5} {success['mean']:>8.4f}  "
                    f"[{success['ci_low']:.4f}, {success['ci_high']:.4f}]  "
                    f"{wrong['mean']:>7.4f}{flag}"
                )

    if args.baseline:
        if not args.baseline.exists():
            raise SystemExit(f"对照轨迹不存在: {args.baseline}")
        baseline = load_outcomes(args.baseline)
        paired = paired_comparison(
            baseline, outcomes, confidence=args.confidence, seed=args.seed
        )
        report["paired_vs_baseline"] = {
            "baseline_file": str(args.baseline), **paired,
        }
        print(f"\n== 对比 {args.baseline.name}（按 task_id 配对）==")
        if not paired["paired_tasks"]:
            print(paired["note"])
        else:
            print(f"配对题数        {paired['paired_tasks']}")
            print(f"baseline 成功率 {paired['baseline_success_rate']:.4f}")
            print(f"本次成功率      {paired['treatment_success_rate']:.4f}")
            print(
                f"差值            {paired['delta']:+.4f}  "
                f"[{paired['ci_low']:+.4f}, {paired['ci_high']:+.4f}]"
            )
            print(
                "结论            "
                + ("区间不跨 0，是真实差异" if paired["significant"]
                   else "区间跨 0，还不能说有差异")
            )
            if paired["baseline_only_tasks"] or paired["treatment_only_tasks"]:
                print(
                    f"未配对          baseline 独有 {paired['baseline_only_tasks']}，"
                    f"本次独有 {paired['treatment_only_tasks']}"
                )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n完整报告写入 {args.json}")


if __name__ == "__main__":
    main()
