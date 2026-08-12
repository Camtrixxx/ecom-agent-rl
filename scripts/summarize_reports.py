#!/usr/bin/env python3
"""把一批 report_metrics.py 的 json 汇成一张表。

E1（9 个检查点）和 E2（4 份 SFT 权重）都产出一堆 `*.report.json`，人肉对比容易只盯
着均值挑最大的那个。这个脚本把配对区间一起打出来，并**按预注册的并列规则**给结论：
若第一名与第二名的差值小于分辨率半宽，就报「并列」，不报排名。

    python3 scripts/summarize_reports.py outputs/rollouts/ckpt_eval --halfwidth 3.2
    python3 scripts/summarize_reports.py outputs/rollouts/sft_variants --halfwidth 3.2

半宽要显式传：它是评测设计的产物（n 与 k），不该由脚本猜。E1/E2 都是 n=200、k=4，
预注册里算的是 ±3.2 pp。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SUFFIX = ".report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, help="放 *.report.json 的目录")
    parser.add_argument("--halfwidth", type=float, required=True,
                        help="这套配置的分辨率半宽（pp），用于并列判定")
    parser.add_argument("--order", nargs="*", default=None,
                        help="想要的展示顺序（不给就按文件名排）")
    args = parser.parse_args()

    reports = {}
    for path in sorted(args.dir.glob("*" + SUFFIX)):
        with path.open(encoding="utf-8") as handle:
            reports[path.name[: -len(SUFFIX)]] = json.load(handle)
    if not reports:
        raise SystemExit(f"{args.dir} 下没有 *{SUFFIX}")

    names = [n for n in (args.order or sorted(reports)) if n in reports]

    # 表头单独排：CJK 在终端占两列，用 str 宽度对齐会歪。这里按显示宽度手写。
    header = ("model         success_rate          95% CI        delta      "
              "        delta CI  tasks")
    print(header)
    print("-" * len(header))

    deltas = []
    for name in names:
        report = reports[name]
        rate = report["overall"]["success_rate"]
        span = f"[{rate['ci_low'] * 100:.2f}, {rate['ci_high'] * 100:.2f}]"
        row = f"{name:<14}{rate['mean'] * 100:>9.2f}%{span:>20}"

        paired = report.get("paired_vs_baseline")
        if paired:
            delta = paired["delta"] * 100
            dspan = f"[{paired['ci_low'] * 100:+.2f}, {paired['ci_high'] * 100:+.2f}]"
            row += f"{delta:>+13.2f}p{dspan:>20}{paired['paired_tasks']:>7}"
            deltas.append((name, delta))
        else:
            row += f"{'(baseline)':>14}{'':>20}{report['overall']['tasks']:>7}"
        print(row)

    if len(deltas) < 2:
        return

    deltas.sort(key=lambda item: -item[1])
    gap = deltas[0][1] - deltas[1][1]
    print("-" * len(header))
    print(f"最高 {deltas[0][0]} {deltas[0][1]:+.2f}pp，次高 {deltas[1][0]} {deltas[1][1]:+.2f}pp，"
          f"相差 {gap:.2f}pp，分辨率半宽 ±{args.halfwidth}pp")
    tied = [name for name, delta in deltas if deltas[0][1] - delta < args.halfwidth]
    if len(tied) > 1:
        print(f"→ 按预注册规则判为并列：{'、'.join(tied)}。不报排名。")
    else:
        print(f"→ {deltas[0][0]} 领先超过分辨率，可以报排名。")


if __name__ == "__main__":
    main()
