#!/usr/bin/env python3
"""#14 步数对照臂的判定：把 docs/stepmatch-preregistration.md 的规则原样执行。

**写于 2026-08-16 00:50，s44 还没开始训、sft_v2_resample3 还没起。** s42/s43 的数已经看过
（0.6507 / 0.6465），s44 和第三份锚是盲的。分析代码在数据齐之前写，等于不给自己留挑口径
的余地。披露：这不是全盲，2/3 已见。

    .venv/bin/python scripts/analyze_stepmatch.py

## 主判定是「同窗口锚」，不是对已发布的 sft_v2 三条

两组的评测日期是**整组错开**的（stepmatch 在 08-15/16 三个窗口，已发布的 sft_v2/_s43/_s44
全在 08-14）。整组错开的跨 session 漂移不是噪声而是**偏置**，加多少个 seed 都平均不掉。
s42 实测符号会翻：对 08-14 基线 −1.19 pp，对同窗口的 sft_v2_resample +1.06 pp。

所以每个 stepmatch seed 配一份**在它自己评测窗口内重评的 sft_v2**，逐窗口作差，再对三个
差值做 n=3 的 t 检验。

**这个区间不含 v2 侧的训练 run 方差**——三份锚是同一份权重重评三次。这是刻意的（见预注册
文档），引用时必须带这句限定。

## 两个不写死的地方，和一个写死的

- 配对差从 report_metrics.py 现算，不从已发布的报告里读：已发布的 s42 报告配的是 08-14 的
  sft_v2，不是同窗口锚，直接读会读到错的基线。
- 报告写到 outputs/rollouts/stepmatch_anchored/，**不覆盖已发布的 .report.json**。
- t₂,₀.₉₇₅ = 4.30 和 n = 3 是常量，没有 --seeds 参数。预注册没写"不够就补第 4 个 seed"，
  但也没允许；能接第 4 个 seed 就等于把 n 变成看完数据再定的东西。
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "outputs" / "rollouts"
OUTDIR = ROLLOUTS / "stepmatch_anchored"
POOL = ROOT / "data" / "task_pools" / "evaluation.jsonl"
PY = ROOT / ".venv" / "bin" / "python"

T_CRIT = 4.30  # df=2 双侧 0.975
N_SEEDS = 3

# (seed, stepmatch 轨迹, 同窗口锚, 该窗口的大致时刻)
WINDOWS = [
    (42, "sft_stepmatch_s42.jsonl", "sft_v2_resample.jsonl", "08-15 20:40"),
    (43, "sft_stepmatch_s43.jsonl", "sft_v2_resample2.jsonl", "08-15 23:50"),
    (44, "sft_stepmatch_s44.jsonl", "sft_v2_resample3.jsonl", "08-16 03:00"),
]

# 次判定：对已发布的 08-14 那三条。照登但不作主判定。
NAIVE = [
    (42, "sft_stepmatch_s42.jsonl", "sft_v2.jsonl"),
    (43, "sft_stepmatch_s43.jsonl", "sft_v2_s43.jsonl"),
    (44, "sft_stepmatch_s44.jsonl", "sft_v2_s44.jsonl"),
]


def paired_delta(treatment: str, baseline: str, tag: str) -> dict | None:
    """跑一次 report_metrics 取配对差。缺文件返回 None（而不是当 0 用）。"""
    t, b = ROLLOUTS / treatment, ROLLOUTS / baseline
    if not (t.exists() and b.exists()):
        missing = [str(p.name) for p in (t, b) if not p.exists()]
        print(f"  缺文件，跳过 {tag}：{', '.join(missing)}")
        return None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{tag}.report.json"
    if not out.exists():
        r = subprocess.run(
            [str(PY), "scripts/report_metrics.py", "--trajectories", str(t),
             "--baseline", str(b), "--pool", str(POOL), "--json", str(out)],
            cwd=ROOT, capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  report_metrics 失败 {tag} exit={r.returncode}\n{r.stderr[-500:]}")
            return None
    return json.loads(out.read_text())["paired_vs_baseline"]


def combine(deltas: list[float], label: str) -> None:
    n = len(deltas)
    mean = statistics.mean(deltas)
    if n < 2:
        print(f"{label}：只有 {n} 个窗口，出不了区间。均值 {mean * 100:+.2f} pp")
        return
    s = statistics.stdev(deltas)
    se = s / (n ** 0.5)
    lo, hi = mean - T_CRIT * se, mean + T_CRIT * se
    print(f"{label}：n={n}  均值 **{mean * 100:+.2f} pp**  组内 s={s * 100:.2f} pp  "
          f"95% 区间 [{lo * 100:+.2f}, {hi * 100:+.2f}] pp")
    if n != N_SEEDS:
        print(f"  !! n={n} ≠ 预注册的 {N_SEEDS}，t={T_CRIT} 是 df=2 的值，这个区间**不合法**，"
              f"只当中途速览")
        return
    crosses = lo < 0 < hi
    print(f"  区间{'跨' if crosses else '不跨'} 0 → ", end="")
    if crosses:
        print("**匹配步数之后 v2 数据没有可测的额外收益**。D2 SFT 侧的 +5.42 pp\n"
              "     主要是 LR 积分（多走 21 步），不是教师数据质量。")
    elif hi < 0:
        print(f"v2 数据确有额外收益，量级 {abs(mean) * 100:.2f} pp；余下的归 LR 积分。")
    else:
        print("v1 数据在 102 步下反而更好。**先怀疑漂移没被锚干净**——看下面锚自己的离散度。")


def main() -> int:
    print("=" * 78)
    print("#14 步数对照臂 · 主判定（同窗口锚）")
    print("=" * 78)
    rows, deltas = [], []
    for seed, treat, anchor, when in WINDOWS:
        d = paired_delta(treat, anchor, f"s{seed}_anchored")
        if d is None:
            continue
        rows.append((seed, when, d))
        deltas.append(d["delta"])
        print(f"  s{seed} ({when})  stepmatch {d['treatment_success_rate']:.4f}  "
              f"锚 {d['baseline_success_rate']:.4f}  配对差 {d['delta'] * 100:+.2f} pp  "
              f"[{d['ci_low'] * 100:+.2f}, {d['ci_high'] * 100:+.2f}]  配对题 {d['paired_tasks']}")
    print()
    if deltas:
        combine(deltas, "主判定")
    else:
        print("主判定：一个窗口都没齐，什么都算不了。")

    # 锚自己的离散度 = 同一份权重在三个窗口的成功率散布 = 窗口漂移的直接估计。
    # 预注册的判读分支 3 要用它，而且它本身就是漂移量级的第三次独立测量。
    print()
    print("锚自身离散度（同一份 sft_v2 权重、三个窗口重评 → 窗口漂移的直接估计）：")
    anchor_rates = []
    for _, _, anchor, when in WINDOWS:
        p = ROLLOUTS / anchor.replace(".jsonl", ".report.json")
        if not p.exists():
            print(f"  {anchor:<26} 报告未出")
            continue
        r = json.loads(p.read_text())["overall"]["success_rate"]["mean"]
        anchor_rates.append(r)
        print(f"  {anchor:<26} {r:.4f}  ({when})")
    if len(anchor_rates) >= 2:
        s_anchor = statistics.stdev(anchor_rates)
        span = max(anchor_rates) - min(anchor_rates)
        print(f"  → s = {s_anchor * 100:.2f} pp，极差 {span * 100:.2f} pp"
              f"（同权重同池同温度，差别全是窗口）")
        if s_anchor > 0.01:
            print("  !! 锚自己的 s 就超过 1 pp：同窗口也压不住漂移，这条臂在现有评测精度下"
                  "**测不动**，\n     该记成「测不动」而不是硬报一个数（预注册判读分支 3）。")

    print()
    print("=" * 78)
    print("次判定（对已发布的 08-14 三条）· 照登，不作主判定：两组评测日期整组错开")
    print("=" * 78)
    naive_deltas = []
    for seed, treat, base in NAIVE:
        d = paired_delta(treat, base, f"s{seed}_naive")
        if d is None:
            continue
        naive_deltas.append(d["delta"])
        print(f"  s{seed}  stepmatch {d['treatment_success_rate']:.4f}  "
              f"08-14 基线 {d['baseline_success_rate']:.4f}  配对差 {d['delta'] * 100:+.2f} pp")
    print()
    if naive_deltas:
        combine(naive_deltas, "次判定")
    if len(deltas) == len(naive_deltas) == N_SEEDS:
        gap = statistics.mean(deltas) - statistics.mean(naive_deltas)
        print(f"\n两个判定相差 {gap * 100:+.2f} pp。**这个差就是组间窗口偏置的估计值**，"
              f"是漂移量级的又一次\n测量，不管两个判定是否一致都要记下来。")

    print()
    print("预注册的 08-14 期望是 **+4.4 pp**（stepmatch 应显著低于 sft_v2，即差值负 4.4 pp 量级）。")
    if len(deltas) == N_SEEDS:
        m = statistics.mean(deltas)
        if abs(m) <= 0.015:
            print(f"实测 {m * 100:+.2f} pp，在 ±1.5 pp 内 → 那条期望**被证伪**，"
                  f"roadmap 里要明写证伪，\n不是「未达预期」。")
        else:
            print(f"实测 {m * 100:+.2f} pp。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
