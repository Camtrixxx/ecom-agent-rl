#!/usr/bin/env python3
"""#27 的判定：把 docs/dayeval-preregistration.md 的规则原样执行。

**写于 2026-08-16 04:30，三条腿一条都没起。** 全盲。

    .venv/bin/python scripts/analyze_concurrency.py

## 先看自变量有没有被操纵成，再看成功率

这条臂拿 `--concurrency` 代理"vLLM 的在飞批构成"。代理可能失效：环境侧每段只有 8 个
GIL 绑死的 worker，c=32 时大部分 rollout 线程可能在等环境而不在等 LLM，于是实际批大小
根本没跟着涨。**那种情况下阴性结果什么都不说明**，必须记成"没能操纵自变量"。

所以先从 vLLM 自己的日志里抽 `Running: N reqs`（每 10 s 一条），算每条腿的实际批大小。
预注册的门槛：c=32 的均值批大小要**至少是 c=8 的 2 倍**，否则判定停在"未能操纵"。

## 为什么不用 t 检验

三条腿是同一份权重、同一批 500 题，所以两两之间可以按 task_id 配对，用 report_metrics 的
bootstrap 配对区间——比对三个点做 n=3 的 t 检验紧得多，而且这里没有"训练 run 方差"要含。
三对区间不是独立的（共 3 条腿），所以**不做多重比较校正也不合并**，逐对照登。
"""

from __future__ import annotations

import itertools
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "outputs" / "rollouts"
LOGS = ROOT / "outputs" / "logs"
OUTDIR = ROLLOUTS / "concurrency"
POOL = ROOT / "data" / "task_pools" / "evaluation.jsonl"
PY = ROOT / ".venv" / "bin" / "python"

CONCURRENCIES = [8, 16, 32]
TARGET_ROWS = 2000
RUNNING_RE = re.compile(r"Running:\s*(\d+)\s*reqs")


def leg(c: int) -> str:
    return f"cc_sft_v2_c{c}"


def batch_stats(c: int) -> dict | None:
    """从 vLLM 日志抽实际在飞请求数。

    丢掉 0：rollout 开始前和收尾后 vLLM 是空转的，那些 0 会把均值压低到没有意义。
    """
    p = LOGS / f"vllm_{leg(c)}.log"
    if not p.exists():
        return None
    vals = [int(m.group(1)) for m in RUNNING_RE.finditer(p.read_text(errors="replace"))]
    nz = [v for v in vals if v > 0]
    if not nz:
        return None
    nz_sorted = sorted(nz)
    return {
        "n": len(nz),
        "mean": statistics.mean(nz),
        "p50": nz_sorted[len(nz_sorted) // 2],
        "p90": nz_sorted[int(len(nz_sorted) * 0.9)],
        "max": max(nz),
    }


def rate(name: str) -> float | None:
    p = ROLLOUTS / f"{name}.report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["overall"]["success_rate"]["mean"]


def rows(name: str) -> int:
    f = ROLLOUTS / f"{name}.jsonl"
    if not f.exists():
        return 0
    with f.open() as fh:
        return sum(1 for _ in fh)


def strict_ok(name: str) -> bool:
    f = ROLLOUTS / f"{name}.jsonl"
    if not f.exists():
        return False
    r = subprocess.run(["grep", "-c", '"tolerant_parse"', str(f)],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0) == 0


def paired(treat: str, base: str, tag: str) -> dict | None:
    t, b = ROLLOUTS / f"{treat}.jsonl", ROLLOUTS / f"{base}.jsonl"
    if not (t.exists() and b.exists()):
        return None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{tag}.report.json"
    if not out.exists():
        r = subprocess.run(
            [str(PY), "scripts/report_metrics.py", "--trajectories", str(t),
             "--baseline", str(b), "--pool", str(POOL), "--json", str(out)],
            cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  report_metrics 失败 {tag} exit={r.returncode}\n{r.stderr[-400:]}")
            return None
    return json.loads(out.read_text())["paired_vs_baseline"]


def main() -> int:
    print("=" * 78)
    print("#27 只改 concurrency · 判定")
    print("=" * 78)

    # ---- 闸门：腿齐不齐、口径严不严 ----
    incomplete = [c for c in CONCURRENCIES if rows(leg(c)) < TARGET_ROWS]
    if incomplete:
        for c in CONCURRENCIES:
            print(f"  c={c:<3} {rows(leg(c))}/{TARGET_ROWS}")
        print(f"\n缺腿（c={incomplete}）→ 不出判定。")
        return 1
    bad = [c for c in CONCURRENCIES if not strict_ok(leg(c))]
    if bad:
        print(f"!! c={bad} 的轨迹里有 tolerant_parse，口径不一致 → 不出判定，重跑那条腿。")
        return 1

    # ---- 第一步：自变量操纵成了吗 ----
    print()
    print("-" * 78)
    print("第一步 · 实际在飞批大小（vLLM 自己打的 Running: N reqs，每 10 s 一条）")
    print("-" * 78)
    stats = {}
    for c in CONCURRENCIES:
        s = batch_stats(c)
        if s is None:
            print(f"  c={c:<3} vLLM 日志里没抽到 Running 行 → 无法判断自变量有没有操纵成")
            return 1
        stats[c] = s
        print(f"  c={c:<3} 均值 {s['mean']:5.1f}  p50 {s['p50']:3d}  p90 {s['p90']:3d}  "
              f"max {s['max']:3d}  （{s['n']} 个采样点）")
    ratio = stats[32]["mean"] / stats[8]["mean"]
    print(f"\n  c=32 / c=8 的实际批大小比 = **{ratio:.2f}×**（名义比 4×）")
    if ratio < 2.0:
        print("  → **没能操纵自变量**（比值 < 2×）。环境侧 8 个 GIL 绑死的 worker 把在飞批")
        print("    大小卡住了，concurrency 名义上翻 4 倍但 LLM 那边没跟着翻。")
        print("    **这不是「假设被否」**，下面的成功率差不能当阴性结果用。")
        print("    补救：每段 env worker 8 → 32 再来，或直接控 vLLM 的 --max-num-seqs。")
        manipulated = False
    else:
        print(f"  → 自变量操纵**成立**（≥ 2×），下面的成功率比较可以解读。")
        manipulated = True

    # ---- 第二步：成功率 ----
    print()
    print("-" * 78)
    print("第二步 · 成功率（同一份 sft_v2 权重、同窗口、同池、只差 concurrency）")
    print("-" * 78)
    rates = {}
    for c in CONCURRENCIES:
        rates[c] = rate(leg(c))
        print(f"  c={c:<3} 成功率 {rates[c]:.4f}")
    span = max(rates.values()) - min(rates.values())
    print(f"  极差 = **{span * 100:.2f} pp**")

    print()
    print("  两两配对（bootstrap 95%，按 task_id 配对；三对不独立，逐对照登不校正）：")
    any_sig = False
    for a, b in itertools.combinations(CONCURRENCIES, 2):
        d = paired(leg(b), leg(a), f"c{b}_vs_c{a}")
        if d is None:
            print(f"    c={b} vs c={a}：算不出")
            continue
        sig = not (d["ci_low"] < 0 < d["ci_high"])
        any_sig = any_sig or sig
        print(f"    c={b} vs c={a}：{d['delta'] * 100:+.2f} pp "
              f"[{d['ci_low'] * 100:+.2f}, {d['ci_high'] * 100:+.2f}]  "
              f"{'**不跨 0**' if sig else '跨 0'}  配对题 {d['paired_tasks']}")

    # ---- 判读 ----
    print()
    print("-" * 78)
    print("判读（按预注册的分支）")
    print("-" * 78)
    if not manipulated:
        print("  自变量没操纵成 → 判定停在**「未能操纵自变量」**。不写成假设被否。")
        return 0
    if any_sig:
        print("  → 有配对区间不跨 0：**concurrency 可测地改变成功率**。")
        print("    结论：评测口径必须钉住 concurrency（现在钉的是 16，所以已发布结果内部")
        print("    一致），且**批构成是跨 session 漂移的一个活机制**。roadmap 阶段 C 里那条")
        print("    机制假设要标成「已检验、成立」。")
    elif span <= 0.010:
        print("  → 三对全跨 0 且极差 ≤ 1 pp：**批构成解释不了那 2 pp 的跨天漂移**。")
        print("    roadmap 阶段 C 那条机制假设要标成**「已检验、不成立」**。剩下的候选：")
        print("    vLLM/驱动的非确定性归约顺序、环境侧状态、以及「根本没有机制，只是配对")
        print("    区间本身比它声称的窄」——最后这个和 #28 的六个 δ 散布对得上就该优先。")
    else:
        print(f"  → 区间跨 0 但极差 {span * 100:.2f} pp > 1 pp：**有信号、测不动**。")
        print(f"    把极差 {span * 100:.2f} pp 记成漂移量级的又一次测量。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
