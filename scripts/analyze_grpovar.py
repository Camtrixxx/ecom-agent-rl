#!/usr/bin/env python3
"""#29 的判定：把 docs/grpovar-preregistration.md 的规则原样执行。

**写于 2026-08-16 06:55，四条腿一条都没起、一个新数都没有。** 已见的只有已发布的三个
配对 delta（它们是比较对象）。分析代码在数据之前写，等于不给自己留挑口径的余地。

    .venv/bin/python scripts/analyze_grpovar.py

## 主统计量是散布，不是均值

预注册写的是 s′ = stdev(三个同窗口重评的配对 delta)，和已发布的 s = 1.37 pp 比。均值照登，
但它不是判定对象——本臂问的是"那 1.37 里有多少是评测漂移"，不是"增益还在不在"。

## 四个写死的地方

- **n=3、三个 seed 是常量**，没有 --seeds 参数。
- **仍配旧基线 outputs/rollouts/sft.jsonl**，且启动时校 sha256。要和 1.37 pp 比就得用产生
  它的那个口径；减数是常数、在标准差里约掉，所以旧基线漂没漂移不影响主判定。
- **自检**：先用已发布的三份报告复算 d̄/s，复现不了 +8.17/1.37 就退出。
- **分支阈值 0.70 / 1.10 pp 在预注册里定死**，依据是 #28 实测的收缩倍率 0.38×/0.53×。
  这里不重新论证，只执行。

## 这条臂几乎没有检验力，脚本要自己说出来

F(2,2) 的单侧 0.95 临界值是 19.0 → 要"显著变小"得 s′ < 0.31 pp，比窗口内噪声底 0.39 pp
还小。所以 F 照算照登，但**分支是点估计分支**，打印时必须带"n=3、方向性、不显著"。
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "outputs" / "rollouts"
OUTDIR = ROLLOUTS / "grpovar"
POOL = ROOT / "data" / "task_pools" / "evaluation.jsonl"
PY = ROOT / ".venv" / "bin" / "python"

TARGET_ROWS = 2000

# R1 用的那份 SFT 基线轨迹。指纹在预注册里抄下来了，对不上就拒跑。
BASELINE = ROLLOUTS / "sft.jsonl"
BASELINE_SHA = "57a85d0d75bb6bc6ca2145b2c766ccb758f2e60c8749d5b8be735d3a5e766da7"

# (seed, 新腿名, 权重, 已发布的配对报告, 预注册抄下来的已发布 delta 与绝对成功率, 已发布评测时刻)
RUNS = [
    (42, "gv_grpo_s42", "grpo",     "grpo_vs_sft",     0.0741, 0.6838, "08-12 12:39"),
    (43, "gv_grpo_s43", "grpo_s43", "grpo_s43_vs_sft", 0.0975, 0.7085, "08-13 20:57"),
    (44, "gv_grpo_s44", "grpo_s44", "grpo_s44_vs_sft", 0.0736, 0.6833, "08-13 21:35"),
]
SFT_LEG = "gv_sft"           # 只服务附带产出 2，缺了不挡主判定
PUB_S = 0.0137               # R1 的 s = 1.37 pp
PUB_MEAN = 0.0817            # R1 的 d̄ = +8.17 pp
NOISE_FLOOR = 0.0039         # #14 的窗口内锚：同一份权重重评三次的 s = 0.39 pp
F_CRIT_2_2 = 19.0            # F₀.₉₅(2,2)，单侧


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(name: str) -> int:
    f = ROLLOUTS / f"{name}.jsonl"
    if not f.exists():
        return 0
    with f.open() as fh:
        return sum(1 for _ in fh)


def strict_ok(name: str) -> bool:
    """口径闸门：轨迹里不能有 tolerant_parse 键（它只在非默认口径下才写，自证）。

    不看 grep 的退出码——无匹配时 grep 退 1，`|| echo 0` 会再追一个 0 进 stdout。
    """
    f = ROLLOUTS / f"{name}.jsonl"
    if not f.exists():
        return False
    r = subprocess.run(["grep", "-c", '"tolerant_parse"', str(f)],
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0) == 0


def rate(name: str) -> float | None:
    p = ROLLOUTS / f"{name}.report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["overall"]["success_rate"]["mean"]


def paired(treat: str, base: Path, tag: str) -> dict | None:
    """按 task_id 配对，report_metrics.py 的 bootstrap 区间。和 R1 逐字同一口径。"""
    t = ROLLOUTS / f"{treat}.jsonl"
    if not (t.exists() and base.exists()):
        return None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{tag}.report.json"
    if not out.exists():
        r = subprocess.run(
            [str(PY), "scripts/report_metrics.py", "--trajectories", str(t),
             "--baseline", str(base), "--pool", str(POOL), "--json", str(out)],
            cwd=ROOT, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  !! report_metrics 失败 {tag} exit={r.returncode}\n{r.stderr[-500:]}")
            return None
    return json.loads(out.read_text())


def _selfcheck() -> bool:
    """用已发布的三个 delta 复算 d̄/s，确认本脚本的统计和 R1 一致。"""
    pub = []
    for seed, _, _, report, pub_delta, _, _ in RUNS:
        p = ROLLOUTS / f"{report}.report.json"
        if not p.exists():
            print(f"!! 已发布的 {report}.report.json 不见了，停")
            return False
        d = json.loads(p.read_text())["paired_vs_baseline"]
        if abs(d["delta"] - pub_delta) > 0.0002:
            print(f"!! seed {seed} 的已发布 delta 变了：预注册抄的是 {pub_delta:.4f}，"
                  f"文件里现在是 {d['delta']:.4f}。有人重跑覆盖了它，判定停。")
            return False
        if d["baseline_file"] != "outputs/rollouts/sft.jsonl":
            print(f"!! seed {seed} 配的基线是 {d['baseline_file']}，不是 R1 那份，停")
            return False
        pub.append(d["delta"])
    m, s = statistics.mean(pub), statistics.stdev(pub)
    print(f"自检：用已发布的三个 delta 复算 → d̄ = {m * 100:+.2f} pp，s = {s * 100:.2f} pp")
    print(f"      R1 发布的是 d̄ = +8.17 pp，s = 1.37 pp", end="  ")
    if abs(m - PUB_MEAN) < 0.0002 and abs(s - PUB_S) < 0.0002:
        print("→ 对齐 ✅")
        return True
    print("→ **对不上**，统计口径和 R1 不一致，判定不可信")
    return False


def _baseline_gate() -> bool:
    if not BASELINE.exists():
        print(f"!! 基线轨迹不见了：{BASELINE}")
        return False
    got = sha256(BASELINE)
    if got != BASELINE_SHA:
        print(f"!! 基线轨迹的 sha256 变了：\n   预注册 {BASELINE_SHA}\n   现在   {got}")
        print("   R1 的 1.37 pp 是在旧文件上算的，两边不可比 → 判定停。")
        return False
    print(f"基线闸门：outputs/rollouts/sft.jsonl 指纹未变（{got[:16]}…）✅")
    return True


def main() -> int:
    print("=" * 78)
    print("#29 三份 GRPO 快照同窗口重评 · 判定")
    print("=" * 78)
    if not _baseline_gate():
        return 1
    if not _selfcheck():
        return 1
    print()

    # ---- 闸门：三条 GRPO 腿齐不齐、口径严不严 ----
    bad = []
    for seed, leg, _, _, _, _, _ in RUNS:
        n, strict = rows(leg), strict_ok(leg)
        # 「文件还没有」和「口径不对」是两件事，混成一句会让人去查一个不存在的问题。
        flag = "还没出" if n == 0 else ("严格 ✅" if strict else "!! 有 tolerant_parse")
        print(f"  {leg:<14} {n}/{TARGET_ROWS}  {flag}")
        if n < TARGET_ROWS or not strict:
            bad.append(leg)
    if bad:
        print(f"\n{bad} 没跑齐或口径不对 → **不出判定**。预注册写死 n=3，"
              f"缺腿不降级成 n=2。")
        return 1

    # ---- 主判定 ----
    print()
    print("-" * 78)
    print("主判定 · 同窗口三个配对 delta 的散布（对同一份未重采的 sft.jsonl）")
    print("-" * 78)
    new = []
    for seed, leg, _, report, pub_delta, pub_rate, pub_when in RUNS:
        rep = paired(leg, BASELINE, f"{leg}_vs_sft")
        if rep is None:
            print(f"  seed {seed}：配对算不出 → 不出判定")
            return 1
        d = rep["paired_vs_baseline"]
        new.append({"seed": seed, "leg": leg, "delta": d["delta"],
                    "ci": (d["ci_low"], d["ci_high"]), "n": d["paired_tasks"],
                    "rate": rep["overall"]["success_rate"]["mean"],
                    "pub_delta": pub_delta, "pub_rate": pub_rate, "pub_when": pub_when})

    print(f"  {'seed':>5} {'已发布Δ':>9} {'时刻':>12} {'同窗口Δ':>9} "
          f"{'采样区间(pp)':>20} {'配对题':>7}")
    for r in new:
        lo, hi = r["ci"]
        print(f"  {r['seed']:>5} {r['pub_delta'] * 100:>+8.2f} {r['pub_when']:>12} "
              f"{r['delta'] * 100:>+8.2f} [{lo * 100:>+7.2f},{hi * 100:>+7.2f}] "
              f"{r['n']:>7}")

    deltas = [r["delta"] for r in new]
    s_new = statistics.stdev(deltas)
    m_new = statistics.mean(deltas)
    print()
    print(f"  同窗口 d̄  = {m_new * 100:+.2f} pp   （对照 R1 发布的 +8.17 pp）")
    print(f"  同窗口 s′ = **{s_new * 100:.2f} pp**   （对照 R1 发布的 1.37 pp，"
          f"收缩 {s_new / PUB_S:.2f}×）")

    # 检验力：先把话说在前面
    f_ratio = (PUB_S ** 2) / (s_new ** 2) if s_new > 0 else float("inf")
    sig = f_ratio > F_CRIT_2_2
    print(f"  方差比 F = 1.37²/{s_new * 100:.2f}² = {f_ratio:.2f}（df=2,2），"
          f"F₀.₉₅ = {F_CRIT_2_2} → {'显著' if sig else '**不显著**'}")
    print(f"  （预注册已交代：要显著需 s′ < 0.31 pp，比窗口内噪声底 0.39 pp 还小，"
          f"本来就够不着。）")

    print()
    if s_new <= 0.0070:
        print("  → s′ ≤ 0.70 pp，与 #28 的收缩倍率一致：**σ = 1.37 pp 高估了纯训练方差**。")
        print(f"    roadmap 里那句悬着的猜测改成「已检验，方向性成立」；实用判据从")
        print(f"    「约 1.4 pp」改成「约 {s_new * 100:.2f} pp」，**并注明 n=3、方向性、不显著**。")
        print("    所有引用 1.4 pp 那把尺子的地方都要标注新值。")
    elif s_new >= 0.0110:
        print("  → s′ ≥ 1.10 pp，收缩没有发生：**「σ 多半高估纯训练方差」不成立**，")
        print("    在 roadmap 里标成已检验、不成立。1.37 pp 基本就是训练方差，")
        print("    「小于约 1.4 pp 分不开」这句判据原样保留。")
    else:
        print(f"  → 0.70 < s′ = {s_new * 100:.2f} < 1.10 pp：**不定**。两把尺子并列照登，")
        print("    roadmap 措辞一个字不改，只补一行「同窗口重评得这个值，n=3，分不出」。")

    _ancillary(new, s_new, m_new)
    return 0


def _ancillary(new: list[dict], s_new: float, m_new: float) -> None:
    # ---- 附带 1：纯训练标准差 ----
    print()
    print("-" * 78)
    print("附带 1 · 扣掉窗口内评测噪声之后的纯训练标准差")
    print("-" * 78)
    var = s_new ** 2 - NOISE_FLOOR ** 2
    print(f"  σ_train ≈ sqrt(s′² − 0.39²) = sqrt({s_new * 100:.2f}² − 0.39²)")
    if var <= 0:
        print("  → 根号内为负：**低于噪声底，测不出**。")
        print("    这**不是**「训练方差是零」——是这把尺子分不开它和评测噪声。")
    else:
        print(f"  → **{math.sqrt(var) * 100:.2f} pp**（点估计，n=3，别当区间用）")


    # ---- 附带 2：第一次完全同窗口的 GRPO 增益 ----
    print()
    print("-" * 78)
    print("附带 2 · 第一次完全同窗口的 GRPO 增益（减数也在本窗口内）")
    print("-" * 78)
    sft_rate = rate(SFT_LEG)
    if sft_rate is None or rows(SFT_LEG) < TARGET_ROWS or not strict_ok(SFT_LEG):
        print(f"  {SFT_LEG} 没跑齐或口径不对 → 这一项跳过（预注册：它不挡主判定）")
    else:
        grpo_rates = [r["rate"] for r in new]
        gain = statistics.mean(grpo_rates) - sft_rate
        print(f"  三条 GRPO 新腿的绝对成功率 {[f'{x:.4f}' for x in grpo_rates]}  "
              f"均值 {statistics.mean(grpo_rates):.4f}")
        print(f"  同窗口 SFT（{SFT_LEG}）= {sft_rate:.4f}  "
              f"（已发布的 sft = 0.6061，08-11 评）")
        print(f"  **同窗口增益 = {gain * 100:+.2f} pp**")
        print("  逐 seed 的同窗口配对（bootstrap 95%）：")
        for r in new:
            rep = paired(r["leg"], ROLLOUTS / f"{SFT_LEG}.jsonl", f"{r['leg']}_vs_{SFT_LEG}")
            if rep is None:
                print(f"    seed {r['seed']}：算不出")
                continue
            d = rep["paired_vs_baseline"]
            print(f"    seed {r['seed']}：{d['delta'] * 100:+.2f} pp "
                  f"[{d['ci_low'] * 100:+.2f}, {d['ci_high'] * 100:+.2f}]  "
                  f"配对题 {d['paired_tasks']}")
        print()
        print("  **引用时必须带的限定**：SFT 侧只有 n=1，所以这个数不含 SFT 侧的训练 run")
        print("  方差（R1 也一样，它的基线同样是单份未重采）。它答的是「同窗口下增益还有")
        print("  多大」，不是「增益的总不确定度」。")

    # ---- 附带 3：三个 δ = 又三个漂移点 ----
    print()
    print("-" * 78)
    print("附带 3 · 三份 GRPO 权重的漂移（间隔 3–4 天，落在「≥40 h」那一档）")
    print("-" * 78)
    devs = []
    for r in new:
        d = r["rate"] - r["pub_rate"]
        devs.append(d)
        print(f"  seed {r['seed']}：已发布 {r['pub_rate']:.4f}（{r['pub_when']}） → "
              f"新值 {r['rate']:.4f}  δ = {d * 100:+.2f} pp")
    s_dev = statistics.stdev(devs)
    print(f"  三个 δ 的均值 = {statistics.mean(devs) * 100:+.2f} pp（整窗口公共位移）")
    print(f"  三个 δ 的 s   = **{s_dev * 100:.2f} pp**（权重相关分量）"
          f"  ← 对照 #28 的 0.62 pp")
    print()
    print("  按 roadmap 阶段 C 的经验规律，≥40 h 间隔预期 1.6–2.3 pp 且**符号随机**，")
    print("  所以 δ 本身的大小说明不了什么；有信息的是它们的**散布**。")


if __name__ == "__main__":
    sys.exit(main())
