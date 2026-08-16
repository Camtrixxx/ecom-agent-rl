#!/usr/bin/env python3
"""D2 判定：SFT-v2 的数据扩容值不值得换基线。

**写在 SFT-v2 训完之前，任何 v2 的数字都还不存在。** 判定规则来自
docs/d2-preregistration.md，这里只是把它变成不能反悔的代码。规则先写死，是因为
预期效应（0.8–1.6 pp）比单 run 对比的噪声带（±3.8 pp）小一个量级——跑完之后再决定
「这个数算不算显著」，等于让噪声挑门槛。

两个判定，分辨率天差地别：

**次判定（3 对 3，唯一能下结论的那个）**：v1 三个 seed vs v2 三个 seed 的成功率，
合并方差的双样本 t，df = 4、t₄,₀.₉₇₅ = 2.78。区间不跨 0 → 可以据此换基线；跨 0 →
降级报结论，**不补第 4 个 seed**（不显著就加 seed 直到显著，会把假阳性率从 5% 推到
20% 以上）。

**主判定（端到端单 run）**：GRPO-v2 vs GRPO-v1 seed 42 的配对差值。|delta| ≤ 3.8 pp
一律判「无法与 run 运气区分」——既不是「有效应」也不是「没效应」。这个数是给人看的，
不承担决策。

会拒绝运行（宁可不出数，也不出一个口径不对的数）：
  - 缺任何一份报告
  - 六份报告不是同一个评测池 / 同一个 k
  - SFT 权重的 metadata 里 seed 或 world_size 与预期不符（卡数一错就混进 global batch）
  - 两组的训练数据文件不是各自预期的那一份

    .venv/bin/python scripts/analyze_d2.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 预注册定死：每组 3 个 seed，df = 3 + 3 − 2 = 4。
T_CRIT_DF4 = 2.78          # t₄,₀.₉₇₅，次判定用
SINGLE_RUN_BAND_PP = 3.8   # 主判定的门槛：σ√2 × 1.96，σ = 1.37 pp 来自 R1
EXPECTED_WORLD_SIZE = 6
EXPECTED_TASKS = 500
EXPECTED_K = 4.0

# (报告名, 权重目录, 期望 seed, 期望训练集)
V1_RUNS = [
    ("sft",           "outputs/models/sft",        42, "data/sft/train.jsonl"),
    ("sft_s43",       "outputs/models/sft_s43",    43, "data/sft/train.jsonl"),
    ("sft_s44",       "outputs/models/sft_s44",    44, "data/sft/train.jsonl"),
]
V2_RUNS = [
    ("sft_v2",        "outputs/models/sft_v2",     42, "data/sft_v2/train.jsonl"),
    ("sft_v2_s43",    "outputs/models/sft_v2_s43", 43, "data/sft_v2/train.jsonl"),
    ("sft_v2_s44",    "outputs/models/sft_v2_s44", 44, "data/sft_v2/train.jsonl"),
]

problems: list[str] = []


def die(msg: str) -> None:
    print(f"!! {msg}")
    problems.append(msg)


def load_report(name: str) -> dict | None:
    path = ROOT / "outputs" / "rollouts" / f"{name}.report.json"
    if not path.exists():
        die(f"缺报告 {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text())


def check_report(name: str, rep: dict) -> float | None:
    """核对口径，返回成功率。口径不对就返回 None（并记问题）。"""
    o = rep["overall"]
    if o["tasks"] != EXPECTED_TASKS:
        die(f"{name}: 题数 {o['tasks']} ≠ {EXPECTED_TASKS}，与其他 run 不同池")
        return None
    if abs(o["attempts_per_task"] - EXPECTED_K) > 1e-6:
        die(f"{name}: k={o['attempts_per_task']} ≠ {EXPECTED_K}，采样量不同不能并列比")
        return None
    return float(o["success_rate"]["mean"])


def check_training(name: str, model_dir: str, seed: int, train_file: str) -> None:
    path = ROOT / model_dir / "metadata.json"
    if not path.exists():
        die(f"{name}: 缺训练血缘 {model_dir}/metadata.json")
        return
    m = json.loads(path.read_text())
    hp = m.get("hyperparams", {})
    if hp.get("seed") != seed:
        die(f"{name}: metadata seed={hp.get('seed')} ≠ 预期 {seed}")
    if hp.get("world_size") != EXPECTED_WORLD_SIZE:
        die(f"{name}: world_size={hp.get('world_size')} ≠ {EXPECTED_WORLD_SIZE}"
            "（卡数变了 global batch 就变了，数据量和卡数会缠在一起）")
    got = m.get("provenance", {}).get("train")
    if got != train_file:
        die(f"{name}: 训练集是 {got}，预期 {train_file}")


def check_recipe_identical(rows: list[tuple[str, str]]) -> None:
    """六个 run 的配方除了 seed 之外必须逐项一致，否则测的不只是数据。"""
    watched = ("epochs", "lr", "grad_accum", "tokens_per_batch", "max_length",
               "warmup_ratio", "weight_decay", "max_grad_norm", "attn", "model")
    seen: dict[str, dict] = {}
    for name, model_dir in rows:
        path = ROOT / model_dir / "metadata.json"
        if not path.exists():
            continue
        hp = json.loads(path.read_text()).get("hyperparams", {})
        seen[name] = {k: hp.get(k) for k in watched}
    if len(seen) < 2:
        return
    ref_name, ref = next(iter(seen.items()))
    for name, hp in seen.items():
        diff = {k: (ref[k], hp[k]) for k in watched if ref[k] != hp[k]}
        if diff:
            die(f"{name} 与 {ref_name} 的配方不一致：{diff}")


# 分层指标：难度段 → 指标名 → 每个 run 一个数。
# 只收这两个：8+ 段的错买率（预测 2）和成功率（roadmap 里那个瓶颈）。
STRAT_BANDS = ("1-2", "3-4", "5-7", "8+")
STRAT_METRICS = ("wrong_purchase_rate", "success_rate")


def collect_stratified(group: str, name: str, rep: dict,
                       strat: dict[str, dict[str, list[float]]]) -> None:
    """把每个 run 的分层数按 段/指标 归到组里。缺段就记问题，不静默跳过。"""
    diff = rep.get("stratified", {}).get("difficulty", {})
    for band in STRAT_BANDS:
        s = diff.get(band)
        if s is None:
            die(f"{name}: stratified 里缺难度段 {band}")
            continue
        for metric in STRAT_METRICS:
            m = s.get(metric, {}).get("mean")
            if m is None:
                die(f"{name}: {band} 段缺 {metric}")
                continue
            strat[group].setdefault(f"{band}/{metric}", []).append(float(m))


def t_two_sample(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """合并方差双样本 t：返回 (均值差, 合并 s, 半宽, t 统计量)。b − a。"""
    na, nb = len(a), len(b)
    ma, mb = sum(a) / na, sum(b) / nb
    va = sum((x - ma) ** 2 for x in a) / (na - 1)
    vb = sum((x - mb) ** 2 for x in b) / (nb - 1)
    sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    se = math.sqrt(sp2 * (1 / na + 1 / nb))
    diff = mb - ma
    half = T_CRIT_DF4 * se
    return diff, math.sqrt(sp2), half, (diff / se if se else float("inf"))


def sd(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main() -> int:
    out: list[str] = []
    def say(line: str = "") -> None:
        print(line)
        out.append(line)

    say("=" * 72)
    say("D2 判定 —— 规则见 docs/d2-preregistration.md（先写后跑）")
    say("=" * 72)

    # ---- 次判定：3 对 3 -----------------------------------------------------
    rates: dict[str, list[float]] = {"v1": [], "v2": []}
    names: dict[str, list[str]] = {"v1": [], "v2": []}
    strat: dict[str, dict[str, list[float]]] = {"v1": {}, "v2": {}}
    for group, runs in (("v1", V1_RUNS), ("v2", V2_RUNS)):
        for name, model_dir, seed, train_file in runs:
            rep = load_report(name)
            if rep is None:
                continue
            check_training(name, model_dir, seed, train_file)
            rate = check_report(name, rep)
            if rate is not None:
                rates[group].append(rate)
                names[group].append(name)
                collect_stratified(group, name, rep, strat)
    check_recipe_identical([(n, d) for n, d, _, _ in V1_RUNS + V2_RUNS])

    say()
    say("### 次判定：SFT 3 对 3（唯一有分辨率的那个）")
    say()
    if len(rates["v1"]) < 2 or len(rates["v2"]) < 2:
        say(f"!! 可用 run 不足（v1 {len(rates['v1'])} 个、v2 {len(rates['v2'])} 个），无法出区间")
    else:
        for group in ("v1", "v2"):
            for name, r in zip(names[group], rates[group]):
                say(f"  {group}  {name:<12} 成功率 {r:.4f}  ({r * 100:.2f} pp)")
            m = sum(rates[group]) / len(rates[group])
            s = sd(rates[group]) if len(rates[group]) > 1 else float("nan")
            say(f"  {group}  均值 {m * 100:.2f} pp   s = {s * 100:.2f} pp   n = {len(rates[group])}")
            say()
        diff, sp, half, tstat = t_two_sample(rates["v1"], rates["v2"])
        lo, hi = (diff - half) * 100, (diff + half) * 100
        n1, n2 = len(rates["v1"]), len(rates["v2"])
        df = n1 + n2 - 2
        say(f"  v2 − v1 = {diff * 100:+.2f} pp")
        say(f"  合并 s = {sp * 100:.2f} pp   t = {tstat:+.2f}   df = {df}"
            f"（预注册按 df=4、t=2.78；实际 df={df}）")
        say(f"  95% 区间 = [{lo:+.2f}, {hi:+.2f}] pp")
        if df != 4:
            say(f"  !! df 不是 4，用的临界值 {T_CRIT_DF4} 与实际 df 不匹配——"
                "结论要按实际 df 重算，别照抄这个区间")
        crosses = lo <= 0 <= hi
        say()
        if crosses:
            say("  → 区间跨 0：**结论降级**。新数据在 SFT 侧测不出效应。")
            say("     按预注册规则 2：**不补第 4 个 seed**。")
        else:
            direction = "有效" if diff > 0 else "有害"
            say(f"  → 区间不跨 0：新数据在 SFT 侧{direction}，可据此换基线。")

    # ---- 分层：给预测 2 一个真能出结论的口径 --------------------------------
    # 补记（2026-08-14 03:15，此时 sft_v2 的评测还在跑、报告不存在）：预注册把预测 2
    # （8+ 段错买率不改善）称为「最有判别力的一条」，但我事后核 v1 报告才发现 **8+ 段单 run
    # 根本没有分辨率**——只有 46 题 / 184 条轨迹，错买率 0.0996 的采样区间是
    # [0.0272, 0.1884]，宽 16.1 pp。也就是说 v2 落在 2.7%–18.8% 之间的任何值都和 v1
    # 「一致」，而这只是采样噪声，还没算 run 间方差。照预注册那样拿单 run 去对 0.100，
    # 无论看到什么都只能得出「测不出」。
    #
    # 所以在这里加一档：用和次判定完全相同的 3 对 3 机制去打分层指标。三个 seed 把
    # run 间方差也纳进来，是比单 run 更强的口径，不是更弱的。
    #
    # **这是对分析的追加，不是对预注册的修改**（那份文件已封）。追加的时机重要：决定加
    # 这一档时，v2 的分层数一个都还不存在，所以这个选择不可能是看着数据挑的。
    say()
    say("### 分层 3 对 3（追加口径，见脚本内注释；预测 2 靠这一档打分）")
    say()
    if len(rates["v1"]) < 2 or len(rates["v2"]) < 2:
        say("!! run 不足，分层区间同样出不来")
    else:
        say(f"  {'段/指标':<26}{'v1 均值':>9}{'v2 均值':>9}{'差':>9}   95% 区间        判定")
        for band in STRAT_BANDS:
            for metric in STRAT_METRICS:
                key = f"{band}/{metric}"
                a, b = strat["v1"].get(key, []), strat["v2"].get(key, [])
                if len(a) < 2 or len(b) < 2:
                    continue
                d, _sp, half, _t = t_two_sample(a, b)
                lo, hi = (d - half) * 100, (d + half) * 100
                ma, mb = sum(a) / len(a) * 100, sum(b) / len(b) * 100
                mark = "跨 0 → 测不出" if lo <= 0 <= hi else \
                       ("v2 更高" if d > 0 else "v2 更低")
                say(f"  {key:<26}{ma:>8.2f}{mb:>9.2f}{d * 100:>+9.2f}   "
                    f"[{lo:+6.2f}, {hi:+6.2f}]  {mark}")
        say()
        wp = "8+/wrong_purchase_rate"
        a, b = strat["v1"].get(wp, []), strat["v2"].get(wp, [])
        if len(a) >= 2 and len(b) >= 2:
            d, _sp, half, _t = t_two_sample(a, b)
            lo, hi = (d - half) * 100, (d + half) * 100
            say("  预测 2（8+ 段错买率**不**改善，即区间不应显著为负）：")
            if lo <= 0 <= hi:
                say("    → 区间跨 0，**没有改善的证据**。预测 2 成立（弱形式：测不出改善）。")
            elif hi < 0:
                say("    → 区间整体为负，**8+ 段错买率显著下降**。**预测 2 被推翻**——"
                    "说明我对这批数据「均匀扩容、不针对瓶颈」的理解是错的。")
            else:
                say("    → 区间整体为正，8+ 段错买率反而显著上升。预测 2 的方向对，但幅度反常，要查。")

    # ---- 主判定：端到端单 run ----------------------------------------------
    say()
    say("### 主判定：端到端 GRPO-v2 vs GRPO-v1 seed 42（单 run，不承担决策）")
    say()
    g = load_report("grpo_v2")
    if g is None:
        say("  !! 缺 grpo_v2 报告")
    else:
        check_report("grpo_v2", g)
        p = g.get("paired_vs_baseline")
        if not p:
            say("  !! grpo_v2 报告里没有配对段——评测时没给 --baseline")
        else:
            if p.get("baseline_file") != "outputs/rollouts/grpo.jsonl":
                die(f"grpo_v2 的对照是 {p.get('baseline_file')}，"
                    "预期 outputs/rollouts/grpo.jsonl（GRPO-v1 seed 42）")
            d = p["delta"] * 100
            say(f"  配对题 {p['paired_tasks']}   "
                f"v1 {p['baseline_success_rate'] * 100:.2f} pp → "
                f"v2 {p['treatment_success_rate'] * 100:.2f} pp")
            say(f"  delta = {d:+.2f} pp   采样区间 "
                f"[{p['ci_low'] * 100:+.2f}, {p['ci_high'] * 100:+.2f}] pp")
            say(f"  预注册门槛 ±{SINGLE_RUN_BAND_PP} pp（σ√2 × 1.96，σ = 1.37 pp 来自 R1）")
            say()
            if abs(d) <= SINGLE_RUN_BAND_PP:
                say("  → **无法与 run 运气区分**。不是「没有效应」，也不是「有效应」。")
                say("     上面那个采样区间只回答「再采一次会怎样」，不含训练 run 间方差。")
            else:
                say(f"  → 超出噪声带的方向性发现（{'正' if d > 0 else '负'}向）。"
                    "仍是单 run，**不足以据此换基线**——要补 seed 才能定。")

    say()
    say("=" * 72)
    if problems:
        say(f"!! 有 {len(problems)} 处口径问题，上面的数不可直接引用：")
        for p_ in problems:
            say(f"   - {p_}")
    else:
        say("口径核对全部通过。")
    say("=" * 72)

    dest = ROOT / "outputs" / "rollouts" / "d2_verdict.txt"
    dest.write_text("\n".join(out) + "\n")
    print(f"\n存档 → {dest.relative_to(ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
