#!/usr/bin/env python3
"""R1 的判定：把 multiseed-preregistration.md 的规则原样执行。

**写在 seed 43 / 44 的评测数字出现之前**（2026-08-13 14:30，seed 44 还在训第 1 轮）。
这是预注册的配套：规则已经写死在 docs/multiseed-preregistration.md，这里只是让它变成
一条不能事后调整的命令。分析代码在看到数据之后才写，等于给自己留了挑口径的余地。

三条刻意写死、不留开关的地方：

1. **t₂,₀.₉₇₅ = 4.30 和 n = 3 是常量。** 没有 `--seeds` 参数。预注册规则 2 明写
   「不补第 4 个 seed」，脚本能接第 4 个 seed 就等于把那条规则变成可选项。
2. **区间跨 0 时打印的是「降级」而不是「未达显著」。** 后者听起来像还差一点，
   而预注册说那本身就是结论。
3. **d₄₂ 从盘上的报告读，不写死 7.41。** 若那份报告被重新生成过，这里要跟着变，
   而不是继续引用一个记在文档里的数。

    .venv/bin/python scripts/analyze_multiseed.py
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "outputs" / "rollouts"

T_CRIT = 4.30  # t 分布 df=2 双侧 0.975。n=3 是预注册定死的，不接受参数。
N_SEEDS = 3

# (seed, 配对报告, 轨迹文件, 训练产物目录)。42 是已发布的那个 run。
RUNS = [
    (42, "grpo_vs_sft.report.json", "grpo.jsonl", "grpo"),
    (43, "grpo_s43_vs_sft.report.json", "grpo_s43.jsonl", "grpo_s43"),
    (44, "grpo_s44_vs_sft.report.json", "grpo_s44.jsonl", "grpo_s44"),
]


def load_delta(report: Path) -> dict:
    d = json.loads(report.read_text())["paired_vs_baseline"]
    # report_metrics.py 存的是分数，这里一次性转成 pp，之后全用 pp。
    return {
        "delta_pp": d["delta"] * 100,
        "ci_pp": (d["ci_low"] * 100, d["ci_high"] * 100),
        "paired_tasks": d["paired_tasks"],
        "baseline_file": d["baseline_file"],
    }


def run_facts(out_dir: Path) -> dict:
    """实际轮数与 seed。预注册：没跑满的按实际轮数如实报，不静默替换。"""
    # 键名不能叫 seed：调用方 row.update(run_facts(...))，会把 RUNS 里的 seed 覆盖掉。
    # 这里叫 meta_seed，正好还能拿来对一次「产物里记的 seed 和我们以为的是不是同一个」。
    facts = {"iterations": None, "meta_seed": None, "world_size": None}
    meta = out_dir / "metadata.json"
    if meta.exists():
        h = json.loads(meta.read_text()).get("hyperparams", {})
        facts["meta_seed"] = h.get("seed")
        facts["world_size"] = h.get("world_size")
    log = out_dir / "train_log.jsonl"
    if log.exists():
        facts["iterations"] = sum(1 for _ in log.open())
    return facts


def early_abstain_count(traj: Path) -> int | None:
    """预注册的次要预测 3：三个 run 都该保持 0。"""
    if not traj.exists():
        return None
    n = 0
    for line in traj.open():
        rec = json.loads(line)
        if rec.get("reward_type") == "early_abstain":
            n += 1
    return n


def main() -> int:
    rows, missing = [], []
    for seed, report, traj, out_dir in RUNS:
        rp = ROLLOUTS / report
        if not rp.exists():
            missing.append(f"seed {seed}: 缺 {rp.relative_to(ROOT)}")
            continue
        row = {"seed": seed, **load_delta(rp)}
        row.update(run_facts(ROOT / "outputs" / "models" / out_dir))
        row["early_abstain"] = early_abstain_count(ROLLOUTS / traj)
        rows.append(row)

    if missing:
        print("!! 还不能判定，缺以下产物：")
        for m in missing:
            print("  ", m)
        print("\n预注册不允许拿 2 个 seed 先算一版——那等于偷看。等齐再跑。")
        return 1

    print("=== 三个 run 的配对 delta（各自对同一份 SFT 轨迹）===")
    print(f"{'seed':>5} {'delta(pp)':>10} {'采样区间(pp)':>20} {'配对题':>7} "
          f"{'轮数':>5} {'卡':>3} {'early_abstain':>13}")
    for r in rows:
        lo, hi = r["ci_pp"]
        print(f"{r['seed']:>5} {r['delta_pp']:>+10.2f} "
              f"[{lo:>+7.2f},{hi:>+7.2f}] {r['paired_tasks']:>7} "
              f"{str(r['iterations']):>5} {str(r['world_size']):>3} "
              f"{str(r['early_abstain']):>13}")

    # 口径一致性：三个 delta 必须配同一份基线，否则 t 区间量的不是同一件事。
    baselines = {r["baseline_file"] for r in rows}
    if len(baselines) != 1:
        print("\n!! 三个 delta 配的不是同一份基线轨迹，区间无意义：")
        for b in sorted(baselines):
            print("   ", b)
        return 1
    print(f"\n基线（三者共用，未重采）：{sorted(baselines)[0]}")

    # 产物里记的 seed / 卡数必须和我们以为的一致。卡数尤其要查：6 卡 global batch 是 12
    # 批、4 卡是 8 批，若某个 run 的卡数不同，卡数就和 seed 混在一起，恰好毁掉要量的东西。
    for r in rows:
        if r["meta_seed"] is not None and r["meta_seed"] != r["seed"]:
            print(f"!! seed {r['seed']} 的 metadata 里记的是 {r['meta_seed']}，产物对不上")
            return 1
    ws = {r["world_size"] for r in rows if r["world_size"] is not None}
    if len(ws) > 1:
        print(f"!! 三个 run 的训练卡数不一致 {sorted(ws)}——global batch 随卡数变，")
        print("   卡数就和 seed 混在一起了。这个 t 区间量的不是 run 间方差。")
        return 1

    # 「读不到轮数」和「没跑满」是两件事，不能混成一句话报出去——前者是我们不知道，
    # 后者是一个关于 run 的事实声明。
    unknown = [r for r in rows if r["iterations"] is None]
    short = [r for r in rows if r["iterations"] is not None and r["iterations"] < 40]
    if unknown:
        print("!! 以下 run 读不到 train_log.jsonl，轮数未知（不等于没跑满）：")
        for r in unknown:
            print(f"   seed {r['seed']}")
    if short:
        print("!! 以下 run 未跑满 40 轮，按实际轮数如实报（预注册：不许丢 seed）：")
        for r in short:
            print(f"   seed {r['seed']}: {r['iterations']}/40")

    deltas = [r["delta_pp"] for r in rows]
    if len(deltas) != N_SEEDS:
        print(f"\n!! 期望 {N_SEEDS} 个 delta，实得 {len(deltas)}")
        return 1

    mean = statistics.mean(deltas)
    s = statistics.stdev(deltas)          # 样本标准差，n-1
    half = T_CRIT * s / (N_SEEDS ** 0.5)
    lo, hi = mean - half, mean + half
    crosses_zero = lo <= 0 <= hi

    print("\n=== 主统计量：三个 delta 的均值，t 区间（df=2, t=4.30）===")
    print(f"  d̄ = {mean:+.2f} pp")
    print(f"  s  = {s:.2f} pp   （run 间散布，这才是已发布区间里缺的那一项）")
    print(f"  半宽 = 4.30 × {s:.2f} / √3 = ±{half:.2f} pp")
    print(f"  95% t 区间 = [{lo:+.2f}, {hi:+.2f}] pp")

    print("\n=== 判定（规则写在 docs/multiseed-preregistration.md，跑前定死）===")
    if not crosses_zero:
        print("  区间不跨 0 → **结论升级**：GRPO 的提升在 run 间稳定，")
        print(f"  均值 {mean:+.2f} pp，区间 [{lo:+.2f}, {hi:+.2f}] pp。停在 3 个 seed。")
        print(f"  已发布的 {rows[0]['delta_pp']:+.2f} pp 保留，并注明它是三个 run 之一。")
    else:
        print("  区间跨 0 → **结论降级**：在 3 个 run 的分辨率下，无法确认 GRPO 的提升")
        print("  在 run 间稳定。**不补第 4 个 seed**——预注册规则 2，这是最重要的一条：")
        print("  「不显著就加 seed 直到显著」会把假阳性率从 5% 抬到 20% 以上。")
        print(f"  三个 delta 原始值：{', '.join(f'{d:+.2f}' for d in deltas)}  s = {s:.2f} pp")

    print("\n=== 预注册的次要预测（写在看到数字之前）===")
    ok = "1 ≤ s ≤ 3" if 1.0 <= s <= 3.0 else f"**未中**，s = {s:.2f}"
    print(f"  1. s 落在 1–3 pp：{ok}")
    if s < 0.5:
        print("     s < 0.5 pp，预注册要求单独核对是不是三个 run 塌到了同一个解。")
    pos = all(d > 0 for d in deltas)
    print(f"  2. 三个 delta 全为正：{'中' if pos else '**未中**'}"
          f"{'' if pos else '——「GRPO 有效」这个结论本身要重写，无论均值多少'}")
    ea = {r["seed"]: r["early_abstain"] for r in rows}
    all_zero = all(v == 0 for v in ea.values())
    print(f"  3. early_abstain 全为 0：{'中' if all_zero else f'**未中** {ea}'}")

    print("\n注：三个 delta 共用同一份 SFT 基线，因此并不完全独立；上面的 t 区间描述的是")
    print("「GRPO 这一侧的 run 间散布」。这一点预注册里就写明了，不是事后补的限制。")
    print("另注：seed 42 与 43/44 的代码有一处差异（472e415 压缩层守卫，seed 42 跑到")
    print("一半时提交，未进它的进程），seed 42 因此在第 20 轮中止重采过一批。见执行日志。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
