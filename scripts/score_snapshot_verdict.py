#!/usr/bin/env python3
"""按 08-15 17:45 写下的预注册，机械地给预测 4/5/6 打分。

    .venv/bin/python scripts/score_snapshot_verdict.py

纯离线：只读 outputs/rollouts/*.report.json。

## 为什么要一个脚本，而不是看着数自己判

预注册的门槛是写在 `docs/optimization-log.md` 08-15 17:45 节里的散文。散文的问题是出数
之后可以被重新解读——「0.69 ~ 0.72」看到 0.685 的时候很容易读成「差不多在带里」。把门槛
抄成代码里的常量，判定就只有过和不过两种结果，重新解读要改代码，改代码会留在 diff 里。

这不是形式主义：08-14 的预测 1 就出现过「点估计超出上沿 0.06 pp」这种正好卡在边界上的
情况，当时是靠文字说明处理的。这次先把边界写死。

## 三条预注册（逐字抄自 optimization-log 08-15 17:45）

预测 4  iter034 在 500 题评测池：终局率 ≥0.98、条件买对率 ≥0.71、成功率 0.69~0.72，
        且对 grpo_v1 的配对差为正、95% 区间不跨 0。
        **证伪条件**：成功率 ≤0.6838（打不过 v1 的已发布值）→「v2 数据有用、只是被单一
        退化盖住」这套反事实推理就是错的，不许再用它解释主判定的 −6.29 pp。
预测 5  iter034 两个留出池的终局率差 ≤1 pp → 「训练池 vs 留出池」这个划分在第二个模型上
        重复出来。差得多 → 池间差异有模型依赖，判读 A 的结论要减弱。
预测 6  v1 的 iter039 在 grpo_val 上终局率 ≥0.95 → 选快照规则对 v1 不触发回退。
        <0.95 → 「只评一个点」的省法作废，两臂都要扫完整曲线。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLL = ROOT / "outputs" / "rollouts"

# ---- 预注册的门槛。改这里就是改预注册，会留在 diff 里。----
P4_TERMINAL_MIN = 0.98
P4_COND_MIN = 0.71
P4_SUCCESS_LO, P4_SUCCESS_HI = 0.69, 0.72
P4_FALSIFY_AT = 0.6838          # grpo_v1 已发布的整体成功率
P5_POOL_GAP_MAX = 0.01
P6_TERMINAL_MIN = 0.95
# 规则的触发阈值：终局率相对本臂最好快照塌 >5 pp 才回退。
RULE_COLLAPSE_PP = 0.05


def load(path: Path) -> dict | None:
    if not path.is_file():
        print(f"!! 缺 {path.relative_to(ROOT)}", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def levels(rep: dict) -> dict[str, float]:
    """终局率 / 成功率 / 条件买对率。

    条件买对率 = 成功率 ÷ 终局率，即「走到终局的回合里买对的比例」。乘性分解的第二项。
    """
    o = rep["overall"]
    n = o["trajectories"]
    term = o["statuses"].get("done", 0) / n if n else float("nan")
    succ = (o.get("success_rate") or {}).get("mean", float("nan"))
    return {
        "n": n,
        "terminal": term,
        "success": succ,
        "conditional": succ / term if term else float("nan"),
    }


def verdict(ok: bool) -> str:
    return "通过" if ok else "**不通过**"


def score_p4(rep: dict, fails: list[str]) -> None:
    L = levels(rep)
    p = rep.get("paired_vs_baseline") or {}

    print(f"轨迹 {L['n']}    终局率 {L['terminal']:.4f}    "
          f"成功率 {L['success']:.4f}    条件买对 {L['conditional']:.4f}")
    checks = [
        (f"终局率 ≥ {P4_TERMINAL_MIN}", L["terminal"] >= P4_TERMINAL_MIN),
        (f"条件买对 ≥ {P4_COND_MIN}", L["conditional"] >= P4_COND_MIN),
        (f"成功率 ∈ [{P4_SUCCESS_LO}, {P4_SUCCESS_HI}]",
         P4_SUCCESS_LO <= L["success"] <= P4_SUCCESS_HI),
    ]
    if p:
        d, lo, hi = p.get("delta"), p.get("ci_low"), p.get("ci_high")
        print(f"配对 vs {p.get('baseline_file')}：{p.get('paired_tasks')} 题，"
              f"基线 {p.get('baseline_success_rate'):.4f} → {p.get('treatment_success_rate'):.4f}")
        print(f"  差 {d:+.4f}  95% 区间 [{lo:+.4f}, {hi:+.4f}]")
        checks.append(("配对差为正", d is not None and d > 0))
        checks.append(("配对区间不跨 0", lo is not None and hi is not None
                       and not (lo <= 0 <= hi)))
    else:
        checks.append(("配对报告存在", False))
    print()
    for label, ok in checks:
        print(f"  {label:<28}{verdict(ok)}")
        if not ok:
            fails.append(f"预测 4／{label}")

    print()
    if L["success"] <= P4_FALSIFY_AT:
        print(f"### 证伪条件触发：成功率 {L['success']:.4f} ≤ {P4_FALSIFY_AT}")
        print("「v2 数据有用、只是被一个单一退化整个盖住」这套反事实推理**是错的**。")
        print("按预注册：不许再用「终局率塌了」去解释主判定的 −6.29 pp。要接受的结论是")
        print("v2 数据在端到端上确实不如 v1。")
    else:
        print(f"证伪条件未触发（{L['success']:.4f} > {P4_FALSIFY_AT}）：反事实推理站得住，")
        print("即 −6.29 pp 里确实有很大一块是「取了塌掉的最后一轮」造成的，不是数据本身。")


def main() -> int:
    fails: list[str] = []

    # ---------------- 预测 4 ----------------
    print("=" * 74)
    print("预测 4：iter034 在 500 题评测池上")
    print("=" * 74)
    rep = load(ROLL / "grpo_v2_iter034.report.json")
    # 缺哪条就跳哪条，不整体退出：三条预测的数据来自三次互不相干的评测（500 题池 /
    # 快照扫描 / grpo_val），谁先落地不固定。原来这里 `return 1` 会让已经备齐的预测 6
    # 也打不了分，而它恰好是决定「v1 要不要扫完整条曲线」的那一条——那是 5 小时 GPU
    # 的取舍，不该等另一条腿。
    #
    # 跳过与不通过要分开记：跳过不进 `fails`（没数据不是坏结果），但退出码仍然非 0，
    # 否则「还没跑完」和「全部命中」在自动化里长得一样。
    skipped: list[str] = []
    if rep is None:
        print("→ 还没出报告，跳过（不影响下面两条）")
        skipped.append("预测 4")
    else:
        score_p4(rep, fails)

    # ---------------- 预测 5 ----------------
    print()
    print("=" * 74)
    print("预测 5：iter034 在两个互不相交的留出池上，终局率是否一致")
    print("=" * 74)
    snap = load(ROLL / "grpo_v2_snap" / "iter034.report.json")
    # 这一条要两个池都在：它比的就是两个池之间的差。缺哪边都只能跳过，不能拿单边充数。
    if snap is None or rep is None:
        print(f"→ 缺{'grpo_val' if snap is None else '500 题池'}报告，跳过")
        skipped.append("预测 5")
    else:
        v, L = levels(snap), levels(rep)
        gap = abs(v["terminal"] - L["terminal"])
        print(f"grpo_val(200×4)   终局率 {v['terminal']:.4f}")
        print(f"evaluation(500×4) 终局率 {L['terminal']:.4f}")
        print(f"差 {gap * 100:.2f} pp    门槛 ≤ {P5_POOL_GAP_MAX * 100:.0f} pp    "
              f"{verdict(gap <= P5_POOL_GAP_MAX)}")
        if gap <= P5_POOL_GAP_MAX:
            print("→ 「训练池 vs 留出池」这个划分在第二个模型上重复出来了。判读 A 的结论加强。")
        else:
            print("→ 池间差异有模型依赖。判读 A 要减弱：不能说两个留出池总是一致。")
            fails.append("预测 5／两池终局率差超门槛")

    # ---------------- 预测 6 ----------------
    print()
    print("=" * 74)
    print("预测 6：v1 的 iter039 在 grpo_val 上终局率，决定规则对 v1 是否触发")
    print("=" * 74)
    v1 = load(ROLL / "val_scan" / "v1_iter039.report.json")
    if v1 is None:
        print("→ 还没出报告，跳过")
        skipped.append("预测 6")
    else:
        w = levels(v1)
        print(f"v1_iter039  轨迹 {w['n']}  终局率 {w['terminal']:.4f}  "
              f"成功率 {w['success']:.4f}  条件买对 {w['conditional']:.4f}")
        ok = w["terminal"] >= P6_TERMINAL_MIN
        print(f"≥ {P6_TERMINAL_MIN}  {verdict(ok)}")
        # 上限论证：终局率最大 1.0，所以最好快照最多比它高 (1 − 它) 个点。
        headroom = 1.0 - w["terminal"]
        print(f"最好快照最多能比它高 {headroom * 100:.2f} pp"
              f"（终局率上限 1.0），规则阈值 {RULE_COLLAPSE_PP * 100:.0f} pp")
        if headroom <= RULE_COLLAPSE_PP:
            print("→ **无论其余七个快照是多少**，差都不可能超过阈值 → 规则判「保留 iter039」。")
            print("  v1 那边不必扫整条曲线，两臂对称成立：v2 退到 iter034，v1 保留 iter039。")
        else:
            print("→ 上限论证失效：v1 也可能有塌陷。按预注册，「只评一个点」的省法作废，")
            print("  两臂都要扫完整曲线才能对称地应用规则。")
            fails.append("预测 6／v1 终局率低于门槛，上限论证失效")

    print()
    print("=" * 74)
    if fails:
        print(f"未通过 {len(fails)} 项：")
        for f in fails:
            print(f"  - {f}")
    if skipped:
        print(f"数据不全未打分：{', '.join(skipped)}")
    if not fails and not skipped:
        print("三条预注册全部通过。")
    print("=" * 74)
    # 有未打分的也返回非 0：这样「跑完了但还缺数据」不会被自动化当成全绿。
    return 1 if (fails or skipped) else 0


if __name__ == "__main__":
    raise SystemExit(main())
