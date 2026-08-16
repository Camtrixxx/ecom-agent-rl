#!/usr/bin/env python3
"""给宽容口径重评的三份权重出对照表，并机械地打分预测 7/8/9。

    .venv/bin/python scripts/score_tolerant_reeval.py

预测写在 docs/optimization-log.md「08-15 17:58」那一节，**写于任何一份重评跑出来之前**。
阈值在下面写成模块常量，理由和 score_snapshot_verdict.py 一样：判读规则一旦被数据影响过，
它就不再是规则了。写成常量的话，事后改口径会在 git diff 里留下痕迹。

## 为什么这个脚本要自己算一个配对差

`eval_model.sh` 只出「对传进去的那个基线」的配对报告，而预测 8 要问的是同一份权重
（grpo_v2/iter034）在**两个口径**下差多少——那不是任何一次评测的基线。所以这里直接拿两个
jsonl 现算，用的是 `metrics.paired_comparison`，和报告里的配对差同一套实现、同一个 bootstrap
种子，不另开一套口径。

## 为什么预测 9 用计数而不用成功率

成功率有采样噪声：单份 2000 回合的绝对值区间半宽约 3.6 pp，两次独立评测按题配对后仍有
±2.7 pp。所以「v1 在两个口径下只差 0.5 pp」这种说法根本验证不了机制。`recovered_tool_calls`
是直接计数——兜底开了几次火就是几次，没有噪声。它才是能证伪实现的那个量。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ecom_agent_rl.evaluation.metrics import (  # noqa: E402
    load_outcomes,
    paired_comparison,
)

# ---- 预注册的阈值（写于看数之前，改动请连同理由写进 optimization-log） ----------
# 预测 7：v2 − v1 同口径配对差落在 0 ~ +1 pp 且区间跨 0 = 打平
P7_LO, P7_HI = 0.0, 0.01
# 落到这个数以下 = 解析损耗解释不了 D2 的 −6.29 pp，roadmap 预测 3 改判「推翻成立」
P7_FALSIFY_AT = -0.03
# 预测 8：同一份 iter034 在两个口径下的配对区间必须跨 0，且兜底触发率 ≤1%
P8_TRIGGER_RATE_MAX = 0.01
# 预测 9：兜底触发次数的预期值。上界按「实测标签率 × 2000 再留 3 倍余量」定：
# 计数本身没有采样噪声，但**哪些题被抽到**有，所以不能钉死成一个点。
P9_EXPECT = {
    "grpo_v1_tol": (0, 30),              # 期望 ≈5（标签率 0.25%）
    "grpo_v2_tol": (120, 320),           # 期望 ≈206（256 条标签里 206 条首块合法）
    "grpo_v2_iter034_tol": (0, 30),      # 期望 ≈2.6（grpo_val 标签率 0.13%）
}

ROLL = ROOT / "outputs/rollouts"
NAMES = ("grpo_v1_tol", "grpo_v2_tol", "grpo_v2_iter034_tol")
# 预测 8 的对照：同一份权重的严格口径评测（#25 的 A 腿产出）
STRICT_COUNTERPART = {"grpo_v2_iter034_tol": "grpo_v2_iter034"}

PASS, FAIL, SKIP = "✅ 命中", "❌ 落空", "⏭ 数据不全，未打分"


def report(name: str) -> dict | None:
    path = ROLL / f"{name}.report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def rate(rep: dict | None) -> float | None:
    """成功率的均值。

    `success_rate` 是 `{mean, ci_low, ci_high}` 而不是一个浮点数——直接 `.get()` 拿来
    格式化会 TypeError（08-15 在 eval_on_val.sh 里正好踩过一次，发车前才发现）。
    """
    if not rep:
        return None
    return ((rep.get("overall") or {}).get("success_rate") or {}).get("mean")


def tolerant_counts(name: str) -> dict[str, int] | None:
    """数这份轨迹里兜底触发了几次、判了几次截断，以及口径标记在不在。

    逐行读而不是 load_outcomes：`Outcome` 里没有这三个字段（它是评测口径的结构，
    不该为了一次核对去改它）。这里只关心三个整数和一个布尔。
    """
    path = ROLL / f"{name}.jsonl"
    if not path.is_file():
        return None
    n = tagged = recovered = truncated = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            n += 1
            if record.get("tolerant_parse"):
                tagged += 1
            recovered += int(record.get("recovered_tool_calls") or 0)
            truncated += int(record.get("truncated_replies") or 0)
    return {
        "trajectories": n,
        "tagged": tagged,
        "recovered": recovered,
        "truncated": truncated,
    }


def pct(x: float | None, width: int = 7) -> str:
    return "  ——   " if x is None else f"{x * 100:>{width}.2f}%"


def main() -> None:
    print("=" * 78)
    print("宽容口径重评（ROLLOUT_TOLERANT_PARSE=1）· 500 题 × k=4 · 温度 0.7")
    print("=" * 78)
    print()
    print("⚠ 这一组数只能三份互相比。08-15 之前发布的全部数字是严格口径采的，")
    print("  并列会得到一个静默错误的结论——两边都是一个成功率，看不出差别在哪。")
    print()

    reps = {name: report(name) for name in NAMES}
    counts = {name: tolerant_counts(name) for name in NAMES}

    print(f"{'权重':<22}{'成功率':>9}{'走到终局':>10}{'轨迹':>7}{'救回':>7}{'截断':>7}{'口径标记':>10}")
    print("-" * 78)
    for name in NAMES:
        rep, cnt = reps[name], counts[name]
        overall = (rep or {}).get("overall") or {}
        statuses = overall.get("statuses") or {}
        total = overall.get("trajectories") or 0
        terminal = (statuses.get("done", 0) / total) if total else None
        c = cnt or {}
        mark = "—"
        if cnt:
            mark = f"{c['tagged']}/{c['trajectories']}" if c["trajectories"] else "空文件"
        print(f"{name:<22}{pct(rate(rep))}{pct(terminal, 9)}"
              f"{c.get('trajectories', 0):>7}{c.get('recovered', 0):>7}"
              f"{c.get('truncated', 0):>7}{mark:>10}")
    print()

    # 口径自证：文件在、却一条标记都没有，说明环境变量没进到 rollout 进程里，
    # 那这份数其实是严格口径的。这比数字难看严重得多，所以先说这个。
    for name in NAMES:
        c = counts[name]
        if c and c["trajectories"] and c["tagged"] != c["trajectories"]:
            print(f"!! {name}: {c['trajectories']} 条里只有 {c['tagged']} 条带 tolerant_parse "
                  f"标记——这份文件混了两个口径，下面的打分对它不成立")
    print()

    verdicts: list[tuple[str, str, str]] = []

    # ---- 预测 7：v2 − v1，两边同口径 -----------------------------------------
    p = (reps["grpo_v2_tol"] or {}).get("paired_vs_baseline") or {}
    if not p.get("paired_tasks"):
        verdicts.append(("预测 7", SKIP, "grpo_v2_tol 的配对报告缺失"))
    else:
        d, lo, hi = p["delta"], p["ci_low"], p["ci_high"]
        crosses_zero = lo <= 0.0 <= hi
        detail = (f"配对差 {d * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]，"
                  f"{p['paired_tasks']} 题配对")
        if d <= P7_FALSIFY_AT:
            verdicts.append(("预测 7", FAIL, detail + f"；≤{P7_FALSIFY_AT * 100:.0f} pp，"
                             "解析损耗解释不了 D2 的 −6.29 pp，roadmap 预测 3 应改判"
                             "「推翻仍然成立」"))
        elif P7_LO <= d <= P7_HI and crosses_zero:
            verdicts.append(("预测 7", PASS, detail + "；落在 0 ~ +1 pp 且跨 0 = 打平"))
        else:
            why = "不在 0 ~ +1 pp 区间内" if not (P7_LO <= d <= P7_HI) else "区间不跨 0"
            verdicts.append(("预测 7", FAIL, detail + f"；{why}"))

    # ---- 预测 8：同一份 iter034，两个口径 ------------------------------------
    tol_name, strict_name = "grpo_v2_iter034_tol", STRICT_COUNTERPART["grpo_v2_iter034_tol"]
    tol_path, strict_path = ROLL / f"{tol_name}.jsonl", ROLL / f"{strict_name}.jsonl"
    c = counts[tol_name]
    if not (tol_path.is_file() and strict_path.is_file()):
        missing = strict_name if not strict_path.is_file() else tol_name
        verdicts.append(("预测 8", SKIP, f"{missing}.jsonl 还没有"))
    else:
        # 顺序是 (基线, 处理) = (严格, 宽容)，所以正号表示宽容口径更高。
        cmp = paired_comparison(load_outcomes(strict_path), load_outcomes(tol_path))
        d, lo, hi = cmp["delta"], cmp["ci_low"], cmp["ci_high"]
        trigger = (c["recovered"] / c["trajectories"]) if c and c["trajectories"] else 0.0
        crosses_zero = lo <= 0.0 <= hi
        detail = (f"宽容 − 严格 = {d * 100:+.2f} pp [{lo * 100:+.2f}, {hi * 100:+.2f}]，"
                  f"兜底触发率 {trigger * 100:.2f}%")
        if crosses_zero and trigger <= P8_TRIGGER_RATE_MAX:
            verdicts.append(("预测 8", PASS, detail + "；区间跨 0 且触发率 ≤1%，"
                             "「兜底只在有标签时起作用」自洽"))
        elif not crosses_zero:
            verdicts.append(("预测 8", FAIL, detail + "；区间不跨 0——同一份权重换个"
                             "解析口径就动了成功率，而它几乎无处可救，机制故事有问题"))
        else:
            verdicts.append(("预测 8", FAIL, detail + f"；触发率超过 "
                             f"{P8_TRIGGER_RATE_MAX * 100:.0f}%，iter034 上兜底开火"
                             "远比标签率预示的多"))

    # ---- 预测 9：兜底触发次数（直接计数，无采样噪声）-------------------------
    missing = [n for n in NAMES if not counts[n] or not counts[n]["trajectories"]]
    if missing:
        verdicts.append(("预测 9", SKIP, f"还缺 {', '.join(missing)}"))
    else:
        bad = []
        for name, (lo_n, hi_n) in P9_EXPECT.items():
            got = counts[name]["recovered"]
            if not lo_n <= got <= hi_n:
                bad.append(f"{name} 救回 {got} 次，预期 {lo_n}~{hi_n}")
        got_all = "，".join(f"{n}={counts[n]['recovered']}" for n in NAMES)
        if bad:
            verdicts.append(("预测 9", FAIL, f"{got_all}；越界：{'；'.join(bad)}。"
                             "计数不受采样噪声影响，所以这是实现问题而不是运气问题"
                             "——先查 tool_call_recovery.py 再看任何成功率"))
        else:
            verdicts.append(("预测 9", PASS, f"{got_all}；三份都在预注册区间内，"
                             "兜底只在有标签的地方开火"))

    print("预注册预测的打分")
    print("-" * 78)
    for tag, verdict, detail in verdicts:
        print(f"{tag}  {verdict}")
        print(f"        {detail}")
    print()

    hit = sum(1 for _, v, _ in verdicts if v == PASS)
    scored = sum(1 for _, v, _ in verdicts if v != SKIP)
    print(f"合计 {hit}/{scored} 命中（{len(verdicts) - scored} 条数据不全未打分）")


if __name__ == "__main__":
    main()
