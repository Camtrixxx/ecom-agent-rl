#!/usr/bin/env python3
"""预测 2 被推翻这件事，换成「买了的里面买错多少」还站得住吗。

    .venv/bin/python scripts/check_wrong_purchase_denominator.py

纯离线：只读 outputs/rollouts/*.report.json。

## 为什么必须补这一步

18:39 的 D2 判定说 8+ 段错买率 v1 9.66 → v2 7.97，95% 区间 [−3.15, −0.23] 不跨 0，
于是**预测 2 被推翻**。但 `metrics.py:204` 的 `wrong_purchase_rate` 分母是**全部有效
轨迹**，包含那些根本没下手的回合——而「不买当然不会买错」这个陷阱在 roadmap 里已经栽过
一次了（单 run 那次：8+ 段全轨迹口径 0.0598 看着改善一半，换成终局内口径 0.0821 就与
0.103 区间重叠严重）。

同一个陷阱不能栽第二次。所以这里把分母换成**真的下手买了的回合**再打一遍分。

## 但这次有一个先验的反驳，先写下来再算

判定表里 8+ 段的成功率是 +4.16 pp，区间 [+0.62, +7.70]，**也不跨 0**。而靠「少下手」
去压错买率会让成功率**一起降**——不买既不会买错也不会买对。所以「错买率降 + 成功率升」
同时出现，很难由弃买造出来。

写在算之前，是为了让下面的数去检验这个推理，而不是反过来拿数编一个解释。

## 口径两处要说明

1. `reward_types` 是计数，这里直接用计数算比率（池化），而 `metrics.py` 走
   `_per_task_means` 先按题平均。两者有 0.1-0.4 pp 的系统偏移（08-14 14:06 量过），
   所以下面的「全轨迹」列不会和判定表精确相等，看的是**同口径内部的差**。
2. 「买了」= reward_type ∈ {gold_purchase, valid_alternative_purchase,
   partial_alternative_purchase, wrong_purchase}。partial 也算买了——它是硬门过了软
   匹配没满，钱确实花出去了，只是买的不完全对。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_d2 import STRAT_BANDS, V1_RUNS, V2_RUNS, t_two_sample  # noqa: E402

PURCHASE_TYPES = (
    "gold_purchase",
    "valid_alternative_purchase",
    "partial_alternative_purchase",
    "wrong_purchase",
)


def band_stats(rep: dict, band: str) -> dict[str, float] | None:
    b = rep.get("stratified", {}).get("difficulty", {}).get(band)
    if not b:
        return None
    types = b.get("reward_types") or {}
    traj = b.get("trajectories") or 0
    bought = sum(int(types.get(t, 0)) for t in PURCHASE_TYPES)
    wrong = int(types.get("wrong_purchase", 0))
    if not traj:
        return None
    return {
        "traj": traj,
        "bought": bought,
        "wrong": wrong,
        "buy_share": bought / traj,
        "wrong_pooled": wrong / traj,
        "wrong_cond": wrong / bought if bought else float("nan"),
    }


def load(name: str) -> dict | None:
    p = ROOT / "outputs" / "rollouts" / f"{name}.report.json"
    if not p.is_file():
        print(f"!! 缺报告 {p.name}", file=sys.stderr)
        return None
    return json.loads(p.read_text())


def main() -> int:
    groups = {
        "v1": [n for n, *_ in V1_RUNS],
        "v2": [n for n, *_ in V2_RUNS],
    }
    reps = {g: [(n, load(n)) for n in names] for g, names in groups.items()}
    if any(r is None for g in reps.values() for _, r in g):
        print("有报告缺失，下面的 3 对 3 不完整", file=sys.stderr)

    print("=" * 76)
    print("一、每个 run 每段：下手率 / 全轨迹错买率 / 买了里的错买率")
    print("=" * 76)
    print(f"{'run':<14}{'段':>5}{'轨迹':>6}{'买了':>6}{'下手率':>9}"
          f"{'全轨迹错买':>12}{'买了里错买':>12}")
    cell: dict[tuple[str, str, str], list[float]] = {}
    for g, items in reps.items():
        for name, rep in items:
            if rep is None:
                continue
            for band in STRAT_BANDS:
                s = band_stats(rep, band)
                if s is None:
                    continue
                print(f"{name:<14}{band:>5}{s['traj']:>6}{s['bought']:>6}"
                      f"{s['buy_share'] * 100:>8.2f}%{s['wrong_pooled'] * 100:>11.2f}%"
                      f"{s['wrong_cond'] * 100:>11.2f}%")
                for metric in ("buy_share", "wrong_pooled", "wrong_cond"):
                    cell.setdefault((g, band, metric), []).append(s[metric])
        print()

    print("=" * 76)
    print("二、3 对 3（同次判定的合并方差 t、df=4、t=2.78）")
    print("=" * 76)
    print(f"{'段/指标':<24}{'v1 均值':>9}{'v2 均值':>9}{'差':>9}   95% 区间        判定")
    verdicts: dict[str, tuple[float, float, float]] = {}
    for band in STRAT_BANDS:
        for metric in ("buy_share", "wrong_pooled", "wrong_cond"):
            a = cell.get(("v1", band, metric), [])
            b = cell.get(("v2", band, metric), [])
            if len(a) < 2 or len(b) < 2:
                continue
            d, _sp, half, _t = t_two_sample(a, b)
            lo, hi = (d - half) * 100, (d + half) * 100
            ma, mb = sum(a) / len(a) * 100, sum(b) / len(b) * 100
            mark = "跨 0 → 测不出" if lo <= 0 <= hi else \
                   ("v2 更高" if d > 0 else "v2 更低")
            print(f"{band + '/' + metric:<24}{ma:>8.2f}{mb:>9.2f}{d * 100:>+9.2f}   "
                  f"[{lo:+6.2f}, {hi:+6.2f}]  {mark}")
            verdicts[f"{band}/{metric}"] = (d * 100, lo, hi)
        print()

    print("=" * 76)
    print("三、给预测 2 打分")
    print("=" * 76)
    cond = verdicts.get("8+/wrong_purchase_rate") or verdicts.get("8+/wrong_cond")
    share = verdicts.get("8+/buy_share")
    if cond is None or share is None:
        print("8+ 段数据不全，打不了分")
        return 1
    d, lo, hi = cond
    sd_, slo, shi = share
    print(f"8+ 段「买了里的错买率」差 {d:+.2f} pp，区间 [{lo:+.2f}, {hi:+.2f}]")
    print(f"8+ 段「下手率」差       {sd_:+.2f} pp，区间 [{slo:+.2f}, {shi:+.2f}]")
    print()
    if lo <= 0 <= hi:
        print("→ 换成终局内口径后**区间跨 0**：预测 2 的推翻不成立，全轨迹口径那次显著")
        print("  很可能又是分母被弃买稀释造成的。判定表里那句「预测 2 被推翻」要改口。")
    elif hi < 0:
        print("→ 换成终局内口径后区间**仍整体为负**：预测 2 的推翻**站得住**，不是分母效应。")
        if shi < 0:
            print("  !! 但下手率也显著下降，两个效应叠在一起，幅度归因仍要留余地。")
        elif slo > 0:
            print("  且下手率显著上升——v2 是「买得更多且买得更准」，比单看错买率更强。")
        else:
            print("  且下手率测不出差异，说明分母稳定，这个改善是真的落在决策质量上。")
    else:
        print("→ 终局内口径下区间整体为正：v2 在买了的回合里错得更多，方向和全轨迹口径相反，")
        print("  那说明全轨迹那次显著完全是分母效应。要查。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
