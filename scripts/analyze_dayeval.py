#!/usr/bin/env python3
"""#28 的判定：把 docs/dayeval-preregistration.md 的规则原样执行。

**写于 2026-08-16 04:25，六条腿刚起、一个新数都没有。** 已发布的六个数是见过的（它们是
比较对象），新的六个数全盲。分析代码在数据之前写，等于不给自己留挑口径的余地。

    .venv/bin/python scripts/analyze_dayeval.py

## 主判定用两样本 t，不是配对

预注册写的是「D_W = mean(v2 三个新值) − mean(v1 三个新值)，两样本 t、合并 sd、df=4」。
选这个不是因为它更好，而是因为**已发布的 +3.06 pp [+0.43, +5.70] 就是这么算的**——本脚本
启动时会用已发布的六个数复算一遍去对齐（`_selfcheck`）。换成按 task_id 配对会得到一个更紧
的区间，但那就没法和已发布那个数比了，而"能不能和已发布的比"正是这条臂的全部意义。

## 三个写死的地方

- t₄,₀.₉₇₅ = 2.776 和 n=3+3 是常量，没有 --seeds 参数。缺腿就拒跑，不降级成 df 更小的检验：
  df 不能是看完数据才定的东西。
- 已发布的六个数从各自的 .report.json 读，但会和预注册里抄下来的值对一遍，差超过
  0.0002 就报错退出——防的是"某个 report.json 被后来的重跑覆盖了而我不知道"。
- 次判定的 δ̄ 用**不含 `sft`** 的五份（`sft` 是 08-11 评的，间隔 5 天而非 2 天）。含 `sft`
  的版本照算照登，但不参与判定。这条在预注册里就定了。
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROLLOUTS = ROOT / "outputs" / "rollouts"
POOL = ROOT / "data" / "task_pools" / "evaluation.jsonl"
PY = ROOT / ".venv" / "bin" / "python"

T_CRIT_4 = 2.776  # df=4 双侧 0.975
TARGET_ROWS = 2000

# (新腿名, 已发布名, 组, 预注册里抄下来的已发布成功率, 已发布评测时刻)
LEGS = [
    ("dw_sft",        "sft",        "v1", 0.6061, "08-11 18:12"),
    ("dw_sft_s43",    "sft_s43",    "v1", 0.6243, "08-14 14:24"),
    ("dw_sft_s44",    "sft_s44",    "v1", 0.6288, "08-14 18:38"),
    ("dw_sft_v2",     "sft_v2",     "v2", 0.6626, "08-14 03:58"),
    ("dw_sft_v2_s43", "sft_v2_s43", "v2", 0.6406, "08-14 16:19"),
    ("dw_sft_v2_s44", "sft_v2_s44", "v2", 0.6479, "08-14 18:38"),
]

PUBLISHED_DELTA = 0.030633  # 已发布的 3 对 3：+3.06 pp


def two_sample(a: list[float], b: list[float]) -> tuple[float, float, float, float]:
    """b 组均值 − a 组均值，合并 sd 的两样本 t 区间。返回 (diff, se, lo, hi)。"""
    na, nb = len(a), len(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    sp2 = ((na - 1) * va + (nb - 1) * vb) / (na + nb - 2)
    se = (sp2 * (1 / na + 1 / nb)) ** 0.5
    diff = statistics.mean(b) - statistics.mean(a)
    return diff, se, diff - T_CRIT_4 * se, diff + T_CRIT_4 * se


def rate(name: str) -> float | None:
    p = ROLLOUTS / f"{name}.report.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["overall"]["success_rate"]["mean"]


def strict_ok(name: str) -> bool | None:
    """口径闸门：轨迹里不能有 tolerant_parse 键（它只在非默认口径下才写，自证）。

    不用 grep -c 的退出码判断（无匹配时 grep 退 1，会被误读成"命令失败"）。
    """
    f = ROLLOUTS / f"{name}.jsonl"
    if not f.exists():
        return None
    r = subprocess.run(["grep", "-c", '"tolerant_parse"', str(f)],
                       capture_output=True, text=True)
    n = int(r.stdout.strip() or 0)
    return n == 0


def _selfcheck() -> bool:
    """用已发布的六个数复算 3 对 3，确认本脚本的统计和已发布的口径一致。"""
    v1 = [pub for _, _, g, pub, _ in LEGS if g == "v1"]
    v2 = [pub for _, _, g, pub, _ in LEGS if g == "v2"]
    diff, _, lo, hi = two_sample(v1, v2)
    print(f"自检：用已发布的六个数复算 3 对 3 → {diff * 100:+.2f} pp "
          f"[{lo * 100:+.2f}, {hi * 100:+.2f}]")
    print(f"      已发布的是 +3.06 pp [+0.43, +5.70]", end="  ")
    if abs(diff - PUBLISHED_DELTA) < 0.0005:
        print("→ 点估计对齐 ✅")
        return True
    print(f"→ **对不上**（差 {(diff - PUBLISHED_DELTA) * 100:+.3f} pp），"
          f"统计口径和已发布的不一致，判定不可信")
    return False


def main() -> int:
    print("=" * 78)
    print("#28 六份权重同窗口重评 · 判定")
    print("=" * 78)
    if not _selfcheck():
        return 1
    print()

    # ---- 收数 + 闸门 ----
    rows = []
    for new, pub, group, pub_rate, pub_when in LEGS:
        published = rate(pub)
        if published is None:
            print(f"!! 已发布的 {pub}.report.json 不见了，停")
            return 1
        if abs(published - pub_rate) > 0.0002:
            print(f"!! {pub} 的已发布值变了：预注册抄的是 {pub_rate:.4f}，"
                  f"文件里现在是 {published:.4f}。有人重跑覆盖了它，判定停。")
            return 1
        new_rate = rate(new)
        strict = strict_ok(new)
        rows.append((new, pub, group, pub_rate, pub_when, new_rate, strict))

    print(f"{'权重':<14} {'组':<3} {'已发布':>8} {'时刻':>12} {'新值':>8} {'δ':>9}  口径")
    missing = []
    for new, pub, group, pub_rate, pub_when, new_rate, strict in rows:
        if new_rate is None:
            print(f"{pub:<14} {group:<3} {pub_rate:>8.4f} {pub_when:>12} {'—':>8} {'—':>9}  未出")
            missing.append(pub)
            continue
        flag = "严格 ✅" if strict else "!! 有 tolerant_parse，口径不一致"
        d = new_rate - pub_rate
        print(f"{pub:<14} {group:<3} {pub_rate:>8.4f} {pub_when:>12} "
              f"{new_rate:>8.4f} {d * 100:>+8.2f}pp  {flag}")

    if missing:
        print(f"\n缺 {len(missing)} 条腿（{', '.join(missing)}）→ **不出判定**。")
        print("预注册写的是 n=3 对 n=3、t₄=2.776；缺腿就得改 df，而 df 不能看完数据再定。")
        return 1
    if not all(r[6] for r in rows):
        print("\n!! 有腿的口径不是严格的 → 不出判定，那条腿要重跑。")
        return 1

    # ---- 主判定 ----
    v1_new = [r[5] for r in rows if r[2] == "v1"]
    v2_new = [r[5] for r in rows if r[2] == "v2"]
    diff, se, lo, hi = two_sample(v1_new, v2_new)
    print()
    print("-" * 78)
    print("主判定 · 窗口内的 3 对 3（含双侧训练 run 方差）")
    print("-" * 78)
    print(f"  v1 三个新值 {[f'{x:.4f}' for x in v1_new]}  均值 {statistics.mean(v1_new):.4f}  "
          f"s={statistics.stdev(v1_new) * 100:.2f} pp")
    print(f"  v2 三个新值 {[f'{x:.4f}' for x in v2_new]}  均值 {statistics.mean(v2_new):.4f}  "
          f"s={statistics.stdev(v2_new) * 100:.2f} pp")
    print(f"  D_W = **{diff * 100:+.2f} pp**  95% [{lo * 100:+.2f}, {hi * 100:+.2f}] pp")
    crosses = lo < 0 < hi
    gap = abs(diff - PUBLISHED_DELTA)
    print(f"  与已发布的 +3.06 pp 相差 {(diff - PUBLISHED_DELTA) * 100:+.2f} pp")
    print()
    if crosses:
        print("  → 区间**跨 0**：3 对 3 对窗口不稳健，降级成**测不出**。roadmap 那一节要把")
        print("    「涨 3.06 pp」改成「测不出」，并写明是被 #28 降级的。")
    elif gap <= 0.01:
        print("  → 区间不跨 0 且与已发布相差 ≤ 1 pp：**3 对 3 扛住了窗口控制**，")
        print("    「v2 配方约有 3 pp 收益」站得住，已发布点估计可继续引用。")
    else:
        print(f"  → 区间不跨 0 但与已发布相差 {gap * 100:.2f} pp（> 1 pp）：见下面的偏置判读。")
    if gap > 0.02:
        print(f"  !! 相差 {gap * 100:.2f} pp > 2 pp：已发布那个差值被窗口严重污染。**以新值为准**，")
        print(f"     并把 {(diff - PUBLISHED_DELTA) * 100:+.2f} pp 记成「整组错开的窗口偏置」的")
        print(f"     第二次直接测量（第一次是 #14 的 +0.91 pp）。")

    # ---- 次判定：0.6626 到底高不高 ----
    print()
    print("-" * 78)
    print("次判定 · 已发布的 sft_v2 = 0.6626 有没有偏高")
    print("-" * 78)
    deltas = {r[1]: r[5] - r[3] for r in rows}
    d_all = statistics.mean(deltas.values())
    five = {k: v for k, v in deltas.items() if k != "sft"}
    d_five = statistics.mean(five.values())
    print(f"  δ̄（六份，含 sft，间隔混了 5 天与 2 天）= {d_all * 100:+.2f} pp  ← 只作参考")
    print(f"  δ̄（五份，去掉 sft，间隔同为 ~2 天）  = {d_five * 100:+.2f} pp  ← **判定用这个**")
    dev = deltas["sft_v2"] - d_five
    print(f"  δ(sft_v2) = {deltas['sft_v2'] * 100:+.2f} pp，偏离 δ̄ = **{dev * 100:+.2f} pp**")
    print()
    if dev <= -0.015:
        print("  → ≤ −1.5 pp：**嫌疑成立**，0.6626 是偏高的离群值。SFT 侧 +5.42 pp 应下修到")
        print("    +3.5 pp 量级，和 3 对 3 的 +3.06 pp 一致。roadmap 要按这个改。")
    elif abs(dev) <= 0.010:
        print("  → |偏离| ≤ 1.0 pp：**嫌疑被否**，0.6626 不比别的数更可疑。roadmap 阶段 C 里")
        print("    「已发布的 sft_v2 有偏高嫌疑」那段要删掉，并写明是被 #28 否掉的。")
    else:
        print("  → 落在 −1.5 与 −1.0 之间：**不定**，照登，不改任何措辞。")

    # ---- 附带：漂移是不是公共位移 ----
    print()
    print("-" * 78)
    print("附带 · 六个 δ 的散布 = 六份不同权重、~2 天间隔的漂移量级")
    print("-" * 78)
    s_five = statistics.stdev(list(five.values()))
    print("  " + "  ".join(f"{k}={v * 100:+.2f}" for k, v in deltas.items()))
    print(f"  五份（去 sft）的 s = **{s_five * 100:.2f} pp**，极差 "
          f"{(max(five.values()) - min(five.values())) * 100:.2f} pp")
    print(f"  对照：同一份权重窗口内重评三次的 s = 0.39 pp（#14 的锚）")
    print()
    if s_five < 0.007:
        print("  → s < 0.7 pp：漂移主要是**权重无关的公共位移**，同窗口配对能几乎完全消掉它。")
        print("    这是个好消息：已发布的同窗口配对差都还算可信。")
    elif s_five > 0.015:
        print("  → s > 1.5 pp：漂移带**权重相关**的分量，同窗口配对只能消掉一部分。")
        print("    **这会削弱所有已发布配对差的可信度**，要在 roadmap 里明写。")
    else:
        print("  → s 在 0.7–1.5 pp：公共位移之外还有权重相关分量，但量级不大。照登。")

    _step_effect()
    return 0


# stepmatch = v1 数据 @102 步，三个 seed，评于 08-15 20:40 / 23:50 与 08-16 03:08。
STEPMATCH = ["sft_stepmatch_s42", "sft_stepmatch_s43", "sft_stepmatch_s44"]


def _step_effect() -> None:
    """探索性：v1@102 步 − v1@81 步 = 步数效应。**不是预注册判定，没有阈值。**

    #28 顺带白送了这条臂：dw_sft/_s43/_s44 就是 v1 数据 @81 步的三个 seed，都在同一窗口。
    先前这个量只有排除法软估计（+3.79 pp，08-11 对 08-15 跨天）。见预注册文档那一节。
    """
    print()
    print("-" * 78)
    print("探索性（**非预注册，无阈值**）· 步数效应：v1@102 步 − v1@81 步")
    print("-" * 78)
    at102 = [rate(n) for n in STEPMATCH]
    at81 = [rate(n) for n in ("dw_sft", "dw_sft_s43", "dw_sft_s44")]
    if any(x is None for x in at102 + at81):
        missing = [n for n, r in zip(STEPMATCH + ["dw_sft", "dw_sft_s43", "dw_sft_s44"],
                                     at102 + at81) if r is None]
        print(f"  缺 {', '.join(missing)}，算不了")
        return
    diff, se, lo, hi = two_sample(at81, at102)
    print(f"  v1@81 步（新，同窗口 08-16 05:00）  {[f'{x:.4f}' for x in at81]}  "
          f"均值 {statistics.mean(at81):.4f}")
    print(f"  v1@102 步（stepmatch，20:40–03:08）{[f'{x:.4f}' for x in at102]}  "
          f"均值 {statistics.mean(at102):.4f}")
    print(f"  步数效应 = **{diff * 100:+.2f} pp**  95% [{lo * 100:+.2f}, {hi * 100:+.2f}] pp")
    print(f"  对照：先前的排除法软估计是 +3.79 pp（08-11 对 08-15，跨天，带 ~2 pp 漂移）")
    print()
    print("  **引用时必须带的两句限定**：(1) 两组窗口相差 2–8 h，不是同窗口，只是比先前的")
    print("  跨 5 天好得多；(2) 这是探索性分析，没有预设阈值，**不能**用来推翻或确认 #14 的")
    print("  主判定——#14 判的是数据效应，和步数效应是两个不同的量。")


if __name__ == "__main__":
    sys.exit(main())
