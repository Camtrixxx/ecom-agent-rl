#!/usr/bin/env python3
"""量「那 256 条写成正文标签的轨迹，有多少是宽容重解析就能救回来的」。

    .venv/bin/python scripts/analyze_tag_recovery.py

纯离线：只读 outputs/rollouts/*.jsonl，不碰 GPU、不连环境池，可以和任何训练/评测并跑。

## 为什么这件事值得单独量

08-14 14:06 直读了安装版 vLLM 的源码（`.venv/.../vllm/tool_parsers/hermes_tool_parser.py`），
`extract_tool_calls` 的行为是**全有或全无**：

    raw_function_calls = [json.loads(match[0] if match[0] else match[1])
                          for match in function_call_tuples]      # 第 91-94 行
    ...
    except Exception:                                             # 第 116-120 行
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output)

一个列表推导式里 `json.loads` 所有块，任何一块坏掉 → 整个 except → `tool_calls=[]`、
把**整段输出**当 content 返回。于是「第一块完全合法、尾巴上多了半块被 1024 截断的垃圾」
这种输出，会连那个合法的第一块一起丢掉。

而我们这边也没有兜底：`llm.py:287` 是 `return dict(message)`，逐字返回服务端给的
message，不看 content 里有没有标签。`agent.py:233-238` 拿不到 `tool_calls` 就立刻
`status = NO_TOOL_CALL` 并 return。

两段一拼，`llm.py:264-266` 那句注释的前提就不成立了：

    「只有 content、没有 tool_calls 是模型真实的决策（它选择说话而不动手），
      那种情况要如实记成 NO_TOOL_CALL，重试它等于篡改模型行为。」

当 content 里**带着 `<tool_call>` 标签**时，这不是「选择说话而不动手」，是解析事故。
272 条 no_tool_call 里有 256 条属于这一类，却被记成了模型决策。这和 finish_reason 那个
洞是同一种错（把基础设施损耗记成模型行为），只是换了一个轴。

## 这个脚本回答什么、不回答什么

回答：**若客户端加一层宽容重解析**（按顺序取第一个能 json.loads、名字合法、arguments
是对象的块——正好对应 `agent.py:240-249` 只留 `calls[0]` 的语义），有多少条能拿回一次
合法调用。

不回答：拿回这一次调用之后，那条轨迹会不会**买对**。恢复的是「走到终局的机会」，不是
成功本身。所以下面只给成功率的**上界估计** = 可恢复率 × 该模型自己的条件买对率，并明确
标成估计值。真数只能靠改完代码重跑一次评测。

顺带把三个之前没量的东西一起量了，因为读同一批文件是免费的：
  - 块的长度分布 —— 用来检验 14:50 那个反推：30-99 块要塞进 1024 token，平均每块
    ≤34 token、99 块的情形 ≤10 token，所以那些块必然是残缺的偏 JSON 而不是完整调用。
  - 复读的块是不是**逐字相同** —— 分「贪心重复」和「反复重试不同参数」。
  - 标签是不是**只**出现在最后一条 assistant 消息里 —— 这是 14:50 从 agent.py 读出的
    前提，这里用数据再确认一遍（应当 100%）。
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# 逐字抄 vLLM 的正则（vllm/tool_parsers/hermes_tool_parser.py:38-40）。它用 `regex`
# 包、我们用 `re`，对这个模式两者行为一致：没有可变长度回顾、没有递归。
TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
START = "<tool_call>"
END = "</tool_call>"

try:
    from ecom_agent_rl.environment.tools import TOOL_SCHEMAS

    VALID_NAMES = {s["function"]["name"] for s in TOOL_SCHEMAS}
except Exception as exc:  # pragma: no cover
    print(f"!! 导不进 TOOL_SCHEMAS（{exc}），名字合法性一律算通过", file=sys.stderr)
    VALID_NAMES = None

# 成功的判据**必须**复用 metrics.py，不能自己重写一份。
#
# 我第一版写的是 `reward >= 1.0`，错得很典型：Reward v3 的取值域是 -0.85 到 1.0
# （metrics.py:31-41），成功的定义是 `reward_type in SUCCESS_TYPES`
# （metrics.py:44，= gold_purchase | valid_alternative_purchase），而
# `valid_alternative_purchase` 的 reward 是 **0.55**——按 `>= 1.0` 数会把这一类
# 全判成失败。另外 `reward_valid=False` 的轨迹要整条剔除（metrics.py:183），
# 不是当成失败计入分母。
#
# 一句话：同一个指标两份实现，早晚分叉。这里只导，不抄。
from ecom_agent_rl.evaluation.metrics import outcome_from_record  # noqa: E402


def blocks_of(text: str) -> list[tuple[str, bool]]:
    """→ [(块内容, 是否闭合)]。顺序与 vLLM findall 一致。"""
    out = []
    for m in TOOL_CALL_REGEX.finditer(text):
        if m.group(1) is not None:
            out.append((m.group(1), True))
        else:
            out.append((m.group(2), False))
    return out


def vllm_would_parse(text: str) -> bool:
    """复现 vLLM 的全有或全无：所有块都得 loads 成功且有 name/arguments。"""
    bs = blocks_of(text)
    if not bs:
        return False
    try:
        for body, _ in bs:
            obj = json.loads(body)
            obj["name"], obj["arguments"]
    except Exception:
        return False
    return True


def tolerant_first_call(text: str) -> dict | None:
    """宽容重解析：按顺序返回第一个合法块，坏块跳过。对应 agent.py 只留 calls[0]。"""
    for body, _ in blocks_of(text):
        try:
            obj = json.loads(body)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        args = obj.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        if VALID_NAMES is not None and name not in VALID_NAMES:
            continue
        if not isinstance(args, dict):
            continue
        return obj
    return None


def analyze(path: Path) -> dict | None:
    if not path.exists():
        return None
    total = tagged = 0
    recoverable = 0
    vllm_ok = 0            # 本该被 vLLM 正常解析的（若 >0 说明口径有问题，见下）
    tags_not_terminal = 0  # 标签出现在非最后一条 assistant 消息里的轨迹数
    dangling = 0           # 最后一块是未闭合的 → 硬截断签名
    all_identical = 0      # 所有块逐字相同 → 贪心复读
    status_of_tagged: Counter = Counter()
    status_of_recoverable: Counter = Counter()
    block_counts: list[int] = []
    block_lens: list[int] = []
    spew_block_lens: list[int] = []   # 只统计 ≥30 块那一峰的块长
    spew_mode_share: list[float] = []  # ≥30 块的轨迹里，最常见的块占了多少
    spew_uniq_ratio: list[float] = []  # 同上，不同块 / 总块
    n_done = 0
    n_valid = 0      # reward_valid=True 的轨迹数，成功率的分母（metrics.py:183）
    n_success = 0    # reward_type ∈ SUCCESS_TYPES
    n_dropped = 0    # reward_valid=False，整条剔除

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            total += 1
            if rec.get("done"):
                n_done += 1
            try:
                oc = outcome_from_record(rec)
            except Exception:
                oc = None
            if oc is not None:
                if oc.reward_valid:
                    n_valid += 1
                    if oc.success:
                        n_success += 1
                else:
                    n_dropped += 1

            messages = rec.get("messages") or []
            assistants = [
                m for m in messages
                if m.get("role") == "assistant" and START in (m.get("content") or "")
            ]
            if not assistants:
                continue
            tagged += 1
            status_of_tagged[rec.get("status")] += 1

            # 前提检查：标签只该出现在最后一条 assistant 消息里（agent.py:233-238 一拿到
            # 无 tool_calls 的回复就 return，所以带标签的回复必然是终局那一条）。
            last_assistant = next(
                (m for m in reversed(messages) if m.get("role") == "assistant"), None
            )
            if len(assistants) > 1 or assistants[0] is not last_assistant:
                tags_not_terminal += 1

            text = "".join(m.get("content") or "" for m in assistants)
            bs = blocks_of(text)
            block_counts.append(len(bs))
            lens = [len(b) for b, _ in bs]
            block_lens.extend(lens)
            if len(bs) >= 30:
                spew_block_lens.extend(lens)
            if bs and not bs[-1][1]:
                dangling += 1
            if len(bs) > 1 and len({b for b, _ in bs}) == 1:
                all_identical += 1
            # `块全同` 太严了：它只排除「全部逐字相同」，一条 97 块里有 96 块相同、
            # 最后一块被截断的轨迹会算成「不全同」，而那显然就是复读。所以再量两个
            # 连续量——众数块占比（最常见的块占了多少）与去重率（不同块 / 总块）。
            #   众数占比高 + 去重率低 → 近似逐字复读 → 该上重复惩罚 / KL 锚。
            #   两者都居中          → 反复重试不同参数 → 是决策层面在打转，罚重复没用。
            if len(bs) >= 30:
                bodies = [b for b, _ in bs]
                top = Counter(bodies).most_common(1)[0][1]
                spew_mode_share.append(top / len(bodies))
                spew_uniq_ratio.append(len(set(bodies)) / len(bodies))

            if vllm_would_parse(text):
                vllm_ok += 1
            if tolerant_first_call(text) is not None:
                recoverable += 1
                status_of_recoverable[rec.get("status")] += 1

    term = n_done / total if total else float("nan")
    sr = n_success / n_valid if n_valid else float("nan")
    cond = sr / term if term else float("nan")
    return dict(
        name=path.stem, total=total, tagged=tagged, recoverable=recoverable,
        vllm_ok=vllm_ok, tags_not_terminal=tags_not_terminal, dangling=dangling,
        all_identical=all_identical, block_counts=block_counts,
        block_lens=block_lens, spew_block_lens=spew_block_lens,
        spew_mode_share=spew_mode_share, spew_uniq_ratio=spew_uniq_ratio,
        status_of_tagged=status_of_tagged, status_of_recoverable=status_of_recoverable,
        term=term, sr=sr, cond=cond, n_dropped=n_dropped,
    )


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.2f}%" if b else "—"


def main() -> None:
    args = sys.argv[1:]
    if args:
        paths = [Path(a) for a in args]
    else:
        d = ROOT / "outputs" / "rollouts"
        paths = [d / f"{n}.jsonl" for n in ("sft", "grpo", "sft_v2", "grpo_v2")]

    rows = [r for r in (analyze(p) for p in paths) if r]
    if not rows:
        print("没有可读的轨迹文件")
        return

    print("=" * 78)
    print("一、标签轨迹与宽容重解析的可恢复量")
    print("=" * 78)
    print(f"{'模型':<10}{'总数':>7}{'标签':>7}{'标签率':>9}"
          f"{'可恢复':>8}{'占标签':>9}{'占总数':>9}{'vLLM本可解析':>13}")
    for r in rows:
        print(f"{r['name']:<10}{r['total']:>7}{r['tagged']:>7}"
              f"{pct(r['tagged'], r['total']):>9}{r['recoverable']:>8}"
              f"{pct(r['recoverable'], r['tagged']):>9}"
              f"{pct(r['recoverable'], r['total']):>9}{r['vllm_ok']:>13}")
    print()
    print("`vLLM本可解析` 应当恒为 0：>0 意味着这条轨迹的标签本该被解析成 tool_calls 却")
    print("没有，那说明口径错了（比如标签来自被压缩塞回历史的旧消息），要先查这个再看别的。")

    print()
    print("=" * 78)
    print("二、块数分布（双峰在不在，峰在哪）")
    print("=" * 78)
    buckets = [(1, 1), (2, 2), (3, 4), (5, 9), (10, 29), (30, 99), (100, 10 ** 9)]
    print(f"{'模型':<10}" + "".join(f"{f'{lo}-{hi}' if lo != hi else str(lo):>9}"
                                    for lo, hi in buckets[:-1]) + f"{'100+':>9}")
    for r in rows:
        cells = []
        for lo, hi in buckets:
            cells.append(sum(1 for c in r["block_counts"] if lo <= c <= hi))
        print(f"{r['name']:<10}" + "".join(f"{c:>9}" for c in cells))

    print()
    print("=" * 78)
    print("三、机制判别：截断签名 / 逐字复读 / 块长")
    print("=" * 78)
    print(f"{'模型':<10}{'末块未闭合':>12}{'占标签':>9}{'块全同':>8}"
          f"{'块长中位':>10}{'复读峰块长中位':>16}{'非终局标签':>12}")
    for r in rows:
        bl = r["block_lens"]
        sbl = r["spew_block_lens"]
        med = f"{statistics.median(bl):.0f}" if bl else "—"
        smed = f"{statistics.median(sbl):.0f}" if sbl else "—"
        print(f"{r['name']:<10}{r['dangling']:>12}{pct(r['dangling'], r['tagged']):>9}"
              f"{r['all_identical']:>8}{med:>10}{smed:>16}{r['tags_not_terminal']:>12}")
    print()
    print(f"{'模型':<10}{'复读轨迹数':>12}{'众数块占比中位':>16}{'去重率中位':>12}")
    for r in rows:
        ms, ur = r["spew_mode_share"], r["spew_uniq_ratio"]
        if not ms:
            print(f"{r['name']:<10}{0:>12}{'—':>16}{'—':>12}")
            continue
        print(f"{r['name']:<10}{len(ms):>12}{statistics.median(ms):>16.3f}"
              f"{statistics.median(ur):>12.3f}")
    print("众数块占比高 + 去重率低 → 近似逐字复读，该上重复惩罚 / KL 锚；")
    print("两者都居中 → 是在反复重试**不同参数**，即决策层面打转，罚重复不解决问题。")
    print()
    print("块长是**字符数**。粗算 token ≈ 字符 / 3（JSON 里英文键名多、中文少）。")
    print("14:50 的反推是：30-99 块要塞进 max_tokens=1024，平均每块 ≤34 token（≈100 字符），")
    print("99 块的情形 ≤10 token（≈30 字符）。若上面`复读峰块长中位`确实是几十个字符，")
    print("反推成立——那些块是残缺的偏 JSON，不是完整调用。若是几百字符，反推不成立，")
    print("那就说明这些回复根本没被 1024 截断，得回去查 max_tokens 到底有没有生效。")

    print()
    print("=" * 78)
    print("四、成功率的上界估计（**估计值，不是测量值**）")
    print("=" * 78)
    print("恢复一次合法调用只是把轨迹从「当场判死」放回「继续走」，它之后会不会买对是未知的。")
    print("所以这里按该模型自己的条件买对率折算，给的是上界估计。真数只能改完代码重跑评测。")
    print(f"{'模型':<10}{'终局率':>9}{'成功率':>9}{'条件买对':>10}"
          f"{'可恢复率':>10}{'成功率上界估计':>16}{'增量':>9}{'剔除':>6}")
    for r in rows:
        rr = r["recoverable"] / r["total"] if r["total"] else 0.0
        est = r["sr"] + rr * r["cond"]
        print(f"{r['name']:<10}{r['term']:>9.4f}{r['sr']:>9.4f}{r['cond']:>10.4f}"
              f"{rr:>10.4f}{est:>16.4f}{est - r['sr']:>+9.4f}{r['n_dropped']:>6}")
    print()
    print("`成功率` 复用 metrics.py 的定义：reward_type ∈ {gold_purchase,")
    print("valid_alternative_purchase}，分母只算 reward_valid=True 的（`剔除` 列是被丢掉的）。")
    print("**先核对 grpo_v2 这一格是不是 0.6225**——不是就说明本脚本读错了字段，下面全别信。")
    print()
    by = {r["name"]: r for r in rows}
    if "grpo_v2" in by and "grpo" in by:
        a, b = by["grpo_v2"], by["grpo"]
        ra = a["recoverable"] / a["total"] if a["total"] else 0.0
        rb = b["recoverable"] / b["total"] if b["total"] else 0.0
        gap = a["sr"] - b["sr"]
        gap_est = (a["sr"] + ra * a["cond"]) - (b["sr"] + rb * b["cond"])
        print(f"grpo_v2 − grpo(v1) 实测差 {gap:+.4f}；两边都加宽容重解析后的估计差 "
              f"{gap_est:+.4f}（{gap_est - gap:+.4f}）。")
        print("注意这是**两边都修**之后的差：v1 的标签率本来就低，所以修解析对它几乎无益，")
        print("差值的改善基本全来自 v2 —— 也就是说这一层损耗是 v2 特有的，不是共同背景。")

    print()
    print("=" * 78)
    print("五、标签轨迹的状态构成")
    print("=" * 78)
    for r in rows:
        if not r["tagged"]:
            continue
        tot = dict(sorted(r["status_of_tagged"].items(), key=lambda kv: -kv[1]))
        rec = dict(sorted(r["status_of_recoverable"].items(), key=lambda kv: -kv[1]))
        print(f"{r['name']}: 标签 {tot}")
        print(f"{'':<{len(r['name']) + 2}}可恢复 {rec}")
    print()
    print("若标签轨迹里出现 no_tool_call 之外的状态（比如 max_steps / repeat_loop），说明")
    print("标签不只在终局那一条消息上出现过，`非终局标签` 那一列应当同时为正，两者要对得上。")


if __name__ == "__main__":
    main()
