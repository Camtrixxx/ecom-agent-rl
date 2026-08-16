"""把 no_tool_call 的回合分类：格式漂移、上下文压缩后遗症，还是真的学会了弃买。

    .venv/bin/python scripts/classify_no_tool_call.py \\
        outputs/rollouts/grpo_v2.jsonl outputs/rollouts/sft_v2.jsonl

## 要判什么

grpo_v2 的 no_tool_call 从 sft_v2 的 7 条涨到 272 条（13.6%），端到端 −6.29 pp 全部来自
这一项（乘性分解：终局率 0.9945 → 0.8615，而条件买对率反而升到四个模型里最高的
0.7226）。三种成因，处理方式完全不同：

- **格式漂移**：kl_beta=0 没有 anchor，模型把 <tool_call> 写坏或写到 1024 token 上限被
  截断，hermes parser 认不出来。→ 要加 anchor（KL / 或对 no_terminal 单独加罚 / 早停）。
- **上下文压缩后遗症**：compact_messages 丢掉历史组之后，prompt 里可能不再有任何
  tool_call 的样例，模型跟着不发工具调用了。→ 这是评测链路的 bug，不是模型退化，
  改 GRPO 超参完全治错地方。
- **学会弃买**：输出了语义完整的「我先不买」。→ 是 reward 设计问题：invalid_reward
  −1.0 虽比 wrong_purchase −0.85 更低，但 env 对「没买」只赔 −0.15；模型没学出
  「主动结束回合」和「干脆不发动作」的区别。→ 要改的是 reward，不是加 KL。

分不清就配错下一个 6.5h run，所以这一步必须先做。

## 判据

**原计划里的两条判据都不成立，看代码才发现：**

- `finish_reason` 全链路没有存过（llm.py:258 只取 message；agent.py:112 的 as_record
  里也没有这个字段）。截断只能靠内容特征认，认不出来的要老实归到 other。
- 「空响应」不可能出现在这里：llm.py:267 判定 content 和 tool_calls 皆空时会重试，
  耗尽后是 **EMPTY_RESPONSE**，是另一个 status。所以 no_tool_call 一定有正文。
  （只有纯空白的正文是例外——`" "` 在 Python 里为真，过得了那道判空。）

改用三个已经落盘的量做交叉表，它们比原计划的判据更能分开上面三种成因：

- `env_steps`：第 0 步就不发工具调用 = 一开始就不动手；第 15 步之后才停 = 长回合走死。
- `usage.compactions` / `dropped_groups`：这一条回合有没有被压缩过。要和**同一文件里
  done 的回合**比，不是看绝对值——长回合本来就更容易被压缩。
- `rejection_count`：先被拒几次再放弃，是「试过但格式一直不对」。
"""

import json
import re
import sys
from collections import Counter

PATHS = sys.argv[1:] or ["outputs/rollouts/grpo_v2.jsonl"]

# 弃买措辞。温度 0.7 下措辞会飘，所以用宽的关键词而不是整句匹配。
ABSTAIN = re.compile(
    r"不(买|购买|下单)|先不|暂(时|不)|没有(合适|符合)|不符合|找不到|无法(满足|找到)"
    r"|放弃|建议(您|你)?(不|另)|not (buy|purchase)|no (suitable|matching)|cannot find",
)
# hermes 模板的工具调用长这样：<tool_call>\n{"name": ...}\n</tool_call>
# 开标签在、闭标签不在 = 写到一半（很可能撞了 max_tokens=1024）。
OPEN_TAG = re.compile(r"<tool_call>|<\|tool_call")
CLOSE_TAG = re.compile(r"</tool_call>")
# 没有开标签但有 JSON 动作骨架，说明它想调工具但没套模板。
BARE_JSON = re.compile(r'"name"\s*:\s*"(search|click|think|buy|back|end)')
ENDING = tuple("。！？.!?\"」）)")


def last_assistant(rec):
    for m in reversed(rec.get("messages") or []):
        if m.get("role") == "assistant":
            return m.get("content") or ""
    return ""


def bucket_of(text):
    if OPEN_TAG.search(text) and not CLOSE_TAG.search(text):
        return "1_tool_call开标签未闭合(疑似截断)"
    if BARE_JSON.search(text):
        return "2_裸JSON动作(没套模板)"
    if not text.strip():
        return "3_纯空白正文"
    if len(text) >= 900 and not text.rstrip().endswith(ENDING):
        return "4_长且无结束标点(疑似截断)"
    if ABSTAIN.search(text):
        return "5_语义完整的弃买"
    return "6_other"


def steps_band(n):
    if n == 0:
        return "0步"
    if n <= 2:
        return "1-2步"
    if n <= 5:
        return "3-5步"
    if n <= 10:
        return "6-10步"
    return "11+步"


for path in PATHS:
    buckets, bands, samples = Counter(), Counter(), {}
    lengths, rejects = [], Counter()
    n_total = n_ntc = 0
    comp_ntc = comp_done = 0
    n_done = 0

    try:
        fh = open(path)
    except OSError as exc:
        print(f"!! 打不开 {path}: {exc}\n")
        continue

    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n_total += 1
            status = rec.get("status")
            compacted = int((rec.get("usage") or {}).get("compactions") or 0) > 0

            if status == "done":
                n_done += 1
                comp_done += compacted
                continue
            if status != "no_tool_call":
                continue

            n_ntc += 1
            comp_ntc += compacted
            text = last_assistant(rec)
            lengths.append(len(text))
            bands[steps_band(int(rec.get("env_steps") or 0))] += 1
            rejects[int(rec.get("rejection_count") or 0)] += 1
            kind = bucket_of(text)
            buckets[kind] += 1
            samples.setdefault(kind, []).append(text[-300:])

    print("=" * 74)
    print(f"{path}   总回合 {n_total}   no_tool_call {n_ntc}"
          f"（{n_ntc / max(n_total, 1):.1%}）")
    print("=" * 74)
    if not n_ntc:
        print("  没有 no_tool_call 回合\n")
        continue

    print("\n[分类]")
    for kind in sorted(buckets):
        print(f"  {kind:<30} {buckets[kind]:>5}  ({buckets[kind] / n_ntc:.1%})")
    drift = sum(v for k, v in buckets.items() if k[0] in "1234")
    abstain = buckets.get("5_语义完整的弃买", 0)
    print(f"  {'—' * 30}")
    print(f"  {'格式类(1+2+3+4)':<30} {drift:>5}  ({drift / n_ntc:.1%})")
    print(f"  {'弃买类(5)':<30} {abstain:>5}  ({abstain / n_ntc:.1%})")

    print("\n[停在第几步]")
    for b in ("0步", "1-2步", "3-5步", "6-10步", "11+步"):
        if bands.get(b):
            print(f"  {b:<10} {bands[b]:>5}  ({bands[b] / n_ntc:.1%})")

    print("\n[被拒次数]")
    for r in sorted(rejects):
        print(f"  {r} 次 {rejects[r]:>5}")

    # 关键对照：压缩率要和同文件的 done 比，绝对值没有意义。
    print("\n[上下文压缩率]（和同文件 done 的回合对比，判断是不是压缩造成的）")
    print(f"  no_tool_call 被压缩过 {comp_ntc}/{n_ntc} = {comp_ntc / n_ntc:.1%}")
    if n_done:
        print(f"  done         被压缩过 {comp_done}/{n_done} = {comp_done / n_done:.1%}")
        if comp_ntc / n_ntc > 2 * max(comp_done / n_done, 1e-9):
            print("  → no_tool_call 的压缩率显著更高，压缩是嫌疑成因，先查评测链路")
        else:
            print("  → 压缩率没有明显偏高，压缩不是主因")

    lengths.sort()
    print(f"\n[最后一条 assistant 长度]  中位 {lengths[len(lengths) // 2]}"
          f"  P10 {lengths[len(lengths) // 10]}  最短 {lengths[0]}  最长 {lengths[-1]}")

    for kind in sorted(samples):
        print(f"\n{'-' * 74}\n{kind}  共 {buckets[kind]} 条，前 2 条的尾部 300 字符")
        for i, s in enumerate(samples[kind][:2]):
            print(f"  [{i}] {'(纯空白)' if not s.strip() else ''}"
                  f"{s.replace(chr(10), ' ⏎ ')}")
    print()
