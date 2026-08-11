"""把成功率拆成「走到终局的比例」×「走到终局后买对的比例」，并量格式失败。

阶段 C 明确要求把「学会了输出格式」和「决策变好」分开报，否则前者会被误读成后者。
成功率恰好是这两个因子的乘积，所以先各自量，再看分别涨了多少。

归因顺序会改变各因子分到多少个百分点（乘性分解没有唯一的加性归因），所以两种顺序都报。
若两种顺序下决策项都显著，才能说增益不只是「学会了格式」。

用法：
    python scripts/decompose_success.py outputs/rollouts/baseline.jsonl \
        outputs/rollouts/sft.jsonl

口径与 `report_metrics.py` 有一处有意的差别：这里按轨迹条数直接算，不做「题内先平均再对
题 bootstrap」，也不剔除 reward 不可解析的轨迹。所以乘积会和主指标差零点几个百分点，
主指标以 `report_metrics.py` 为准。这里要的是分解，不是点估计。
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecom_agent_rl.environment.tools import TOOL_SCHEMAS  # noqa: E402

TOOL_NAMES = [schema["function"]["name"] for schema in TOOL_SCHEMAS]

# 买对 = 命中目标 asin，或命中等价替代品。partial 只有部分分，不算成功。
SUCCESS_TYPES = ("gold_purchase", "valid_alternative_purchase")


def load(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def terminal_type(record):
    """终局标签取环境的权威判定，不从 messages 猜。没走到终局的按 status 标注。"""
    audit = record.get("audit") or {}
    detail = (audit.get("terminal") or {}).get("reward_detail") or {}
    reward_type = detail.get("reward_type")
    return reward_type or f"no_terminal:{record.get('status')}"


def count_format_failures(records):
    """数「已经选出合法动作、只是没能合法发出去」的回合，并分成两类。

    只看最后一条 assistant 的正文，不在整条消息 JSON 上搜工具名——工具 schema 本身就含
    工具名，在整条 JSON 上搜会命中每一个 no_tool_call 回合（实测 37.9%，即全部）。

    两类要分开，因为可修复性完全不同：
      - block：写了 <tool_call> 但块内混进垃圾，hermes 对整块 json.loads 就失败。
      - prose：压根没用块，把动作写成自然语言。换任何 parser 都救不回来。
    """
    block = prose = 0
    for record in records:
        if record.get("status") != "no_tool_call":
            continue
        assistants = [m for m in record.get("messages", []) if m.get("role") == "assistant"]
        if not assistants:
            continue
        text = str(assistants[-1].get("content") or "")
        if "<tool_call>" in text:
            block += 1
        elif any(name in text for name in TOOL_NAMES):
            prose += 1
    return block, prose


def analyse(path, label):
    records = load(path)
    total = len(records)
    if not total:
        raise SystemExit(f"{path} 是空的")

    types = Counter(terminal_type(r) for r in records)
    reached = sum(c for k, c in types.items() if not k.startswith("no_terminal"))
    bought_right = sum(types.get(t, 0) for t in SUCCESS_TYPES)
    block, prose = count_format_failures(records)

    reach_rate = reached / total
    cond_rate = bought_right / reached if reached else 0.0

    print(f"\n== {label} ==  {total} 条")
    print(f"  走到终局      {reached}/{total} = {reach_rate:.4f}")
    print(f"  终局内买对    {bought_right}/{reached or '-'} = {cond_rate:.4f}")
    print(f"  乘积(=成功率) {reach_rate * cond_rate:.4f}")
    print(f"  格式失败合计  {block + prose}/{total} = {(block + prose) / total:.4f}")
    print(f"    tool_call 块解析失败 {block} = {block / total:.4f}")
    print(f"    动作写成自然语言     {prose} = {prose / total:.4f}")
    for key in ("repeat_loop", "early_abstain", "wrong_purchase", "max_steps"):
        count = types.get(key, 0)
        share = f" / 终局内 {count / reached:.4f}" if reached else ""
        print(f"  {key:14s} {count:4d} = 全体 {count / total:.4f}{share}")
    return reach_rate, cond_rate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="对照轨迹 jsonl")
    parser.add_argument("treatment", help="待比较轨迹 jsonl")
    args = parser.parse_args()

    b_reach, b_cond = analyse(args.baseline, "baseline")
    t_reach, t_cond = analyse(args.treatment, "treatment")

    base = b_reach * b_cond
    print("\n== 增益归因（乘性分解，顺序不同则归属不同，故两种都报）==")
    print(f"  走到终局率  {b_reach:.4f} → {t_reach:.4f}  (×{t_reach / b_reach:.2f})")
    print(f"  条件买对率  {b_cond:.4f} → {t_cond:.4f}  (×{t_cond / b_cond:.2f})")
    print(
        f"  先算格式：+{(t_reach * b_cond - base) * 100:.1f}pp "
        f"然后 +{(t_reach * t_cond - t_reach * b_cond) * 100:.1f}pp"
    )
    print(
        f"  先算决策：+{(b_reach * t_cond - base) * 100:.1f}pp "
        f"然后 +{(t_reach * t_cond - b_reach * t_cond) * 100:.1f}pp"
    )


if __name__ == "__main__":
    main()
