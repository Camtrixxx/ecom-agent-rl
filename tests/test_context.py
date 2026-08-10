"""上下文压缩的不变量。

压缩是长回合能不能跑满 35 步的前提（实测第 18 步撞 24576），但它有两个很容易
悄悄搞坏的地方，这里逐条钉住：

1. 删历史不能删成半截 tool_call——对话结构不合法，vLLM 直接报错。
2. 删历史不能删掉「已经搜过什么」——`repeat_loop` 是 −0.65，专罚重复动作。
   参考实现整组丢弃，正好会造成它要避免的失败。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ecom_agent_rl.rollout.context import (
    SUMMARY_MARKER,
    ContextBudgetError,
    compact_messages,
)


def call(name: str, args: dict[str, Any], call_id: str = "c") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def conversation(steps: int, observation: str = "页面内容") -> list[dict[str, Any]]:
    """system + user + steps 个（assistant tool_call, tool 响应）组。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "系统提示"},
        {"role": "user", "content": "任务说明"},
    ]
    for i in range(steps):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("search_products", {"query": f"查询{i}"}, f"c{i}")],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"c{i}", "content": f"{observation}{i}"}
        )
    return messages


def words(messages: list[dict[str, Any]]) -> int:
    """朴素计数器：按字符数算，测试里够用且可预测。"""
    total = 0
    for message in messages:
        total += len(str(message.get("content") or ""))
        for c in message.get("tool_calls") or []:
            total += len(str(c["function"]["arguments"]))
    return total


# --- 不压缩的情况 ---------------------------------------------------------


def test_a_conversation_within_budget_is_returned_unchanged():
    messages = conversation(2)
    out, stats = compact_messages(messages, words, 10_000)
    assert out == messages
    assert stats.dropped_groups == 0


def test_the_input_is_never_mutated():
    """调用方的 trajectory.messages 必须保持完整——它是要写盘的训练数据。"""
    messages = conversation(20)
    before = json.dumps(messages, ensure_ascii=False)
    compact_messages(messages, words, 200)
    assert json.dumps(messages, ensure_ascii=False) == before


# --- 结构合法性 -----------------------------------------------------------


def test_compaction_never_leaves_a_tool_call_without_its_response():
    """半截 tool_call 会让 vLLM 直接拒绝整个请求。"""
    messages = conversation(30)
    out, stats = compact_messages(messages, words, 400)
    assert stats.dropped_groups > 0
    pending: list[str] = []
    for message in out:
        if message["role"] == "assistant" and message.get("tool_calls"):
            assert not pending, "上一轮的 tool_call 没有响应"
            pending = [c["id"] for c in message["tool_calls"]]
        elif message["role"] == "tool":
            assert pending and message["tool_call_id"] == pending.pop(0)
    assert not pending


def test_the_anchor_survives_any_amount_of_compaction():
    messages = conversation(50)
    out, _ = compact_messages(messages, words, 300)
    assert out[0] == {"role": "system", "content": "系统提示"}
    assert out[1] == {"role": "user", "content": "任务说明"}


def test_the_latest_observation_is_always_kept():
    """模型靠它决策，删了就只能瞎猜。"""
    messages = conversation(40, observation="很长的页面" * 20)
    out, _ = compact_messages(messages, words, 500)
    assert out[-1]["role"] == "tool"
    assert out[-1]["content"] == messages[-1]["content"]


def test_the_result_actually_fits_the_budget():
    messages = conversation(40, observation="页面" * 50)
    out, stats = compact_messages(messages, words, 900)
    assert words(out) <= 900
    assert stats.final_tokens <= 900


# --- 摘要保住「已经做过什么」（与参考实现的关键差别）----------------------


def test_dropped_history_leaves_a_summary_of_what_was_already_searched():
    """整组丢弃会让模型重复搜索，而 repeat_loop 是 −0.65。"""
    messages = conversation(30)
    out, stats = compact_messages(messages, words, 400)
    assert stats.dropped_groups > 0
    summaries = [m for m in out if str(m.get("content", "")).startswith(SUMMARY_MARKER)]
    assert len(summaries) == 1, "应恰好有一条摘要"
    body = summaries[0]["content"]
    # 摘要有界，最老的会被挤掉；但被删掉的那些步必须留下动作记录，
    # 而不是整组消失（那就是参考实现的行为）。
    assert "查询" in body, "被删的历史没有留下任何动作记录"
    assert f"{stats.dropped_groups} 步" in body


def test_the_summary_reports_how_many_steps_were_omitted():
    messages = conversation(30)
    out, stats = compact_messages(messages, words, 400)
    summary = next(m for m in out if str(m["content"]).startswith(SUMMARY_MARKER))
    assert f"{stats.dropped_groups} 步" in summary["content"]


def test_opened_products_are_recorded_in_the_summary():
    messages = conversation(1)
    for i in range(20):
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("open_product", {"asin": f"90000000000{i}"}, f"o{i}")],
            }
        )
        messages.append({"role": "tool", "tool_call_id": f"o{i}", "content": "详情" * 30})
    out, _ = compact_messages(messages, words, 500)
    summary = next(m for m in out if str(m["content"]).startswith(SUMMARY_MARKER))
    assert "商品9000000000" in summary["content"], "打开过的商品没有被记下"


def test_the_summary_is_bounded_so_it_cannot_grow_back():
    """摘要若随回合线性增长，压缩就只是把问题推后。"""
    short, _ = compact_messages(conversation(20), words, 400)
    long, _ = compact_messages(conversation(200), words, 400)
    a = next(m for m in short if str(m["content"]).startswith(SUMMARY_MARKER))
    b = next(m for m in long if str(m["content"]).startswith(SUMMARY_MARKER))
    # 10 倍步数，摘要不该跟着涨 10 倍。
    assert len(b["content"]) < len(a["content"]) * 3


def test_earlier_summaries_are_carried_forward_when_compacted_again():
    """第二次压缩不能把第一次记住的动作丢掉。"""
    first, _ = compact_messages(conversation(20), words, 400)
    # 接着往后走几步，再压一次。
    grown = list(first)
    for i in range(20, 30):
        grown.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [call("search_products", {"query": f"查询{i}"}, f"c{i}")],
            }
        )
        grown.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"页面{i}"})
    second, _ = compact_messages(grown, words, 400)
    summaries = [m for m in second if str(m["content"]).startswith(SUMMARY_MARKER)]
    assert len(summaries) == 1, "二次压缩不该并存两条摘要"
    # 步数必须累计：只报本次省略量的话，长回合会告诉模型「省略了 1 步」而实际几十步。
    import re
    total = int(re.search(r"已省略 (\d+) 步", summaries[0]["content"]).group(1))
    kept = sum(1 for m in second if m["role"] == "assistant")
    assert total + kept == 30, f"步数账不平: 省略 {total} + 保留 {kept}"


def test_the_summary_says_how_many_items_it_had_to_elide():
    """项数超上限时要说明省略了多少，不能静默截断。"""
    out, _ = compact_messages(conversation(60), words, 400)
    summary = next(m for m in out if str(m["content"]).startswith(SUMMARY_MARKER))
    assert "已省略" in summary["content"]


def test_the_summary_keeps_the_most_recent_actions_when_it_must_elide():
    """防重复要看最近做过什么：刚搜过的词立刻重搜，代价比重搜很久前的更高。"""
    messages = conversation(2)
    for i in range(40):
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [call("search_products", {"query": f"最近{i}"}, f"r{i}")],
        })
        messages.append({"role": "tool", "tool_call_id": f"r{i}", "content": "页面" * 40})
    out, _ = compact_messages(messages, words, 700)
    summary = next(m for m in out if str(m["content"]).startswith(SUMMARY_MARKER))
    body = summary["content"]
    latest_kept = max(i for i in range(40) if f"最近{i}" in body or True)
    assert "最近0" not in body, "最老的动作应该先被挤掉"
    assert any(f"最近{i}" in body for i in range(30, 40)), "较新的动作必须留着"


# --- 放不下的边界 ---------------------------------------------------------


def test_an_unfittable_anchor_raises_rather_than_silently_truncating():
    messages = [{"role": "system", "content": "很长的系统提示" * 100}]
    with pytest.raises(ContextBudgetError, match="固定前缀"):
        compact_messages(messages, words, 50)


def test_an_unfittable_latest_step_raises_with_actionable_advice():
    """删历史已经没用了，错误信息要说清该调什么。"""
    messages = conversation(5, observation="巨大的页面" * 200)
    with pytest.raises(ContextBudgetError, match="max_tokens"):
        compact_messages(messages, words, 300)


def test_a_non_positive_budget_is_rejected():
    with pytest.raises(ValueError):
        compact_messages(conversation(2), words, 0)


# --- 统计 -----------------------------------------------------------------


def test_stats_account_for_every_group():
    messages = conversation(30)
    _, stats = compact_messages(messages, words, 400)
    assert stats.dropped_groups + stats.kept_groups == 30
    assert stats.final_tokens < stats.original_tokens
    assert set(stats.to_dict()) == {
        "original_tokens",
        "final_tokens",
        "dropped_groups",
        "kept_groups",
    }
