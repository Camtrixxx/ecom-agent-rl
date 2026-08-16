"""正文 Hermes 标签的宽容重解析。

这层兜底存在的唯一理由是 vLLM 的 hermes 解析器**全有或全无**：一块坏掉就把整段输出
当 content 返回，合法的第一块跟着陪葬。所以这里最该守住的用例是
`test_valid_first_block_survives_truncated_tail`——「首块合法 + 尾部半块被 1024 截断」
正是实测里 grpo_v2 那 206 条能救回来的轨迹的形状。其余用例都是防止兜底**救过头**：
名字不合法、arguments 不是对象、根本没有标签，都必须原样判负，否则就从「漏记模型行为」
翻到另一边「凭空造出模型没发过的动作」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ecom_agent_rl.rollout.agent import _tool_call_fields
from ecom_agent_rl.rollout.tool_call_recovery import (
    RECOVERED_ID_PREFIX,
    first_valid_call,
    is_recovered,
    iter_blocks,
    recover_tool_calls,
)

VALID = {"search", "click", "buy"}


def block(payload: str) -> str:
    return f"<tool_call>\n{payload}\n</tool_call>"


GOOD = block(json.dumps({"name": "search", "arguments": {"query": "狗狗雨衣"}}))


def test_clean_single_block() -> None:
    calls = recover_tool_calls(GOOD, VALID)
    assert len(calls) == 1
    fn = calls[0]["function"]
    assert fn["name"] == "search"
    # arguments 必须是 JSON 字符串，和真实 vLLM 响应同形状。
    assert isinstance(fn["arguments"], str)
    assert json.loads(fn["arguments"]) == {"query": "狗狗雨衣"}


def test_valid_first_block_survives_truncated_tail() -> None:
    """实测里那 206 条的形状：首块合法，尾巴是被 max_tokens 截断的半块。"""
    text = GOOD + '\n<tool_call>\n{"name": "cli'
    # 前提：vLLM 会因为第二块 loads 失败而整段放弃（第二个分支吃到未闭合的尾巴）。
    blocks = list(iter_blocks(text))
    assert len(blocks) == 2
    assert blocks[1][1] is False
    with pytest.raises(Exception):
        json.loads(blocks[1][0])

    calls = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"


def test_skips_broken_block_and_takes_the_next_good_one() -> None:
    text = block("{这不是 JSON}") + "\n" + GOOD
    calls = recover_tool_calls(text, VALID)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"


def test_only_first_call_is_returned() -> None:
    """agent.py 每轮只执行 calls[0]，后面的块基于同一个旧 observation，不能一起救。"""
    second = block(json.dumps({"name": "buy", "arguments": {}}))
    calls = recover_tool_calls(GOOD + "\n" + second, VALID)
    assert len(calls) == 1
    assert calls[0]["function"]["name"] == "search"


@pytest.mark.parametrize(
    "payload",
    [
        json.dumps({"name": "teleport", "arguments": {}}),          # 环境没有这个工具
        json.dumps({"name": "", "arguments": {}}),                  # 名字为空
        json.dumps({"name": "search", "arguments": "查询"}),         # arguments 不是对象
        json.dumps({"name": "search"}),                             # 缺 arguments
        json.dumps(["search", {}]),                                 # 顶层不是对象
        "{只有半个",
    ],
)
def test_does_not_invent_calls(payload: str) -> None:
    assert recover_tool_calls(block(payload), VALID) == []


def test_no_tags_at_all() -> None:
    assert recover_tool_calls("我觉得这件商品不合适，就不买了。", VALID) == []
    assert recover_tool_calls("", VALID) == []
    assert recover_tool_calls(None, VALID) == []


def test_valid_names_none_skips_the_name_check() -> None:
    text = block(json.dumps({"name": "teleport", "arguments": {}}))
    assert recover_tool_calls(text, None)
    assert recover_tool_calls(text, VALID) == []


def test_recovered_calls_are_auditable() -> None:
    call = recover_tool_calls(GOOD, VALID)[0]
    assert call["id"].startswith(RECOVERED_ID_PREFIX)
    assert is_recovered(call)
    assert not is_recovered({"id": "chatcmpl-tool-abc", "type": "function"})


def test_shape_matches_what_agent_expects() -> None:
    """救回来的调用要能直接喂给 agent.py 的 _tool_call_fields，不然接线那步会炸。"""
    call = recover_tool_calls(GOOD, VALID)[0]
    name, arguments, call_id = _tool_call_fields(call)
    assert name == "search"
    assert arguments == {"query": "狗狗雨衣"}
    assert call_id.startswith(RECOVERED_ID_PREFIX)


def test_first_valid_call_returns_plain_dict() -> None:
    assert first_valid_call(GOOD, VALID) == {
        "name": "search",
        "arguments": {"query": "狗狗雨衣"},
    }


# --- 与实测数对齐 -----------------------------------------------------------
# 上面的用例都是手编形状；这一条把兜底钉在**真实轨迹**上。08-14 14:06 用
# scripts/analyze_tag_recovery.py 量到 grpo_v2 的 256 条标签轨迹里 206 条可恢复。
# 若哪天改了规则让这个数变了，这里会立刻响——而不是等下一次重跑评测才发现。
ROLLOUTS = Path(__file__).resolve().parent.parent / "outputs" / "rollouts"


@pytest.mark.skipif(
    not (ROLLOUTS / "grpo_v2.jsonl").exists(), reason="没有 grpo_v2 轨迹文件"
)
def test_matches_the_measured_recovery_count() -> None:
    from ecom_agent_rl.environment.tools import TOOL_SCHEMAS

    names = {s["function"]["name"] for s in TOOL_SCHEMAS}
    tagged = recovered = 0
    with (ROLLOUTS / "grpo_v2.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            texts = [
                m.get("content") or ""
                for m in (rec.get("messages") or [])
                if m.get("role") == "assistant" and "<tool_call>" in (m.get("content") or "")
            ]
            if not texts:
                continue
            tagged += 1
            if recover_tool_calls("".join(texts), names):
                recovered += 1
    assert tagged == 256, f"标签轨迹数变了：{tagged}"
    assert recovered == 206, f"可恢复数变了：{recovered}"
