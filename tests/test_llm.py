"""ChatClient 的重试与错误分类。

之前没有测试，而这里的分类直接决定「一条长回合」和「模型服务挂了」谁该中止整批：
超上下文必须是 ContextOverflowError（只结束这一个回合），其余 4xx 才是 LLMError。

不起真服务：把 requests.Session.post 换掉，断言重试次数与抛出的异常类型。
"""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest

from ecom_agent_rl.rollout.llm import (
    DEFAULT_CONTEXT_WINDOW,
    RETRYABLE_STATUS,
    ChatClient,
    ContextOverflowError,
    LLMError,
    Usage,
    _is_context_overflow,
)


class FakeResponse:
    def __init__(self, status_code: int, body: Any = None, text: str | None = None) -> None:
        self.status_code = status_code
        self._body = body
        self.text = text if text is not None else json.dumps(body or {})

    def json(self) -> Any:
        if self._body is None:
            raise ValueError("响应不是合法 JSON")
        return self._body


def ok_body(content: str = "hi") -> dict[str, Any]:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
    }


def client(monkeypatch, responses: list[Any], **kwargs) -> tuple[ChatClient, list]:
    """把 post 换成按序返回 responses；元素是异常则抛出。"""
    calls: list[dict[str, Any]] = []
    queue = list(responses)

    def fake_post(self, url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "payload": json, "headers": headers})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("requests.Session.post", fake_post)
    # 退避睡眠在测试里没意义，去掉。
    monkeypatch.setattr("time.sleep", lambda *_: None)
    return ChatClient(retries=3, **kwargs), calls


# --- 超上下文的识别（最关键的一条） ---------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        "This model's maximum context length is 24576 tokens, however you requested 44458",
        '{"error": {"code": "context_length_exceeded"}}',
        "The prompt is longer than the maximum model length of 24576",
        "Please reduce the length of the messages.",
        "PLEASE REDUCE THE LENGTH",  # 大小写不该影响判定
    ],
)
def test_context_overflow_messages_are_recognised(body: str):
    assert _is_context_overflow(body)


@pytest.mark.parametrize(
    "body",
    [
        "invalid tool schema: additionalProperties",
        '{"error": "model not found"}',
        "unauthorized",
        "",
    ],
)
def test_other_4xx_bodies_are_not_treated_as_context_overflow(body: str):
    assert not _is_context_overflow(body)


def test_a_context_overflow_400_raises_the_dedicated_error(monkeypatch):
    """必须是 ContextOverflowError：LLMError 会被当成 infra 失败中止整批。"""
    c, calls = client(monkeypatch, [
        FakeResponse(400, text="maximum context length is 24576 tokens")
    ])
    with pytest.raises(ContextOverflowError):
        c.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 1, "4xx 不该重试"


def test_context_overflow_is_a_subclass_of_llm_error(monkeypatch):
    """调用方按 LLMError 兜底时仍要能接住，只是 agent 里先匹配更具体的那个。"""
    assert issubclass(ContextOverflowError, LLMError)


def test_an_unrecognised_400_is_a_plain_llm_error(monkeypatch):
    """措辞变了宁可保守：中止整批，而不是静默把长回合算成模型失败。"""
    c, _ = client(monkeypatch, [FakeResponse(400, text="tool schema invalid")])
    with pytest.raises(LLMError) as info:
        c.complete([{"role": "user", "content": "hi"}])
    assert not isinstance(info.value, ContextOverflowError)


# --- 重试 -----------------------------------------------------------------


@pytest.mark.parametrize("status", sorted(RETRYABLE_STATUS))
def test_retryable_statuses_are_retried_then_succeed(monkeypatch, status: int):
    """vLLM 过载时最常见的就是 5xx；参考实现在这里直接打死整条轨迹。"""
    c, calls = client(monkeypatch, [
        FakeResponse(status, text="overloaded"),
        FakeResponse(200, ok_body()),
    ])
    message = c.complete([{"role": "user", "content": "hi"}])
    assert message["content"] == "hi"
    assert len(calls) == 2


def test_retries_are_bounded_and_then_raise(monkeypatch):
    c, calls = client(monkeypatch, [FakeResponse(503, text="down")] * 4)
    with pytest.raises(LLMError, match="重试"):
        c.complete([{"role": "user", "content": "hi"}])
    assert len(calls) == 4, "retries=3 意味着首次 + 3 次重试"


def test_a_malformed_json_body_is_retried_not_fatal(monkeypatch):
    """参考实现在这里抛 KeyError: 'choices'，一条轨迹就废了。"""
    c, calls = client(monkeypatch, [
        FakeResponse(200, None, text="<html>502 bad gateway</html>"),
        FakeResponse(200, ok_body("recovered")),
    ])
    assert c.complete([{"role": "user", "content": "x"}])["content"] == "recovered"
    assert len(calls) == 2


def test_a_body_without_choices_is_retried(monkeypatch):
    c, calls = client(monkeypatch, [
        FakeResponse(200, {"usage": {}}),
        FakeResponse(200, ok_body("recovered")),
    ])
    assert c.complete([{"role": "user", "content": "x"}])["content"] == "recovered"
    assert len(calls) == 2


def test_transport_errors_are_retried(monkeypatch):
    import requests

    c, calls = client(monkeypatch, [
        requests.ConnectionError("connection refused"),
        FakeResponse(200, ok_body("recovered")),
    ])
    assert c.complete([{"role": "user", "content": "x"}])["content"] == "recovered"
    assert len(calls) == 2


# --- 请求构造与用量 --------------------------------------------------------


def test_tools_are_sent_only_when_provided(monkeypatch):
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body()), FakeResponse(200, ok_body())])
    c.complete([{"role": "user", "content": "x"}])
    assert "tools" not in calls[0]["payload"]
    c.complete([{"role": "user", "content": "x"}], [{"type": "function"}])
    assert calls[1]["payload"]["tools"] == [{"type": "function"}]


def test_the_api_key_goes_in_the_header_and_nowhere_else(monkeypatch):
    """教师采集用真 key，而汇总和轨迹都要写盘、日志会留存。

    key 只该出现在 Authorization 头里。谁要是为了调试把 payload 或 usage dump
    出来，这个断言会先失败。
    """
    secret = "sk-do-not-leak-me"
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body())], api_key=secret)
    c.complete([{"role": "user", "content": "x"}], [{"type": "function"}])
    call = calls[0]
    assert call["headers"]["Authorization"] == f"Bearer {secret}"
    assert secret not in json.dumps(call["payload"], ensure_ascii=False)
    assert secret not in json.dumps(c.usage.snapshot(), ensure_ascii=False)


def test_no_authorization_header_without_a_key(monkeypatch):
    """本地 vLLM 不需要 key，别发一个空 Bearer 让服务端困惑。"""
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body())])
    c.complete([{"role": "user", "content": "x"}])
    assert "Authorization" not in calls[0]["headers"]


def test_sampling_parameters_reach_the_payload(monkeypatch):
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body())],
                      temperature=0.3, top_p=0.8, max_tokens=256, model="m")
    c.complete([{"role": "user", "content": "x"}])
    payload = calls[0]["payload"]
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 256
    assert payload["model"] == "m"


def test_usage_accumulates_across_calls(monkeypatch):
    c, _ = client(monkeypatch, [FakeResponse(200, ok_body())] * 3)
    for _ in range(3):
        c.complete([{"role": "user", "content": "x"}])
    snapshot = c.usage.snapshot()
    assert snapshot["calls"] == 3
    assert snapshot["prompt_tokens"] == 30
    assert snapshot["completion_tokens"] == 9


def test_usage_counts_a_call_even_without_a_usage_block():
    usage = Usage()
    usage.add(None)
    snapshot = usage.snapshot()
    # 只断言这三项，不锁死整个字典：snapshot 会随观测项增加而扩展。
    assert snapshot["calls"] == 1
    assert snapshot["prompt_tokens"] == 0
    assert snapshot["completion_tokens"] == 0


# --- 上下文压缩接入 --------------------------------------------------------


class CountingCounter:
    """按内容字符数计数，测试里可预测。"""

    def counter_for(self, tools):
        def count(messages):
            total = 0
            for m in messages:
                total += len(str(m.get("content") or ""))
                for c in m.get("tool_calls") or []:
                    total += len(str(c["function"]["arguments"]))
            return total

        return count


def long_conversation(steps: int) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]
    for i in range(steps):
        messages.append({
            "role": "assistant", "content": "",
            "tool_calls": [{"id": f"c{i}", "type": "function",
                            "function": {"name": "search_products",
                                         "arguments": '{"query": "q%d"}' % i}}],
        })
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": "页" * 200})
    return messages


def test_without_a_context_window_nothing_is_compacted(monkeypatch):
    """默认行为不变：短回合和单元测试不该被压缩逻辑影响。"""
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body())])
    messages = long_conversation(20)
    c.complete(messages)
    assert len(calls[0]["payload"]["messages"]) == len(messages)
    assert c.usage.snapshot()["compactions"] == 0


def test_an_oversized_prompt_is_compacted_before_being_sent(monkeypatch):
    """不压就是 HTTP 400、回合到此为止。"""
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body())],
                      context_window=2000, max_tokens=200, context_margin=50,
                      token_counter=CountingCounter())
    messages = long_conversation(30)
    c.complete(messages)
    sent = calls[0]["payload"]["messages"]
    assert len(sent) < len(messages)
    snapshot = c.usage.snapshot()
    assert snapshot["compactions"] == 1
    assert snapshot["dropped_groups"] > 0
    # 压缩前的峰值必须超预算，否则这个用例根本没触发到压缩路径。
    assert snapshot["peak_original_tokens"] > c.input_budget


def test_compaction_does_not_mutate_the_caller_s_messages(monkeypatch):
    """trajectory.messages 是要写盘的训练数据，必须保持完整。"""
    c, _ = client(monkeypatch, [FakeResponse(200, ok_body())],
                  context_window=2000, max_tokens=200, context_margin=50,
                  token_counter=CountingCounter())
    messages = long_conversation(30)
    before = json.dumps(messages, ensure_ascii=False)
    c.complete(messages)
    assert json.dumps(messages, ensure_ascii=False) == before


def test_a_prompt_within_budget_is_sent_verbatim(monkeypatch):
    c, calls = client(monkeypatch, [FakeResponse(200, ok_body())],
                      context_window=100_000, max_tokens=200,
                      token_counter=CountingCounter())
    messages = long_conversation(5)
    c.complete(messages)
    assert calls[0]["payload"]["messages"] == messages
    snapshot = c.usage.snapshot()
    assert snapshot["compactions"] == 0
    # 没丢历史，但峰值仍被记下——用来判断离撞窗口还有多远。
    assert snapshot["peak_original_tokens"] > 0


def test_compaction_counts_add_up_under_concurrent_rollout():
    """rollout 多线程共用一个 client，压缩计数必须是累加而非覆盖。

    注意这个用例**不能**证明线程安全：CPython 下 `+= 1` 的竞争窗口极窄，无锁实现
    在这里大概率同样通过（已实测）。它钉住的是累加语义和 snapshot 的自洽——真正
    的线程安全靠 `Usage` 里的锁，靠 review 保证，不靠这个断言。
    """
    usage = Usage()
    threads = [
        threading.Thread(target=lambda: [usage.add_compaction(100, 1) for _ in range(200)])
        for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    snapshot = usage.snapshot()
    assert snapshot["compactions"] == 8 * 200
    assert snapshot["dropped_groups"] == 8 * 200


def test_a_no_op_compaction_records_the_peak_but_not_a_compaction():
    """未丢历史时不该计数，否则「压缩了几次」永远等于请求数、失去意义。"""
    usage = Usage()
    usage.add_compaction(1234, 0)
    snapshot = usage.snapshot()
    assert snapshot["compactions"] == 0
    assert snapshot["dropped_groups"] == 0
    assert snapshot["peak_original_tokens"] == 1234


def test_the_peak_keeps_the_largest_not_the_latest():
    usage = Usage()
    usage.add_compaction(9000, 1)
    usage.add_compaction(3000, 1)
    assert usage.snapshot()["peak_original_tokens"] == 9000


def test_the_input_budget_leaves_room_for_generation_and_margin():
    c = ChatClient(context_window=24576, max_tokens=1024, context_margin=512)
    assert c.input_budget == 24576 - 1024 - 512


def test_no_context_window_means_no_budget():
    assert ChatClient().input_budget is None


def test_the_client_window_matches_the_serve_script():
    """两边不一致就是白压缩或照样撞 400，而且不会有任何报错提示。"""
    import re
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "serve_model.sh"
    match = re.search(r'MAX_MODEL_LEN="\$\{MAX_MODEL_LEN:-(\d+)\}"', script.read_text())
    assert match, "serve_model.sh 里找不到 MAX_MODEL_LEN 默认值"
    assert int(match.group(1)) == DEFAULT_CONTEXT_WINDOW
