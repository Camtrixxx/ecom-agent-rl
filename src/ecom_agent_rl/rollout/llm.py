"""OpenAI 兼容的 chat completions 客户端。

只依赖 requests，不引 openai sdk：我们用到的就是一个带 tools 的 POST，
而 vLLM、教师 API、以及后面 GRPO 侧的 rollout 都是同一个协议。少一个依赖，
少一处版本地雷。

重试策略与参考实现的差别：它只重试传输层异常，5xx 和畸形响应会直接
`KeyError: 'choices'` 打死整条轨迹——而这恰好是 vLLM 过载时最常见的表现。
这里把 5xx、429 和解析失败都算成可重试，并且用带抖动的退避，避免几十个
并发 worker 在同一时刻一起重试。
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8180/v1"
RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
# 与 scripts/serve_model.sh 的 MAX_MODEL_LEN 必须一致：客户端按这个数决定何时压缩
# 历史，服务端按这个数决定何时回 400。两边不一致就是白压或照样撞墙。
DEFAULT_CONTEXT_WINDOW = 24576


class LLMError(RuntimeError):
    """模型调用失败且重试耗尽。"""


class ContextOverflowError(LLMError):
    """prompt 超过模型上下文窗口。

    单独一类，因为它不是「基础设施坏了」而是这一个回合走太远了：长回合把几十个
    observation 累进 messages，35 步的实测外推是 ~44k tokens 对 24576 的窗口。
    当成 infra 失败会让一条长回合掐掉整批（batch.py 遇到 infra 失败就中止），
    而它其实和 max_steps 同类——是这道题的一个结局。
    """


# vLLM / OpenAI 在超上下文时都回 400，靠这些片段认出来。措辞变了会退化成普通
# LLMError（保守方向：中止整批而不是静默把长回合算成失败），测试钉住当前措辞。
_CONTEXT_OVERFLOW_MARKERS = (
    "maximum context length",
    "context_length_exceeded",
    "longer than the maximum model length",
    "reduce the length of the messages",
    "please reduce the length",
)


def _is_context_overflow(body: str) -> bool:
    lowered = body.lower()
    return any(marker in lowered for marker in _CONTEXT_OVERFLOW_MARKERS)


@dataclass
class Usage:
    """累计 token 用量与上下文压缩量，用于估教师采集成本与上下文压力。

    压缩计数也放这里而不是 `ChatClient` 的裸属性：rollout 是多线程的，裸 `+= 1`
    会丢计数，而这里本来就有锁。且压缩是否触发只能从汇总里看到——不报出来就等于
    没法确认压缩在真实链路里生效。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    compactions: int = 0
    dropped_groups: int = 0
    # 压缩前的峰值输入 token。未压缩时它就是最大的一次请求，压缩后这个数会超过
    # 窗口——正是它说明「不压缩就会撞 400」。
    peak_original_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Mapping[str, Any] | None) -> None:
        with self._lock:
            self.calls += 1
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def add_compaction(self, original_tokens: int, dropped_groups: int) -> None:
        """记一次压缩。`dropped_groups == 0` 表示本次未真正丢历史。"""
        with self._lock:
            if original_tokens > self.peak_original_tokens:
                self.peak_original_tokens = original_tokens
            if dropped_groups > 0:
                self.compactions += 1
                self.dropped_groups += dropped_groups

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "compactions": self.compactions,
                "dropped_groups": self.dropped_groups,
                "peak_original_tokens": self.peak_original_tokens,
            }


class ChatClient:
    """线程安全的 chat completions 客户端。

    每个线程各持一个 `requests.Session`：Session 本身不保证线程安全，但连接复用
    对我们很重要——一个回合最多几十次调用，逐次握手的开销不该白付。
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = "ecom-agent",
        api_key: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        retries: int = 4,
        extra_body: Mapping[str, Any] | None = None,
        context_window: int | None = None,
        context_margin: int = 512,
        token_counter: Any | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.extra_body = dict(extra_body or {})
        # context_window 为 None 表示不压缩（单元测试与短回合场景）。设了就必须能数
        # token，否则压缩判断无从下手。
        self.context_window = context_window
        # 安全边际吸收计数器与服务端的残差（实测 1.55%，且偏高）以及模板小改动。
        self.context_margin = context_margin
        self._token_counter = token_counter
        # 压缩计数在 `usage` 里，不在这里另开一个属性：同一个数两个入口日后必然分叉。
        self.usage = Usage()
        self._local = threading.local()

    @property
    def input_budget(self) -> int | None:
        """留给 prompt 的 token 上限：窗口减去要生成的部分再减边际。"""
        if self.context_window is None:
            return None
        return self.context_window - self.max_tokens - self.context_margin

    def _counter(self):
        if self._token_counter is None:
            from .tokens import TokenCounter

            self._token_counter = TokenCounter()
        return self._token_counter

    def _session(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            # 本机服务，别让系统代理把请求转出去。
            session.trust_env = False
            self._local.session = session
        return session

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """返回 assistant message（含 `tool_calls`），失败重试后仍不行则 raise。

        配了 `context_window` 时，超预算的历史会先被压缩（见 `context.py`）：
        长回合的 prompt 会超 24576，不压缩就是 HTTP 400、回合到此为止。
        """
        sent: Sequence[Mapping[str, Any]] = messages
        budget = self.input_budget
        if budget is not None:
            from .context import compact_messages

            sent, stats = compact_messages(
                messages, self._counter().counter_for(tools), budget
            )
            self.usage.add_compaction(stats.original_tokens, stats.dropped_groups)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(sent),
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        payload.update(self.extra_body)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        url = f"{self.base_url}/chat/completions"
        last: str = "no attempt made"
        for attempt in range(self.retries + 1):
            try:
                response = self._session().post(
                    url, json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code in RETRYABLE_STATUS:
                    last = f"HTTP {response.status_code}: {response.text[:200]}"
                    raise _Retry(last)
                if response.status_code >= 400:
                    # 4xx 是请求本身有问题（超上下文、schema 不合法），重试没意义。
                    detail = response.text[:500]
                    if _is_context_overflow(detail):
                        raise ContextOverflowError(
                            f"HTTP {response.status_code}: {detail}"
                        )
                    raise LLMError(f"HTTP {response.status_code}: {detail}")
                body = response.json()
                message = body["choices"][0]["message"]
            except (requests.RequestException, _Retry, ValueError, KeyError, IndexError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt >= self.retries:
                    raise LLMError(f"{url}: 重试 {self.retries} 次仍失败，最后一次 {last}") from exc
                # 指数退避 + 抖动：并发 worker 同时失败时不要同步重试。
                time.sleep(min(2.0 ** attempt, 8.0) * (0.5 + random.random()))
                continue
            self.usage.add(body.get("usage"))
            return dict(message)
        raise LLMError(f"{url}: unreachable, last={last}")


class _Retry(Exception):
    """内部信号：这个响应值得重试。"""
