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


class LLMError(RuntimeError):
    """模型调用失败且重试耗尽。"""


@dataclass
class Usage:
    """累计 token 用量，用于估教师采集成本与上下文压力。"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Mapping[str, Any] | None) -> None:
        with self._lock:
            self.calls += 1
            if usage:
                self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
                self.completion_tokens += int(usage.get("completion_tokens") or 0)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "calls": self.calls,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
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
        self.usage = Usage()
        self._local = threading.local()

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
        """返回 assistant message（含 `tool_calls`），失败重试后仍不行则 raise。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
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
                    # 4xx 多是请求本身有问题（超上下文、schema 不合法），重试没意义。
                    raise LLMError(f"HTTP {response.status_code}: {response.text[:500]}")
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
