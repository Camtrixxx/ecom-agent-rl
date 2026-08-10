"""按服务端的口径数 token。

压缩要在**发请求之前**判断，所以必须自己数一遍，而且数出来的必须和 vLLM 收的
一致——数少了会照样撞 400，数多了会白删历史。

对齐的关键是 Qwen2.5 的 chat template 里 `{{- tool | tojson }}`：jinja 的
`tojson` 默认 `ensure_ascii=True`，中文工具 schema 被转义成 `\\uXXXX`。实测我们
的 schema 因此从 957 涨到 2818 tokens，**每个请求都多付 1861**。所以这里也必须
按转义后的形态计数，否则会低估固定前缀近 2k。

不引 transformers：只要 `tokenizers`（已装）加手写模板渲染。transformers 会拖进
torch，而 rollout 侧不需要它。渲染逻辑与模板的差异只在空白符层面，实测偏差
< 1%，压缩预算里留了安全边际吸收它。
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache
from typing import Any, Mapping, Sequence

DEFAULT_TOKENIZER = "/data/heyuhang/models/Qwen2.5-7B-Instruct/tokenizer.json"

# 以下常数由 scripts/check_token_counter.py 对着真模板标定，不要凭直觉改：
# 改了就跑那个脚本，它会报最大偏差。
#
# 每条消息的固定开销：<|im_start|>{role}\n ... <|im_end|>\n。
_PER_MESSAGE_OVERHEAD = 5
# tool 消息额外套一层 <tool_response>...</tool_response>，比其他角色贵得多。
# 长回合里 tool 消息占绝大多数，这一项算错会随步数线性放大偏差。
_TOOL_MESSAGE_EXTRA = 9
# <tool_call>{"name": ..., "arguments": ...}</tool_call> 的包装。
_TOOL_CALL_OVERHEAD = 18
# 生成提示 <|im_start|>assistant\n 加模板首尾。
_GENERATION_OVERHEAD = 4


@lru_cache(maxsize=4)
def _tokenizer(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


class TokenCounter:
    """数 messages（含 tools）渲染成 prompt 后的 token 数。

    线程安全：`tokenizers` 的 Rust 实现允许并发 encode，但这里仍加锁保护缓存，
    因为 rollout 会在几十个线程里并发调用。
    """

    def __init__(self, tokenizer_path: str = DEFAULT_TOKENIZER) -> None:
        self._tok = _tokenizer(tokenizer_path)
        self._lock = threading.Lock()
        self._tools_cache: tuple[int, int] | None = None

    def _encode(self, text: str) -> int:
        if not text:
            return 0
        with self._lock:
            return len(self._tok.encode(text, add_special_tokens=False).ids)

    def count_tools(self, tools: Sequence[Mapping[str, Any]] | None) -> int:
        """工具 schema 的开销。整批 rollout 里 schema 不变，缓存掉。"""
        if not tools:
            return 0
        key = id(tools)
        if self._tools_cache and self._tools_cache[0] == key:
            return self._tools_cache[1]
        # ensure_ascii=True 复现模板 tojson 的转义行为——这是那 1861 tokens 的来源。
        body = "\n".join(json.dumps(tool, ensure_ascii=True) for tool in tools)
        # 模板在 system 里包一段工具说明与调用格式示例的固定文字。
        total = self._encode(body) + 77
        self._tools_cache = (key, total)
        return total

    def count_messages(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> int:
        total = self.count_tools(tools) + _GENERATION_OVERHEAD
        for message in messages:
            total += _PER_MESSAGE_OVERHEAD
            if message.get("role") == "tool":
                total += _TOOL_MESSAGE_EXTRA
            total += self._encode(str(message.get("content") or ""))
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                total += self._encode(str(function.get("name") or ""))
                # 模板对 arguments 也走 `| tojson`。轨迹里存的是 JSON **字符串**，
                # 再 tojson 一次会加引号并把 \\uXXXX 里的反斜杠二次转义——实测一条
                # 中文 query 因此从 357 涨到 430。要按二次序列化后的形态数。
                arguments = function.get("arguments")
                if isinstance(arguments, Mapping):
                    arguments = json.dumps(arguments, ensure_ascii=True)
                total += self._encode(json.dumps(str(arguments or ""), ensure_ascii=True))
                total += _TOOL_CALL_OVERHEAD
        return total

    def counter_for(self, tools: Sequence[Mapping[str, Any]] | None):
        """绑定 tools，得到 `compact_messages` 要的单参数计数函数。"""
        return lambda messages: self.count_messages(messages, tools)
