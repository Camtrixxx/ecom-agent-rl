"""从正文里的 Hermes 标签救回一次工具调用。

**这个模块现在还没有被任何地方 import**，是故意的：接线要等 D2 链条跑完（改解析口径
会让链条前后两半的臂不可比）。先落地 + 测好，接线那一步就只剩三行。

## 为什么需要它

08-14 14:06 读了装在 .venv 里的 vLLM 源码（`vllm/tool_parsers/hermes_tool_parser.py`
第 87-120 行），`extract_tool_calls` 是**全有或全无**的：

    raw_function_calls = [json.loads(match[0] if match[0] else match[1])
                          for match in function_call_tuples]
    ...
    except Exception:
        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output)

一个列表推导式里 loads 所有块，任何一块坏掉 → 整个 except → `tool_calls=[]`，把**整段
输出**当 content 返回。于是「第一块完全合法、尾巴上多了半块被 max_tokens=1024 截断的
垃圾」这种输出，合法的第一块也一起丢。

而 `agent.py:233-238` 拿不到 tool_calls 就立刻 `status = NO_TOOL_CALL` 并 return。
两段一拼：**基础设施的解析损耗被记成了模型的决策**。这和 finish_reason 那个洞（任务
 #21）是同一种错，只是换了一个轴。

实测（`scripts/analyze_tag_recovery.py`，08-14 14:06）：grpo_v2 的 272 条 no_tool_call
里 256 条正文带 `<tool_call>` 标签，其中 **206 条（80.5%）**按下面的规则能取到一次合法
调用。sft / grpo(v1) / sft_v2 三个对照臂可恢复数是 **0**——这层损耗是 v2 特有的。

## 为什么在 agent.py 接线，不在 llm.py

任务 #23 原来记的是「给 llm.py 加宽容重解析兜底」，**层次记错了**：

  - `llm.py` 是传输层，它的职责是如实返回服务端说了什么。它 264-266 行那句注释
    （「只有 content、没有 tool_calls 是模型真实的决策」）对传输层依然成立。
  - 错的是**解释**：`agent.py` 把「没有 tool_calls」直接等同于「模型选择不动手」。
    正文里带着 Hermes 标签时，这个等式不成立。
  - 而且 `agent.py` 本来就 import 了 `..environment.tools`，名字合法性校验不需要新
    引一条跨层依赖；放在 llm.py 里则要么漏掉校验，要么让传输层认识业务 schema。

接线点是 `agent.py:231-233` 之间：

    calls = list(assistant.get("tool_calls") or [])
    if not calls:
        recovered = recover_tool_calls(assistant.get("content"), VALID_TOOL_NAMES)
        if recovered:
            calls = recovered
    if not calls:
        ...原样保留...

## 只救第一块，不救全部

`agent.py:240-249` 每轮只执行 `calls[0]`，后面的调用是基于同一个旧 observation 生成的。
所以这里返回**单元素列表**——恢复一次有把握的调用即可。把后面那些块也塞进去是猜测：
实测复读峰的众数块占比中位 0.444、去重率中位 0.167，那些块是残缺的偏 JSON，不是模型
想发的第二个动作。

## 恢复出来的调用带可审计的 id

`id` 一律形如 `recovered-<hex>`，前缀是 `RECOVERED_ID_PREFIX`。这样事后直接在轨迹里
数这个前缀就能知道有多少次调用是救回来的，不必再解析一遍正文。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Collection, Iterator

# 逐字抄 vLLM 的正则（`vllm/tool_parsers/hermes_tool_parser.py:38-40`）。它用 `regex`
# 包、我们用 `re`；对这个模式两者行为一致（没有可变长度回顾、没有递归）。
# 第二个分支 `<tool_call>(.*)` 专门吃「开标签在、闭标签被截断」的最后一块。
TOOL_CALL_REGEX = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)

TOOL_CALL_START = "<tool_call>"
RECOVERED_ID_PREFIX = "recovered-"


def iter_blocks(text: str) -> Iterator[tuple[str, bool]]:
    """→ (块内容, 是否闭合)，顺序与 vLLM 的 findall 一致。"""
    for match in TOOL_CALL_REGEX.finditer(text):
        if match.group(1) is not None:
            yield match.group(1), True
        else:
            yield match.group(2), False


def first_valid_call(
    text: str, valid_names: Collection[str] | None = None
) -> dict[str, Any] | None:
    """按顺序返回第一个「能 loads、名字合法、arguments 是对象」的块，坏块跳过。

    `valid_names=None` 表示不校验名字。传进来时会校验——救回一个环境根本没有的工具名
    毫无意义，那种块只会在下一步被 `tools.check` 拒掉，白占一次拒绝额度。
    """
    for body, _closed in iter_blocks(text):
        try:
            obj = json.loads(body)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        arguments = obj.get("arguments")
        if not isinstance(name, str) or not name:
            continue
        if valid_names is not None and name not in valid_names:
            continue
        if not isinstance(arguments, dict):
            continue
        return {"name": name, "arguments": arguments}
    return None


def recover_tool_calls(
    content: str | None, valid_names: Collection[str] | None = None
) -> list[dict[str, Any]]:
    """正文里若藏着一次合法的 Hermes 工具调用，按 OpenAI 的 tool_calls 形状返回它。

    返回空列表 = 没救回来（包括 content 为空、根本没有标签、所有块都坏）。调用方可以
    直接 `if recovered: calls = recovered`。

    `arguments` 序列化成 **JSON 字符串**而不是留成 dict：真实的 vLLM 响应就是字符串，
    这条消息之后要写进 `trajectory.messages`、进 SFT 样本、还会被原样回传给 API，形状
    不一致会在下游某处炸。`_tool_call_fields` 两种都收，但那不是留下不一致的理由。
    """
    if not content or TOOL_CALL_START not in content:
        return []
    call = first_valid_call(content, valid_names)
    if call is None:
        return []
    return [
        {
            "id": f"{RECOVERED_ID_PREFIX}{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": call["name"],
                "arguments": json.dumps(call["arguments"], ensure_ascii=False),
            },
        }
    ]


def is_recovered(call: Any) -> bool:
    """这次调用是不是救回来的——事后统计用。"""
    return (
        isinstance(call, dict)
        and str(call.get("id") or "").startswith(RECOVERED_ID_PREFIX)
    )
