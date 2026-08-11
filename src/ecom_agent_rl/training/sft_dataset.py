"""把 SFT 样本渲染成 input_ids / labels，loss 只记在 assistant token 上。

三个决定值得写下来，因为它们都是「看起来无所谓、实际会静默毁掉训练」的那类：

## 1. tool_call.arguments 必须转成 dict 再渲染

数据集里 `arguments` 是 OpenAI 标准的 JSON **字符串**，而 Qwen2.5 的 chat template
写的是 `tool_call.arguments | tojson`。字符串进去会被再转义一层：

    传字符串 → {"name": "search_products", "arguments": "{\\"query\\": \\"x\\"}"}
    传 dict   → {"name": "search_products", "arguments": {"query": "x"}}

推理侧不给选择权：vLLM 的 hermes parser 按 `"arguments": {object}` 解析再重新
序列化（见 vllm/tool_parsers/hermes_tool_parser.py）。所以只有 dict 形态的渲染
与推理时的格式一致。用字符串形态训练，学生会学着输出带转义的字符串，parser 拿到
的就是一个 str 而不是 object——工具调用在推理时解析失败，而训练 loss 一路好看。

## 2. 超长样本丢弃，不截断

截断会砍在任意位置，很可能砍掉最后一个 tool_call 的一半，等于教模型输出残缺的
JSON。长回合本来就是这个任务的主体（p90 25776 tokens），宁愿丢样本也不能污染格式。
丢了多少会报出来，不静默。

## 3. 逐回合定位 assistant span，而不是找特殊 token

拿「前缀 + generation prompt」和「含该回合」两次渲染的 token 差当可训练区间：
边界完全由 chat template 决定，不硬编码 `<|im_start|>assistant`。换模型（换模板）
时这套逻辑不用动，而硬编码会在换模型时静默错位——mask 偏一个 token，loss 照样
下降，学出来的东西却是错的。
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

# HuggingFace 的约定：labels 里这个值的位置不计入 loss。
IGNORE_INDEX = -100


@dataclass
class RenderStats:
    """渲染审计。丢弃是有代价的，代价要能看见。"""

    total: int = 0
    kept: int = 0
    dropped_too_long: int = 0
    dropped_bad_arguments: int = 0
    dropped_no_assistant: int = 0
    dropped_boundary: int = 0
    # 保留样本的 token 长度，用来核对 max_length 设得合不合理。
    lengths: list[int] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return (
            self.dropped_too_long
            + self.dropped_bad_arguments
            + self.dropped_no_assistant
            + self.dropped_boundary
        )

    def to_dict(self) -> dict[str, Any]:
        lengths = sorted(self.lengths)
        out: dict[str, Any] = {
            "total": self.total,
            "kept": self.kept,
            "dropped": self.dropped,
            "dropped_too_long": self.dropped_too_long,
            "dropped_bad_arguments": self.dropped_bad_arguments,
            "dropped_no_assistant": self.dropped_no_assistant,
            "dropped_boundary": self.dropped_boundary,
        }
        if lengths:
            out["tokens"] = {
                "mean": round(sum(lengths) / len(lengths), 1),
                "median": lengths[len(lengths) // 2],
                "p90": lengths[min(len(lengths) - 1, int(len(lengths) * 0.9))],
                "max": lengths[-1],
            }
        return out


def normalize_tool_arguments(
    messages: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]] | None:
    """把 `function.arguments` 从 JSON 字符串转成 dict，供 chat template 渲染。

    返回 None 表示这条样本不可训练：`arguments` 不是合法 JSON object 时，渲染出来
    的工具调用格式与推理侧不一致，训了反而有害（见模块 docstring 第 1 条）。

    原始 messages 不改动——数据集文件保持 OpenAI 标准形态，转换只发生在训练期。
    """
    out = deepcopy([dict(m) for m in messages])
    for message in out:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if isinstance(arguments, Mapping):
                continue
            if not isinstance(arguments, str):
                return None
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                # 顶层是数组或标量的 arguments 没法当函数入参，模板也渲染不对。
                return None
            function["arguments"] = parsed
    return out


def _encode(tokenizer: Any, text: str) -> list[int]:
    # add_special_tokens=False：边界由 chat template 给，不要让 tokenizer 再加一层。
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def build_example(
    row: Mapping[str, Any],
    tokenizer: Any,
    *,
    max_length: int = 32768,
    stats: RenderStats | None = None,
) -> dict[str, Any] | None:
    """一条 SFT 样本 → {input_ids, attention_mask, labels}。返回 None 表示丢弃。"""
    stats = stats if stats is not None else RenderStats()

    messages = normalize_tool_arguments(row.get("messages") or [])
    if messages is None:
        stats.dropped_bad_arguments += 1
        return None

    tools = list(row.get("tools") or [])
    assistant_indices = [
        i for i, m in enumerate(messages) if m.get("role") == "assistant"
    ]
    if not assistant_indices:
        stats.dropped_no_assistant += 1
        return None

    def render(upto: int, generation_prompt: bool) -> list[int]:
        text = tokenizer.apply_chat_template(
            messages[:upto],
            tools=tools or None,
            tokenize=False,
            add_generation_prompt=generation_prompt,
        )
        return _encode(tokenizer, text)

    input_ids = render(len(messages), False)
    if len(input_ids) > int(max_length):
        stats.dropped_too_long += 1
        return None

    labels = [IGNORE_INDEX] * len(input_ids)
    for index in assistant_indices:
        prefix_ids = render(index, True)
        through_ids = render(index + 1, False)
        # generation prompt 与真实 assistant 开头可能差一个换行之类，用公共前缀定位
        # 而不是假设 prefix 一定是 through 的前缀。
        start = _common_prefix_length(prefix_ids, through_ids)
        end = len(through_ids)
        # 增量渲染必须真的是整条渲染的前缀。不成立说明模板不是逐条追加的（例如把
        # 工具 schema 塞在最后一条消息里），此时 mask 会错位——丢弃而不是猜。
        if start >= end or end > len(input_ids) or input_ids[:end] != through_ids:
            stats.dropped_boundary += 1
            return None
        labels[start:end] = input_ids[start:end]

    if all(label == IGNORE_INDEX for label in labels):
        stats.dropped_boundary += 1
        return None

    stats.kept += 1
    stats.lengths.append(len(input_ids))
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
        "task_id": row.get("task_id"),
        "trajectory_id": row.get("trajectory_id"),
    }


def load_examples(
    path: Path,
    tokenizer: Any,
    *,
    max_length: int = 32768,
    progress: bool = False,
) -> tuple[list[dict[str, Any]], RenderStats]:
    """读 train.jsonl，渲染成训练样本，同时给出审计。"""
    stats = RenderStats()
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    stats.total = len(rows)

    iterator: Any = rows
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(rows, desc=f"tokenizing {Path(path).name}", unit="样本")
        except ImportError:
            pass

    examples = []
    for row in iterator:
        example = build_example(row, tokenizer, max_length=max_length, stats=stats)
        if example is not None:
            examples.append(example)
    return examples, stats


def collate(
    batch: Sequence[Mapping[str, Any]], *, pad_token_id: int
) -> dict[str, Any]:
    """右侧 padding 到批内最长。

    padding 位的 label 必须是 IGNORE_INDEX 而不是 pad_token_id：后者会让模型去学
    「预测 padding」，在长度参差的批里这部分能占掉相当比例的 token。
    """
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids, attention_mask, labels = [], [], []
    for item in batch:
        pad = width - len(item["input_ids"])
        input_ids.append(list(item["input_ids"]) + [pad_token_id] * pad)
        attention_mask.append(list(item["attention_mask"]) + [0] * pad)
        labels.append(list(item["labels"]) + [IGNORE_INDEX] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
