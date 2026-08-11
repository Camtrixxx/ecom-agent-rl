"""SFT 渲染的测试。

重点钉四件事：tool_call 的 arguments 渲染成 object 而不是转义字符串（这条错了推理
时工具调用会解析失败，而训练 loss 一路好看）、loss 只落在 assistant token 上、
padding 位不参与 loss、以及模板边界对不上时丢弃而不是猜。

用字符级的假 tokenizer 跑结构性断言（快、且前缀关系精确），另有几条用真的 Qwen2.5
tokenizer 与真模板验端到端——假模板再像也证明不了真模板的行为。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ecom_agent_rl.training.sft_dataset import (
    IGNORE_INDEX,
    RenderStats,
    build_example,
    collate,
    load_examples,
    normalize_tool_arguments,
)

QWEN = Path("/data/heyuhang/models/Qwen2.5-7B-Instruct")


class FakeTokenizer:
    """字符级 tokenizer + 一个模仿 Qwen2.5 关键行为的模板。

    模板照抄真模板的两个要点：assistant 的 tool_call 走 `arguments | tojson`
    （所以字符串会被再转义一层），以及 tools 渲染进 system 段。
    """

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        parts = []
        if tools:
            parts.append(f"<sys>tools={json.dumps(tools, ensure_ascii=False)}</sys>")
        for message in messages:
            role = message["role"]
            if role == "assistant":
                body = message.get("content") or ""
                for call in message.get("tool_calls") or []:
                    function = call["function"]
                    # 与真模板一致：arguments 走 tojson。
                    body += (
                        f'<tool_call>{{"name": "{function["name"]}", '
                        f'"arguments": {json.dumps(function["arguments"], ensure_ascii=False)}}}'
                        "</tool_call>"
                    )
                parts.append(f"<a>{body}<end>")
            else:
                parts.append(f"<{role}>{message.get('content') or ''}<end>")
        if add_generation_prompt:
            parts.append("<a>")
        return "".join(parts)


def row(
    *,
    arguments: Any = '{"query": "x"}',
    task_id: int = 1,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "trajectory_id": f"t{task_id}",
        "task_id": task_id,
        "messages": [
            {"role": "system", "content": "你是购物助手"},
            {"role": "user", "content": "帮我买个枕头"},
            {"role": "assistant", "content": "先搜索", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search_products", "arguments": arguments}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "【搜索结果】第1页"},
            {"role": "assistant", "content": "买它", "tool_calls": [
                {"id": "c2", "type": "function",
                 "function": {"name": "buy_now", "arguments": "{}"}}]},
        ],
        "tools": tools if tools is not None else [
            {"type": "function", "function": {"name": "search_products"}}],
    }


def decode(tokenizer: Any, ids: list[int]) -> str:
    return "".join(chr(i) for i in ids)


# --- arguments 的形态 ---------------------------------------------------------

def test_string_arguments_are_parsed_into_an_object():
    """数据集存 OpenAI 标准的 JSON 字符串，模板要的是 object。"""
    out = normalize_tool_arguments(row()["messages"])
    assert out is not None
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}


def test_already_parsed_arguments_pass_through():
    out = normalize_tool_arguments(row(arguments={"query": "x"})["messages"])
    assert out is not None
    assert out[2]["tool_calls"][0]["function"]["arguments"] == {"query": "x"}


def test_normalizing_does_not_mutate_the_dataset_row():
    """数据集文件保持 OpenAI 形态，转换只发生在训练期。"""
    original = row()
    normalize_tool_arguments(original["messages"])
    assert original["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'


@pytest.mark.parametrize("bad", ['{"query": ', "[1, 2]", '"just a string"', "42", None, 7])
def test_arguments_that_are_not_a_json_object_make_the_sample_unusable(bad: Any):
    """渲染不出与推理一致的格式，就不该拿它训练。"""
    assert normalize_tool_arguments(row(arguments=bad)["messages"]) is None


def test_a_sample_with_broken_arguments_is_dropped_and_counted():
    stats = RenderStats()
    assert build_example(row(arguments="{oops"), FakeTokenizer(), stats=stats) is None
    assert stats.dropped_bad_arguments == 1
    assert stats.kept == 0


def test_the_rendered_tool_call_carries_an_object_not_an_escaped_string():
    """这条是本模块存在的理由：转义字符串会让推理侧的 parser 拿到 str。"""
    example = build_example(row(), FakeTokenizer())
    assert example is not None
    text = decode(FakeTokenizer(), example["input_ids"])
    assert '"arguments": {"query": "x"}' in text
    assert '\\"query\\"' not in text


# --- loss 只落在 assistant 上 -------------------------------------------------

def test_only_assistant_tokens_are_trainable():
    tokenizer = FakeTokenizer()
    example = build_example(row(), tokenizer)
    assert example is not None
    trainable = decode(tokenizer, [
        t for t, l in zip(example["input_ids"], example["labels"]) if l != IGNORE_INDEX
    ])
    # 用户指令与环境回复是上下文，不该被学。
    assert "帮我买个枕头" not in trainable
    assert "【搜索结果】第1页" not in trainable
    assert "你是购物助手" not in trainable
    # 两个 assistant 回合的正文与工具调用都要在。
    assert "先搜索" in trainable and "买它" in trainable
    assert "search_products" in trainable and "buy_now" in trainable


def trainable_spans(example: dict[str, Any]) -> list[list[int]]:
    """按连续可训练区间切开，用于逐字比对边界。"""
    spans: list[list[int]] = []
    current: list[int] = []
    for token, label in zip(example["input_ids"], example["labels"]):
        if label != IGNORE_INDEX:
            current.append(token)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return spans


def test_each_trainable_span_starts_exactly_at_the_assistant_content():
    """逐字钉边界，不只钉「内容在里面」。

    mask 偏一个 token 是这里最阴的 bug：多学一个 `>`、少学一个 `<end>`，loss 照样
    平稳下降，只有推理时才暴露（学不会停、或多吐一个符号）。所以必须精确相等，
    含 in 判断的测试抓不住它——实测偏移 1 个 token 时那些断言全过。
    """
    tokenizer = FakeTokenizer()
    spans = trainable_spans(build_example(row(), tokenizer))
    assert [decode(tokenizer, s) for s in spans] == [
        '先搜索<tool_call>{"name": "search_products", "arguments": {"query": "x"}}</tool_call><end>',
        '买它<tool_call>{"name": "buy_now", "arguments": {}}</tool_call><end>',
    ]


def test_labels_match_input_ids_wherever_they_are_trainable():
    """可训练位置的 label 必须等于该位置的 token，不能错位。"""
    example = build_example(row(), FakeTokenizer())
    assert example is not None
    for token, label in zip(example["input_ids"], example["labels"]):
        assert label in (IGNORE_INDEX, token)


def test_there_is_one_trainable_span_per_assistant_turn():
    assert len(trainable_spans(build_example(row(), FakeTokenizer()))) == 2


def test_a_conversation_without_an_assistant_turn_is_dropped():
    sample = row()
    sample["messages"] = sample["messages"][:2]
    stats = RenderStats()
    assert build_example(sample, FakeTokenizer(), stats=stats) is None
    assert stats.dropped_no_assistant == 1


def test_shapes_line_up():
    example = build_example(row(), FakeTokenizer())
    assert example is not None
    assert len(example["input_ids"]) == len(example["labels"]) == len(example["attention_mask"])
    assert set(example["attention_mask"]) == {1}


def test_the_sample_keeps_its_lineage():
    example = build_example(row(task_id=42), FakeTokenizer())
    assert example is not None
    assert example["task_id"] == 42
    assert example["trajectory_id"] == "t42"


# --- 超长与边界 ---------------------------------------------------------------

def test_an_oversized_sample_is_dropped_rather_than_truncated():
    """截断会砍掉半个 tool_call，等于教模型输出残缺 JSON。"""
    stats = RenderStats()
    assert build_example(row(), FakeTokenizer(), max_length=10, stats=stats) is None
    assert stats.dropped_too_long == 1


def test_a_sample_at_the_limit_is_kept():
    tokenizer = FakeTokenizer()
    exact = len(build_example(row(), tokenizer)["input_ids"])
    assert build_example(row(), tokenizer, max_length=exact) is not None
    assert build_example(row(), tokenizer, max_length=exact - 1) is None


def test_a_template_that_is_not_incrementally_prefixed_is_rejected():
    """模板若把内容塞在末尾（增量渲染不是整条渲染的前缀），mask 会错位——丢弃。"""

    class TrailingTokenTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tools=None, tokenize=False,
                                add_generation_prompt=False):
            text = super().apply_chat_template(
                messages, tools=tools, add_generation_prompt=add_generation_prompt)
            # 只在完整渲染时追加尾巴，让前缀关系断掉。
            return text + ("<tail>" if not add_generation_prompt else "")

    stats = RenderStats()
    assert build_example(row(), TrailingTokenTokenizer(), stats=stats) is None
    assert stats.dropped_boundary == 1


def test_the_boundary_is_found_even_when_the_generation_prompt_differs_slightly():
    """generation prompt 与真实 assistant 开头差个换行是常见的，不该因此丢样本。"""

    class NewlineTokenizer(FakeTokenizer):
        def apply_chat_template(self, messages, tools=None, tokenize=False,
                                add_generation_prompt=False):
            text = super().apply_chat_template(
                messages, tools=tools, add_generation_prompt=add_generation_prompt)
            return text + ("\n" if add_generation_prompt else "")

    assert build_example(row(), NewlineTokenizer()) is not None


# --- padding ------------------------------------------------------------------

def test_padding_is_masked_out_of_the_loss():
    """padding 位若给 pad_token_id 当 label，模型会去学「预测 padding」。"""
    torch = pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    short = build_example(row(), tokenizer)
    long = build_example(row(task_id=2), tokenizer)
    long["input_ids"] = long["input_ids"] + [65] * 20
    long["attention_mask"] = long["attention_mask"] + [1] * 20
    long["labels"] = long["labels"] + [IGNORE_INDEX] * 20

    batch = collate([short, long], pad_token_id=151643)
    width = len(long["input_ids"])
    assert batch["input_ids"].shape == (2, width)
    padded = width - len(short["input_ids"])
    assert torch.all(batch["labels"][0, -padded:] == IGNORE_INDEX)
    assert torch.all(batch["attention_mask"][0, -padded:] == 0)
    assert torch.all(batch["input_ids"][0, -padded:] == 151643)


def test_collate_leaves_real_tokens_attended():
    pytest.importorskip("torch")
    tokenizer = FakeTokenizer()
    example = build_example(row(), tokenizer)
    batch = collate([example], pad_token_id=0)
    assert batch["attention_mask"].sum().item() == len(example["input_ids"])


# --- 批量读取与审计 -----------------------------------------------------------

def test_load_examples_reports_what_it_dropped(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in [
            row(task_id=1),
            row(task_id=2, arguments="{broken"),
            row(task_id=3),
        ]) + "\n",
        encoding="utf-8",
    )
    examples, stats = load_examples(path, FakeTokenizer(), max_length=100000)
    assert len(examples) == 2
    assert stats.total == 3 and stats.kept == 2
    assert stats.dropped == 1 and stats.dropped_bad_arguments == 1
    assert stats.to_dict()["tokens"]["max"] > 0


def test_the_report_survives_an_empty_batch():
    assert RenderStats().to_dict()["kept"] == 0


# --- 真 tokenizer 与真模板 ----------------------------------------------------

@pytest.mark.skipif(not QWEN.exists(), reason="需要本地 Qwen2.5-7B-Instruct")
class TestAgainstTheRealTemplate:
    """假模板证明不了真模板的行为，这几条用真的。"""

    @classmethod
    @pytest.fixture(scope="class")
    def tokenizer(cls):
        transformers = pytest.importorskip("transformers")
        return transformers.AutoTokenizer.from_pretrained(str(QWEN))

    def test_the_real_template_escapes_string_arguments(self, tokenizer):
        """这就是必须转 dict 的原因：真模板的 `| tojson` 会把字符串再转义一层。"""
        messages = [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "c1", "type": "function", "function": {
                            "name": "search_products", "arguments": '{"query": "x"}'}}]}]
        escaped = tokenizer.apply_chat_template(messages, tokenize=False)
        assert '\\"query\\"' in escaped

        clean = tokenizer.apply_chat_template(
            normalize_tool_arguments(messages), tokenize=False)
        assert '"arguments": {"query": "x"}' in clean
        assert '\\"query\\"' not in clean

    def test_masking_covers_exactly_the_assistant_turns(self, tokenizer):
        example = build_example(row(), tokenizer, max_length=32768)
        assert example is not None
        # 真模板下也逐字钉：每段必须从 assistant 正文开始、到 im_end 结束，
        # 不含 `<|im_start|>assistant` 那段 header（那是 prompt，不是要学的内容）。
        spans = [tokenizer.decode(s) for s in trainable_spans(example)]
        assert spans == [
            '先搜索\n<tool_call>\n{"name": "search_products", '
            '"arguments": {"query": "x"}}\n</tool_call><|im_end|>\n',
            '买它\n<tool_call>\n{"name": "buy_now", "arguments": {}}\n</tool_call><|im_end|>\n',
        ]

    def test_the_tool_schemas_land_in_the_masked_context(self, tokenizer):
        """工具 schema 是上下文，不是要学的内容。"""
        tools = [{"type": "function", "function": {
            "name": "search_products", "description": "搜索商品",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}}}]
        example = build_example(row(tools=tools), tokenizer, max_length=32768)
        assert example is not None
        masked = tokenizer.decode(
            [t for t, l in zip(example["input_ids"], example["labels"]) if l == IGNORE_INDEX])
        assert "搜索商品" in masked
