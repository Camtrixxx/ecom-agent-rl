"""上下文窗口管理：把长回合塞进模型窗口，而不牺牲「已经看过什么」。

## 为什么必须做这件事

实测 448 个真实 observation：搜索结果页均值 2406 tokens（p90 2676），占全部
observation 开销的 91%；固定前缀（system + 转义后的工具 schema）3494 tokens。
按均值外推，35 步回合需要 ~44k tokens，而服务端 `MAX_MODEL_LEN` 是 24576 ——
**第 18 步就会撞墙**。撞墙的表现是 HTTP 400，即 `ContextOverflowError`：
回合到此为止，前面的步数白跑。

窗口开到模型上限 32768 也只是把第 18 步推到第 25 步，仍然盖不住 35 步。所以
压缩不是优化，是跑满 `SHOP_MAX_STEPS=35` 的前提。

## 与参考实现的差别（关键的一处）

参考实现 `environment/context.py` 的 `compact_chat_messages` 做二分查找，保留
能放进预算的**最新若干个完整组**，其余整组删除。方向对，但对这个任务有害：

删掉旧组等于删掉「我已经搜过哪些词、翻过哪几页」的全部记录。而 Reward v3 里
`repeat_loop` 是 **−0.65**，是除买错（−0.85）以外最重的惩罚——专门罚重复动作。
一个看不见自己历史的模型必然重复，于是压缩本身就在制造它要避免的失败。

这里的做法：删掉旧组的**正文**，但把每个被删组压成一行摘要留在上下文里
（`_summarise_group`），形如

    [已省略 12 步] 搜过: 铝合金雕花板 / 白色外墙板(第2页) / …; 看过: 852514426157, …

摘要是有界的（`_MAX_SUMMARY_ITEMS`），所以长回合的摘要不会重新长回去。代价是
模型看不到旧页面的商品明细——这是主动取舍：明细可以重新搜出来，而「已经试过
什么」一旦丢失就只能靠重复动作去重新发现。

## 边界

`anchor`（system + 首个 user）永不删除。若 anchor 加最新一组都放不下，抛
`ContextBudgetError` ——这时该调窗口或 `max_tokens`，不是继续删历史。
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Sequence

# 摘要里每类最多列几项。够模型判断「这个词搜过了」，又不会让摘要随回合线性增长。
_MAX_SUMMARY_ITEMS = 12
# 摘要行的前缀，同时作为识别「这是摘要而非真实 observation」的标记。
SUMMARY_MARKER = "[已省略"


class ContextBudgetError(RuntimeError):
    """固定前缀加最新一步都放不进预算——删历史已经无济于事。"""


@dataclass(frozen=True)
class CompactionStats:
    original_tokens: int
    final_tokens: int
    dropped_groups: int
    kept_groups: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


TokenCounter = Callable[[Sequence[Mapping[str, Any]]], int]


def _split_groups(
    messages: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
    """切成 anchor 与「assistant tool_call + 对应 tool 响应」的完整组。

    组必须完整：留下半截 tool_call（assistant 说要调用，却没有 tool 响应）
    会让对话结构不合法，vLLM 直接报错。
    """
    anchor: list[dict[str, Any]] = []
    groups: list[list[dict[str, Any]]] = []
    for message in messages:
        role = message.get("role")
        if not groups and role in {"system", "user"}:
            # 上一轮压缩留下的摘要也是 user，但它**不是** anchor：留在 anchor 里的话
            # 下次压缩会在它旁边再加一条新摘要，两条并存且只有新的那条计步数。
            # 挑出来交给 _summarise_group 合并。
            if str(message.get("content") or "").startswith(SUMMARY_MARKER):
                groups.append([dict(message)])
                continue
            anchor.append(dict(message))
            continue
        if role == "assistant":
            groups.append([dict(message)])
        elif groups:
            groups[-1].append(dict(message))
        else:
            # 没有前置 assistant 的 tool 消息：结构本就异常，当 anchor 保住不动。
            anchor.append(dict(message))
    return anchor, groups


def _query_of(group: Sequence[Mapping[str, Any]]) -> str | None:
    """从 assistant 的 tool_call 里取出这一步做了什么，用于摘要。"""
    for message in group:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            raw = function.get("arguments")
            try:
                args = raw if isinstance(raw, Mapping) else json.loads(raw or "{}")
            except (TypeError, ValueError):
                args = {}
            if name == "search_products":
                query = str(args.get("query") or "").strip()
                return query or None
            if name == "open_product":
                asin = str(args.get("asin") or "").strip()
                return f"商品{asin}" if asin else None
            if name == "next_page":
                return "翻页"
    return None


def _carried_summaries(group: Sequence[Mapping[str, Any]]) -> tuple[list[str], int]:
    """摘要组被再次压缩时，把它记住的动作**和步数**都接着传下去。

    步数必须累计：一条长回合会被压缩很多次，每次只报本次省略量的话，第 30 步的
    摘要会写「已省略 1 步」，而实际省了 28 步。那是在给模型错误信息。
    """
    items: list[str] = []
    steps = 0
    for message in group:
        content = str(message.get("content") or "")
        if message.get("role") == "user" and content.startswith(SUMMARY_MARKER):
            head, _, tail = content.partition("]")
            match = re.search(r"(\d+)\s*步", head)
            if match:
                steps += int(match.group(1))
            body = tail.split(":", 1)[-1]
            items.extend(
                part.strip()
                for part in body.split(";")
                if part.strip() and not part.strip().startswith("及更早")
            )
    return items, steps


def _summarise_group(groups: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """把被删掉的组压成一条 user 消息。

    只保留「做过什么」，不保留页面内容。前者丢了会导致重复动作（−0.65），
    后者丢了可以再搜一次。
    """
    carried: list[str] = []
    actions: list[str] = []
    steps = 0
    for group in groups:
        items, carried_steps = _carried_summaries(group)
        carried.extend(items)
        # 之前摘要里记的步数 + 这次真正丢掉的组，两者都算。
        steps += carried_steps
        if any(m.get("role") == "assistant" for m in group):
            steps += 1
        what = _query_of(group)
        if what and what not in actions:
            actions.append(what)

    merged: list[str] = []
    for item in carried + actions:
        if item and item not in merged:
            merged.append(item)
    # 超上限时丢**最老**的：防重复要看的是最近做过什么，几十步前搜过的词再搜一次
    # 代价远小于刚搜过的词立刻重搜。摘要因此始终是一个滑动窗口。
    shown = merged[-_MAX_SUMMARY_ITEMS:]
    omitted = len(merged) - len(shown)
    body = "; ".join(shown) if shown else "（无可摘要的动作）"
    if omitted > 0:
        body = f"（更早 {omitted} 项已省略）; " + body
    # 用 user 角色而不是 system：system 在 anchor 里，混进来会让「固定前缀」
    # 这个概念失效，也会干扰某些模板对 system 只取首条的处理。
    return {
        "role": "user",
        "content": f"{SUMMARY_MARKER} {steps} 步，仅保留动作记录] 已做过: {body}",
    }


def _steps_in(groups: Sequence[Sequence[Mapping[str, Any]]]) -> int:
    return sum(1 for g in groups if any(m.get("role") == "assistant" for m in g))


def compact_messages(
    messages: Sequence[Mapping[str, Any]],
    count_tokens: TokenCounter,
    max_input_tokens: int,
) -> tuple[list[dict[str, Any]], CompactionStats]:
    """把 messages 压到 `max_input_tokens` 以内，返回压缩后的副本与统计。

    不修改入参。未超预算时原样返回（`dropped_groups == 0`）。
    """
    if int(max_input_tokens) < 1:
        raise ValueError("max_input_tokens 必须为正")
    original = [dict(m) for m in messages]
    original_tokens = int(count_tokens(original))
    if original_tokens <= max_input_tokens:
        return original, CompactionStats(
            original_tokens=original_tokens,
            final_tokens=original_tokens,
            dropped_groups=0,
            kept_groups=len(_split_groups(original)[1]),
        )

    anchor, groups = _split_groups(original)
    if not groups:
        raise ContextBudgetError(
            f"固定前缀就用了 {original_tokens} tokens，超出预算 {max_input_tokens}"
        )

    # 最新一组必须留：它是模型当前看到的页面，没有它没法决策。
    floor = anchor + [_summarise_group(groups[:-1])] + list(groups[-1])
    floor_tokens = int(count_tokens(floor))
    if floor_tokens > max_input_tokens:
        raise ContextBudgetError(
            f"固定前缀加最新一步用了 {floor_tokens} tokens，超出预算 "
            f"{max_input_tokens}；应调整窗口或 max_tokens，删更多历史也没用"
        )

    # 二分找能保留的最大组数后缀。count_tokens 可能不单调（摘要长度随删除量变化），
    # 所以最后用找到的 kept 重新构造并校验，不信任二分过程中的中间值。
    low, high, kept = 1, len(groups), 1
    while low <= high:
        middle = (low + high) // 2
        candidate = _build(anchor, groups, middle)
        if int(count_tokens(candidate)) <= max_input_tokens:
            kept = max(kept, middle)
            low = middle + 1
        else:
            high = middle - 1

    compacted = _build(anchor, groups, kept)
    final_tokens = int(count_tokens(compacted))
    # 二分基于可能非单调的度量，这里兜底收缩到真正合规为止。
    while final_tokens > max_input_tokens and kept > 1:
        kept -= 1
        compacted = _build(anchor, groups, kept)
        final_tokens = int(count_tokens(compacted))
    if final_tokens > max_input_tokens:
        compacted, final_tokens = floor, floor_tokens

    return compacted, CompactionStats(
        original_tokens=original_tokens,
        final_tokens=final_tokens,
        dropped_groups=len(groups) - kept,
        kept_groups=kept,
    )


def _build(
    anchor: Sequence[Mapping[str, Any]],
    groups: Sequence[Sequence[Mapping[str, Any]]],
    kept: int,
) -> list[dict[str, Any]]:
    """anchor + （被删组的摘要）+ 最新 kept 组。"""
    head = [dict(m) for m in anchor]
    dropped = groups[: len(groups) - kept]
    if dropped:
        head.append(_summarise_group(dropped))
    for group in groups[len(groups) - kept :]:
        head.extend(dict(m) for m in group)
    return head
