"""回合循环的不变量。

用假的模型和假的环境驱动真实的 `run_episode`，钉住三件参考实现踩过的事：

1. 轨迹里不能出现答案字段——它会被写进 jsonl，下游任何人读到就是泄漏。
2. 被拒的动作要计入总额度，否则「合法一步 + 连拒三次」可以无限循环。
3. 每轮只执行第一个工具调用，且消息里多余的 tool_call 必须删掉，否则对话不合法。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ecom_agent_rl.environment.observation import OBSERVATION_VERSION
from ecom_agent_rl.rollout.agent import Status, run_episode
from ecom_agent_rl.rollout.llm import Usage
from conftest import SEARCH_HOME, detail_state, search_state


def tool_call(name: str, arguments: dict[str, Any], call_id: str = "c1") -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
    }


class FakeClient:
    """按脚本依次返回 assistant 消息。脚本用完还被调用就是测试写错了。"""

    def __init__(
        self, script: list[dict[str, Any]], compactions_per_call: int = 0
    ) -> None:
        self.script = list(script)
        self.seen: list[list[dict[str, Any]]] = []
        self.compactions_per_call = compactions_per_call
        # 用真的 Usage：每回合计数与批级计数共用同一套累加逻辑，假一个就测不到分叉。
        self.usage = Usage()

    def complete(self, messages, tools=None, usage=None):
        self.seen.append([dict(m) for m in messages])
        if not self.script:
            raise AssertionError("模型被调用的次数超出脚本")
        for sink in (self.usage,) if usage is None else (self.usage, usage):
            sink.add({"prompt_tokens": 100, "completion_tokens": 10})
            if self.compactions_per_call:
                sink.add_compaction(9000, self.compactions_per_call)
        return self.script.pop(0)


class FakeEpisode:
    def __init__(self, reset_result: dict[str, Any], steps: list[dict[str, Any]]) -> None:
        self.reset_result = reset_result
        self._steps = list(steps)
        self.actions: list[str] = []

    def interact(self, action: str):
        self.actions.append(action)
        raw = self._steps.pop(0)

        class _Step:
            def __init__(self, raw: dict[str, Any]) -> None:
                self.raw = raw
                self.done = bool(raw.get("done"))
                self.over = bool(raw.get("over"))

            @property
            def reward(self):
                value = self.raw.get("reward")
                return float(value) if isinstance(value, (int, float)) else None

        return _Step(raw)


class FakePool:
    def __init__(self, episode: FakeEpisode) -> None:
        self._episode = episode
        self.urls = ["http://fake"]

    def episode(self, task_idx: int):
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            yield self._episode

        return _ctx()


def reset_payload(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """带上真实环境确实会回的答案字段，验证它们不会流进轨迹。"""
    return {
        "instruction": "想给狗狗买件衣服，预算 35",
        "instruction_simple": "推荐香草奶昔色狗狗衣服",
        "goal_options": ["香草奶昔运动衣 绿", "XS：胸围31cm（建议1-3斤）"],
        "user_persona": "养狗的年轻人",
        "reason_key": "k",
        "env_idx": 1,
        "idx": 7,
        "message": "Task 7 started",
        "environment_version": "shopsimulator-environment-v2.1",
        "observation_state": state or SEARCH_HOME,
    }


def step_payload(state: dict[str, Any], **extra: Any) -> dict[str, Any]:
    payload = {
        "done": False,
        "over": False,
        "reward": 0.0,
        "instruction": "rendered text",
        "message": "Continue interaction",
        "env_idx": 1,
        "idx": "slot-1-7",
        "reward_detail": {},
        "purchase": {},
        "goal": {},
        "observation_state": state,
        # 实测每一步都带 progress，里面有逐条约束判定。
        "progress": {"credited_evidence_added": ["constraint:900000000000:budget:fail"]},
    }
    payload.update(extra)
    return payload


def run(script, steps, reset_state=None, compactions_per_call=0, **kwargs):
    client = FakeClient(script, compactions_per_call=compactions_per_call)
    episode = FakeEpisode(reset_payload(reset_state), steps)
    trajectory = run_episode(
        pool=FakePool(episode), client=client, task_id=7, **kwargs
    )
    return trajectory, client, episode


def test_no_answer_field_reaches_the_trajectory_messages():
    """goal_options / progress / reward_detail 都不许出现在模型看过的内容里。"""
    trajectory, _, _ = run(
        [{"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]}],
        [step_payload(search_state(), done=True, reward=1.0,
                      goal={"asin": "900000000000"},
                      reward_detail={"brand": 1.0},
                      purchase={"asin": "900000000000"})],
    )
    assert trajectory.status == Status.DONE
    blob = json.dumps(trajectory.messages, ensure_ascii=False)
    for secret in ("香草奶昔运动衣", "budget:fail", "reward_detail", "养狗的年轻人"):
        assert secret not in blob, f"{secret} 泄漏进了 messages"


def test_answer_fields_are_kept_in_audit_for_offline_use():
    """剥离不等于丢掉：过滤教师轨迹要用 reward，所以答案收进 audit。"""
    trajectory, _, _ = run(
        [{"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]}],
        [step_payload(search_state(), done=True, reward=1.0, goal={"asin": "900000000000"})],
    )
    assert trajectory.reward == 1.0
    assert trajectory.audit["reset_blocked"]["goal_options"]
    assert trajectory.audit["terminal"]["goal"] == {"asin": "900000000000"}


def test_terminal_label_is_hoisted_to_the_trajectory_top_level():
    """终局标签要能不翻 audit 就读到：audit 这个名字本身就是「别读我」。"""
    trajectory, _, _ = run(
        [{"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]}],
        [step_payload(search_state(), done=True, reward=1.0,
                      reward_detail={"reward_type": "gold_purchase", "reward_valid": True})],
    )
    record = trajectory.as_record()
    assert record["reward_type"] == "gold_purchase"
    assert record["reward_valid"] is True
    assert record["reward"] == 1.0


def test_an_unverifiable_reward_is_never_exposed_as_the_number_zero():
    """`reward_unverifiable` 的取值恰好是 0.0，而 0.0 在 -0.85~1.0 里是正常的中间值。

    留着它，GRPO 按组算优势时会把一条「其实是缺数据」的轨迹当成「不好不坏」算进
    基线，静默偏移整组的信号。置 None 让误用变成显式的 TypeError。
    """
    trajectory, _, _ = run(
        [{"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]}],
        [step_payload(search_state(), done=True, reward=0.0,
                      reward_detail={"reward_type": "reward_unverifiable",
                                     "reward_valid": False})],
    )
    record = trajectory.as_record()
    assert record["reward"] is None, "不可信的 0.0 仍然可以被当成真值读出去"
    assert record["reward_valid"] is False
    assert record["reward_type"] == "reward_unverifiable"
    # 原始判定照旧留在 audit 里，剥离不等于丢掉。
    assert trajectory.audit["terminal"]["reward_detail"]["reward_valid"] is False


def test_an_episode_without_a_terminal_has_no_reward_type():
    """没走到终局与「打分不可信」是两件事，不能挤进同一个字段。

    reward_valid 沿用环境的默认 True：没有打分不等于打分不可信，「有没有分」由
    reward_type 是不是 None 表达。指标层靠这条区分「模型的失败」和「缺数据」。
    """
    trajectory, _, _ = run(
        [{"content": "我觉得应该搜一下 open_product(...)"}],
        [],
    )
    record = trajectory.as_record()
    assert record["status"] == Status.NO_TOOL_CALL
    assert record["reward"] is None
    assert record["reward_type"] is None
    assert record["reward_valid"] is True


def test_compaction_is_counted_per_episode_not_only_per_batch():
    """批级计数说明「压缩生效了」，但说明不了「哪些回合被压缩过」。

    roadmap 阶段 D 有条没定论的线索：`repeat_loop` 的回合明显更长、也几乎都被压缩
    过，而买对的回合多数没有。要判断压缩是不是在制造 `repeat_loop`，就得能按回合
    做等步数对照——只有批级计数是做不了的。
    """
    trajectory, client, _ = run(
        [
            {"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]},
            {"tool_calls": [tool_call("open_product", {"asin": "900000000000"})]},
        ],
        [step_payload(search_state()), step_payload(detail_state(), done=True, reward=1.0)],
        compactions_per_call=1,
    )
    usage = trajectory.as_record()["usage"]
    assert usage["calls"] == 2
    assert usage["compactions"] == 2
    assert usage["peak_original_tokens"] == 9000
    # 批级计数必须同步走，不能被每回合的计数器截走。
    assert client.usage.snapshot()["compactions"] == 2


def test_a_failed_episode_still_records_its_usage():
    """撞上下文超限的回合恰恰是压缩压力最大的样本，漏记它会把分布裁掉一头。"""
    client = FakeClient([{"content": "我不动手"}], compactions_per_call=1)
    episode = FakeEpisode(reset_payload(), [])
    trajectory = run_episode(pool=FakePool(episode), client=client, task_id=7)
    assert trajectory.status == Status.NO_TOOL_CALL
    assert trajectory.as_record()["usage"]["compactions"] == 1


def test_the_first_user_message_contains_the_initial_observation():
    """参考实现只给 instruction，模型第一步是在没看到页面的情况下猜的。"""
    _, client, _ = run(
        [{"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]}],
        [step_payload(search_state(), done=True, reward=1.0)],
    )
    first_user = client.seen[0][1]["content"]
    assert "预算 35" in first_user
    assert "【搜索首页】" in first_user


def test_rejected_actions_never_reach_the_environment():
    """守卫拒掉的动作一步都不能发出去——环境对非法 click 是静默 no-op。"""
    trajectory, _, episode = run(
        [
            {"tool_calls": [tool_call("open_product", {"asin": "111111111111"})]},
            {"tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]},
        ],
        [step_payload(search_state(), done=True, reward=1.0)],
    )
    assert episode.actions == ["search[狗狗衣服]"]
    assert [r["reason"] for r in trajectory.rejections] == ["asin_not_on_page"]


def test_three_consecutive_rejections_end_the_episode():
    calls = [{"tool_calls": [tool_call("buy_now", {})]} for _ in range(3)]
    trajectory, _, episode = run(calls, [])
    assert trajectory.status == Status.REJECTION_LIMIT
    assert episode.actions == []
    assert len(trajectory.rejections) == 3


def test_a_total_rejection_budget_stops_the_legal_step_plus_three_rejections_loop():
    """参考实现只看「连续」次数，被拒又不算步数，于是这个循环可以无限跑下去。"""
    script: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for _ in range(10):
        script.append({"tool_calls": [tool_call("search_products", {"query": "q"})]})
        steps.append(step_payload(SEARCH_HOME))  # 仍在首页，可以继续搜
        script.append({"tool_calls": [tool_call("buy_now", {})]})
        script.append({"tool_calls": [tool_call("buy_now", {})]})
    trajectory, _, _ = run(script, steps, max_rejections=4)
    assert trajectory.status == Status.REJECTION_LIMIT
    assert len(trajectory.rejections) == 4
    assert "累计" in (trajectory.error or "")


def test_a_successful_step_resets_the_consecutive_counter():
    trajectory, _, episode = run(
        [
            {"tool_calls": [tool_call("buy_now", {})]},
            {"tool_calls": [tool_call("buy_now", {})]},
            {"tool_calls": [tool_call("search_products", {"query": "q"})]},
            {"tool_calls": [tool_call("buy_now", {})]},
            {"tool_calls": [tool_call("search_products", {"query": "q2"})]},
        ],
        [step_payload(SEARCH_HOME), step_payload(SEARCH_HOME, done=True, reward=0.55)],
    )
    assert trajectory.status == Status.DONE
    assert len(trajectory.rejections) == 3  # 但从未连续 3 次
    assert episode.actions == ["search[q]", "search[q2]"]


def test_only_the_first_tool_call_runs_and_the_rest_are_removed_from_the_message():
    """多余的 tool_call 留在 assistant 消息里，对话就少了对应的 tool 响应，非法。"""
    trajectory, _, episode = run(
        [
            {
                "tool_calls": [
                    tool_call("search_products", {"query": "狗狗衣服"}, "a"),
                    tool_call("buy_now", {}, "b"),
                ]
            }
        ],
        [step_payload(search_state(), done=True, reward=1.0)],
    )
    assert episode.actions == ["search[狗狗衣服]"]
    assistant = [m for m in trajectory.messages if m["role"] == "assistant"][0]
    assert len(assistant["tool_calls"]) == 1
    assert trajectory.steps[0]["dropped_tool_calls"][0]["id"] == "b"


def test_every_assistant_tool_call_has_exactly_one_tool_response():
    """SFT 会直接吃 messages，对话结构不合法就训不了。"""
    trajectory, _, _ = run(
        [
            {"tool_calls": [tool_call("open_product", {"asin": "111111111111"})]},
            {"tool_calls": [tool_call("search_products", {"query": "q"}, "x")]},
            {"tool_calls": [tool_call("open_product", {"asin": "900000000000"}, "y")]},
        ],
        [step_payload(search_state()), step_payload(detail_state(), done=True, reward=1.0)],
    )
    assert trajectory.status == Status.DONE
    pending: list[str] = []
    for message in trajectory.messages:
        if message["role"] == "assistant" and message.get("tool_calls"):
            assert not pending, "上一轮的 tool_call 还没有响应"
            pending = [c["id"] for c in message["tool_calls"]]
        elif message["role"] == "tool":
            assert pending and message["tool_call_id"] == pending.pop(0)
    assert not pending


def test_malformed_tool_arguments_are_treated_as_a_rejection_not_a_crash():
    bad = {"id": "c1", "type": "function",
           "function": {"name": "search_products", "arguments": "{not json"}}
    trajectory, _, _ = run(
        [{"tool_calls": [bad]},
         {"tool_calls": [tool_call("search_products", {"query": "q"})]}],
        [step_payload(SEARCH_HOME, done=True, reward=0.55)],
    )
    assert trajectory.status == Status.DONE
    assert trajectory.rejections[0]["reason"].startswith("malformed_tool_call:")


def test_a_reply_without_tool_calls_ends_the_episode():
    trajectory, _, _ = run([{"content": "我建议你买这件。"}], [])
    assert trajectory.status == Status.NO_TOOL_CALL
    assert trajectory.reward is None


def test_the_step_budget_is_enforced():
    script = [{"tool_calls": [tool_call("search_products", {"query": f"q{i}"})]} for i in range(5)]
    steps = [step_payload(SEARCH_HOME) for _ in range(5)]
    trajectory, _, episode = run(script, steps, max_steps=3)
    assert trajectory.status == Status.MAX_STEPS
    assert len(episode.actions) == 3


def test_an_unknown_environment_field_fails_loudly():
    """环境升级悄悄加了含答案的字段时，宁可整批失败也不要静默泄漏。"""
    trajectory, _, _ = run(
        [{"tool_calls": [tool_call("search_products", {"query": "q"})]}],
        [step_payload(search_state()) | {"gold_answer_v2": "x"}],
    )
    assert trajectory.status == Status.OBSERVATION_ERROR
    assert trajectory.infra_failure


def test_infrastructure_failures_are_distinguished_from_bad_model_behaviour():
    """两者都算「失败」，但一个要修机器，一个是数据。混在一起就没法判断该做什么。"""
    trajectory, _, _ = run([{"content": "算了"}], [])
    assert not trajectory.infra_failure


class _RaisingClient:
    """第 n 次调用时抛指定异常。"""

    def __init__(self, exc: Exception, after: int = 0) -> None:
        self.exc = exc
        self.after = after
        self.calls = 0
        self.usage = Usage()

    def complete(self, messages, tools=None, usage=None):
        self.calls += 1
        for sink in (self.usage,) if usage is None else (self.usage, usage):
            sink.add_compaction(9000, 1)
        if self.calls > self.after:
            raise self.exc
        # 动作要跟着页面走，否则守卫会拒掉（搜过之后就不在搜索首页了），
        # 步数根本涨不上去，测试就测不到「保留已走过的步」。
        latest = messages[-1].get("content") or ""
        if "【搜索首页】" in latest:
            return {"tool_calls": [tool_call("search_products", {"query": "q"})]}
        if "【搜索结果】" in latest:
            return {"tool_calls": [tool_call("back_to_search", {})]}
        return {"tool_calls": [tool_call("view_description", {})]}


def test_context_overflow_ends_the_episode_but_is_not_an_infra_failure():
    """长回合超窗口是这道题的结局，不是机器坏了。

    算成 infra 失败会让一条长回合掐掉整批采集——实测 35 步外推到 ~44k tokens，
    对 24576 的窗口来说这不是罕见情况。
    """
    from ecom_agent_rl.rollout.llm import ContextOverflowError

    episode = FakeEpisode(reset_payload(), [step_payload(search_state())])
    trajectory = run_episode(
        pool=FakePool(episode),
        client=_RaisingClient(ContextOverflowError("HTTP 400: maximum context length")),
        task_id=7,
    )
    assert trajectory.status == Status.CONTEXT_OVERFLOW
    assert not trajectory.infra_failure, "超上下文不该中止整批"


def test_a_genuine_llm_error_is_still_an_infra_failure():
    """模型服务真的挂了要停下来修，不能继续稳定地生产垃圾。"""
    from ecom_agent_rl.rollout.llm import LLMError

    episode = FakeEpisode(reset_payload(), [step_payload(search_state())])
    trajectory = run_episode(
        pool=FakePool(episode),
        client=_RaisingClient(LLMError("重试 3 次仍失败")),
        task_id=7,
    )
    assert trajectory.status == Status.LLM_ERROR
    assert trajectory.infra_failure


def test_context_overflow_keeps_the_steps_taken_so_far():
    """超窗口发生在第 n 步，前 n-1 步是真数据，不该丢。"""
    from ecom_agent_rl.rollout.llm import ContextOverflowError

    episode = FakeEpisode(
        reset_payload(),
        [step_payload(search_state()), step_payload(SEARCH_HOME)],
    )
    trajectory = run_episode(
        pool=FakePool(episode),
        client=_RaisingClient(ContextOverflowError("maximum context length"), after=2),
        task_id=7,
    )
    assert trajectory.status == Status.CONTEXT_OVERFLOW
    # 前 2 次调用各走一步，第 3 次抛；这 2 步是真数据，不该丢。
    assert trajectory.env_steps == 2
    assert not trajectory.rejections


def test_the_teacher_s_reasoning_is_kept_on_the_assistant_message():
    """thinking 模式的教师要求把 reasoning_content 回传，缺了就是 HTTP 400。

    实测：一条探针轨迹因此死掉，且回放稳定复现。所以这里保留原样，由
    `llm.echo_reasoning` 负责回传，`data/sft.py` 的字段白名单负责在建训练集时剥掉。
    """
    trajectory, _, _ = run(
        [{"content": "", "reasoning_content": "先搜一下看看有什么",
          "tool_calls": [tool_call("search_products", {"query": "狗狗衣服"})]}],
        [step_payload(search_state(), done=True, reward=1.0)],
    )
    assistant = [m for m in trajectory.messages if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "先搜一下看看有什么"


def test_an_absent_reasoning_content_is_not_invented():
    """非 thinking 模型不该被塞一个空字段进轨迹——那会写进训练数据。"""
    trajectory, _, _ = run(
        [{"content": "搜一下", "tool_calls": [tool_call("search_products", {"query": "x"})]}],
        [step_payload(search_state(), done=True, reward=1.0)],
    )
    assistant = [m for m in trajectory.messages if m["role"] == "assistant"]
    assert "reasoning_content" not in assistant[0]


# ---------------------------------------------------------------------------
# 宽容重解析与截断记账（默认关，由 tolerant_parse 显式打开）
#
# 背景：vLLM 的 hermes 解析器是全有或全无。正文里首块合法、尾部有垃圾，整条回复就一个
# tool_call 都不返回，于是回合被记成 no_tool_call——而 08-14 实测 grpo_v2 的 256 条标签
# 轨迹里 206 条（80.5%）首块其实是合法的。
#
# 这组测试里**只有一条**是证明兜底能救回来的（test_a_tagged_call_...）。其余全部朝相反
# 方向把关：不许救过头。理由是代价不对称——救不回来只是少算一条，凭空造出一个模型没发过
# 的调用，会把假动作写进 SFT 训练样本，而那种错误在下游是查不出来的。
# ---------------------------------------------------------------------------

TAGGED = '<tool_call>\n{"name": "search_products", "arguments": {"query": "狗狗衣服"}}\n</tool_call>'


def test_a_tagged_call_in_the_content_is_recovered_and_the_episode_continues():
    """首块合法、尾部被腰斩：模型其实动了手，不该记成「选择不动手」。"""
    trajectory, _, episode = run(
        [{"content": TAGGED + '\n<tool_call>\n{"name": "open_prod'}],
        [step_payload(search_state(), done=True, reward=1.0)],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.DONE
    assert episode.actions, "救回来的调用没有真的发给环境"
    assert trajectory.recovered_tool_calls == 1


def test_the_same_reply_is_no_tool_call_when_the_switch_is_off():
    """默认口径必须与 08-15 之前逐字一致，否则新旧数字不可比。"""
    trajectory, _, episode = run(
        [{"content": TAGGED}],
        [],
    )
    assert trajectory.status == Status.NO_TOOL_CALL
    assert not episode.actions
    assert "tolerant_parse" not in trajectory.as_record()


def test_the_recovered_call_is_auditable_in_the_record():
    """救回来的调用要能在事后认出来，否则没法回答「这个数里有多少是兜底救的」。"""
    trajectory, _, _ = run(
        [{"content": TAGGED}],
        [step_payload(search_state(), done=True, reward=1.0)],
        tolerant_parse=True,
    )
    record = trajectory.as_record()
    assert record["tolerant_parse"] is True
    assert record["recovered_tool_calls"] == 1
    assistant = [m for m in trajectory.messages if m["role"] == "assistant"]
    assert assistant[0]["tool_calls"][0]["id"].startswith("recovered-")


def test_prose_that_merely_mentions_a_tool_is_not_recovered():
    """没有标签就没有调用。这是不许救过头的主防线。"""
    trajectory, _, episode = run(
        [{"content": "我觉得应该 search_products({'query': '狗狗衣服'}) 一下"}],
        [],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.NO_TOOL_CALL
    assert not episode.actions
    assert trajectory.recovered_tool_calls == 0


def test_a_tag_naming_an_unknown_tool_is_not_recovered():
    """名字不在 TOOL_SCHEMAS 里就不能执行——救回一个不存在的工具比不救更糟。"""
    trajectory, _, episode = run(
        [{"content": '<tool_call>{"name": "buy_everything", "arguments": {}}</tool_call>'}],
        [],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.NO_TOOL_CALL
    assert not episode.actions


def test_a_truncated_reply_is_not_recorded_as_a_choice_not_to_act():
    """finish_reason=length 是我们把 max_tokens 定在 1024 造成的，不是模型的决策。

    混进 no_tool_call 就永远分不开——而 no_tool_call 正是 D2 里判 v2 输掉 6.29 pp 的
    那个指标。
    """
    trajectory, _, _ = run(
        [{"content": "我先看看这几件衣服哪个" * 40, "finish_reason": "length"}],
        [],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.TRUNCATED
    assert trajectory.as_record()["truncated_replies"] == 1


def test_a_reply_that_really_only_talks_stays_no_tool_call():
    """截断这一类不能把「真的只说话」也吞进去，否则等于换个名字掩盖同一个问题。"""
    trajectory, _, _ = run(
        [{"content": "我觉得这些都不合适，就不买了。", "finish_reason": "stop"}],
        [],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.NO_TOOL_CALL


def test_recovery_wins_over_truncation_when_the_first_block_is_valid():
    """被腰斩但首块合法：这是实测里最常见的一种，必须继续跑而不是记成截断。"""
    trajectory, _, episode = run(
        [{"content": TAGGED + '\n<tool_call>\n{"name": "op', "finish_reason": "length"}],
        [step_payload(search_state(), done=True, reward=1.0)],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.DONE
    assert trajectory.recovered_tool_calls == 1
    assert trajectory.truncated_replies == 0
    assert episode.actions


def test_truncation_is_not_an_infrastructure_failure():
    """它是这道题的一个结局，和 MAX_STEPS 同类。放进 INFRA_FAILURES 会掐掉整批采集。"""
    trajectory, _, _ = run(
        [{"content": "x" * 100, "finish_reason": "length"}],
        [],
        tolerant_parse=True,
    )
    assert trajectory.status == Status.TRUNCATED
    assert not trajectory.infra_failure
