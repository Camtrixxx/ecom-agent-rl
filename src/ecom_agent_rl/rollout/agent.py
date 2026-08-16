"""单回合 rollout：把模型、守卫和环境接起来，产出一条轨迹。

三处刻意与参考实现不同：

1. **答案字段在写盘之前就剥离。** 参考实现把环境返回的 `goal`/`purchase`/
   `reward_detail` 原样存进 `terminal_result`，只在喂 Judge 时才过滤——于是任何
   直接读轨迹文件的下游代码都能看到 gold。这里 `Trajectory.messages` 只含模型真正
   看过的内容，答案统一收进 `audit`，命名上就说明它不能进 prompt。

2. **被拒的动作也计入预算。** 参考实现只在「连续 3 次」被拒时才终止，且被拒不算步数，
   于是「一步合法 + 三次被拒」可以无限循环，一条轨迹能打出远超 max_steps 次模型调用。
   这里另设 `max_rejections` 总额度。

3. **首轮就把初始 observation 给模型。** 参考实现只把 instruction 作为 user 消息，
   初始页面状态压根没进 prompt，模型第一步是盲的。

轨迹里同时记录 `messages`（可训练的对话）和 `steps`（可审计的逐步细节），SFT 直接用
前者，评测和过滤用后者。
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..environment import observation as obs
from ..environment import tools
from ..environment.pool import EnvironmentPool, EnvironmentServiceError
from .llm import ChatClient, ContextOverflowError, EmptyResponseError, LLMError, Usage
from .prompt import SYSTEM_PROMPT
from .tool_call_recovery import recover_tool_calls

logger = logging.getLogger(__name__)

_TRUE = frozenset({"1", "true", "yes", "on"})
# 宽容重解析 + 截断单独记账**是改口径的**：它把一部分 no_tool_call 变成正常步骤或
# TRUNCATED，于是同一份权重在开关两侧的成功率不能直接比。所以默认关。
#
# 为什么做成开关而不是直接改：08-15 之前发布的全部数字（baseline / sft / grpo v1 / v2 /
# 各 seed / 各快照）都是在严格口径下采的。直接改会让「新采的数」和「已发布的数」不可比，
# 而不可比是**静默**的——两边都是一个成功率，看不出差别在哪。开关让旧口径仍然可复现。
#
# 打开时每条轨迹记录里会多出 `tolerant_parse: true`，所以任何一个 jsonl 都能自己说清
# 它是哪个口径采的，不必去翻当时的环境变量。
TOLERANT_PARSE = os.environ.get("ROLLOUT_TOLERANT_PARSE", "").strip().lower() in _TRUE

# 环境侧 SHOP_MAX_STEPS 默认 35，两边必须一致，否则评测的地平线和训练不同。
DEFAULT_MAX_STEPS = 35
# 连续被拒上限：到了就判定模型卡住了。
DEFAULT_MAX_CONSECUTIVE_REJECTIONS = 3
# 全回合被拒总上限，防止「合法一步 + 被拒三次」无限循环。
DEFAULT_MAX_REJECTIONS = 12


class Status:
    """轨迹的终止原因。"""

    DONE = "done"                       # 环境判定回合结束（买了 / 主动结束 / 触顶）
    MAX_STEPS = "max_steps"             # 用完步数预算
    NO_TOOL_CALL = "no_tool_call"       # 模型只说话不调工具（有正文，没动手）
    TRUNCATED = "truncated"             # 回复被 max_tokens 截断，没能给出可执行的调用
    EMPTY_RESPONSE = "empty_response"   # 服务端反复返回空消息（无正文也无工具调用）
    REJECTION_LIMIT = "rejection_limit"  # 被守卫拒绝太多次
    CONTEXT_OVERFLOW = "context_overflow"  # prompt 超出模型上下文窗口
    LLM_ERROR = "llm_error"
    ENV_ERROR = "env_error"
    OBSERVATION_ERROR = "observation_error"


# 这些状态说明是基础设施坏了，不是模型表现差。批量采集遇到就该停下来修，
# 而不是把它当成一条"失败轨迹"混进数据里。
#
# CONTEXT_OVERFLOW 刻意不在此列：它是这一个回合走太远了（长回合把几十个 observation
# 累进 messages），和 MAX_STEPS 同类，是这道题的一个结局。放进来会让一条长回合
# 掐掉整批采集。
#
# EMPTY_RESPONSE 同理不在此列：客户端已经重试过，走到这个状态说明教师在这条回合上
# 反复返回空消息。它不该中止整批，但**也不是模型的决策失败**——采集侧要把它和
# NO_TOOL_CALL 分开统计，否则会把服务端异常算进教师的能力评估里。
INFRA_FAILURES = frozenset({Status.LLM_ERROR, Status.ENV_ERROR, Status.OBSERVATION_ERROR})


@dataclass
class Trajectory:
    task_id: int
    attempt: int = 0
    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: str = "running"
    messages: list[dict[str, Any]] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    rejections: list[dict[str, Any]] = field(default_factory=list)
    # 答案与奖励只进这里。名字就是警告：不要放进 prompt，也不要喂给 Judge。
    audit: dict[str, Any] = field(default_factory=dict)
    # `reward` 是 None 就表示**没有可用的标量**，只有两种成因：回合没走到终局，
    # 或者环境判了 `reward_valid=False`。下面这两个字段用来区分这两种成因。
    #
    # 为什么不把不可信的 0.0 留在这里：`reward_unverifiable` 的取值恰好是 0.0，而
    # 0.0 在 -0.85 ~ 1.0 的区间里是个完全正常的中间值。GRPO 按组算优势，一条
    # 「其实是缺数据」的 0.0 会被当成「不好不坏」参与基线，静默偏移整组的优势信号。
    # 置 None 让「拿 0.0 当真值」在数值上直接不可能——误用会炸，不会悄悄错。
    # 具体的 mask / 罚分策略留到 GRPO 的奖励整形时定，这里只保证信息不丢也不骗人。
    reward: float | None = None
    # 环境的权威终局标签（`reward_detail.reward_type`），没走到终局则为 None。
    reward_type: str | None = None
    # 环境对自己这次打分的信任度。默认 True 与环境一致：没走到终局不代表打分不可信，
    # 只代表没有打分——那种情况由 `reward_type is None` 表达。
    reward_valid: bool = True
    done: bool = False
    error: str | None = None
    # 这一个回合的 token 与压缩用量。批级计数（`ChatClient.usage`）能说明压缩在链路里
    # 生效了，但说明不了是哪些回合被压缩过——`repeat_loop` 的回合明显更长、也几乎都
    # 被压缩过，要判断压缩是不是在制造它就必须能按回合对照（roadmap 阶段 D 末尾）。
    usage: dict[str, int] = field(default_factory=dict)
    # 这条轨迹是在哪个解析口径下采的，以及宽容口径下各救回/判截断了几次。
    # 默认 False 时**不写进记录**：让默认路径产出的 jsonl 与 08-15 之前逐字节同构，
    # 免得同一个文件里前后半段的字段集不一样（评测中途重试会重新导入本模块）。
    tolerant_parse: bool = False
    recovered_tool_calls: int = 0
    truncated_replies: int = 0

    @property
    def env_steps(self) -> int:
        return len(self.steps)

    @property
    def infra_failure(self) -> bool:
        return self.status in INFRA_FAILURES

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "status": self.status,
            "done": self.done,
            "reward": self.reward,
            "reward_type": self.reward_type,
            "reward_valid": self.reward_valid,
            "env_steps": self.env_steps,
            "rejection_count": len(self.rejections),
            "messages": self.messages,
            "steps": self.steps,
            "rejections": self.rejections,
            "audit": self.audit,
            "usage": self.usage,
            "error": self.error,
        }
        # 只在非默认口径下才加这三个键。见 `tolerant_parse` 字段上的说明。
        if self.tolerant_parse:
            record["tolerant_parse"] = True
            record["recovered_tool_calls"] = self.recovered_tool_calls
            record["truncated_replies"] = self.truncated_replies
        return record


def _tool_call_fields(call: Mapping[str, Any]) -> tuple[str, dict[str, Any], str]:
    """从 tool_call 里取出名字与参数。参数是 JSON 字符串，模型经常写坏。"""
    function = call.get("function") or {}
    name = str(function.get("name") or "")
    raw = function.get("arguments")
    call_id = str(call.get("id") or "")
    if raw in (None, ""):
        return name, {}, call_id
    if isinstance(raw, Mapping):
        return name, dict(raw), call_id
    arguments = json.loads(raw)
    if not isinstance(arguments, dict):
        raise ValueError(f"arguments 不是对象: {type(arguments).__name__}")
    return name, arguments, call_id


def _tool_message(call_id: str, content: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": content}


def run_episode(
    *,
    pool: EnvironmentPool,
    client: ChatClient,
    task_id: int,
    attempt: int = 0,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_consecutive_rejections: int = DEFAULT_MAX_CONSECUTIVE_REJECTIONS,
    max_rejections: int = DEFAULT_MAX_REJECTIONS,
    system_prompt: str = SYSTEM_PROMPT,
    tolerant_parse: bool | None = None,
) -> Trajectory:
    """跑一个回合。任何异常都收进 `Trajectory.status`，不向外抛。

    `tolerant_parse` 给 None 就取模块级的 `TOLERANT_PARSE`（来自环境变量
    `ROLLOUT_TOLERANT_PARSE`，默认关）。做成参数而不是只读环境变量，是为了让测试能
    直接把两个口径都跑一遍，而不必去改进程环境再重新导入模块。
    """
    tolerant = TOLERANT_PARSE if tolerant_parse is None else bool(tolerant_parse)
    trajectory = Trajectory(task_id=task_id, attempt=attempt, tolerant_parse=tolerant)
    episode_usage = Usage()
    try:
        with pool.episode(task_id) as episode:
            _drive(
                trajectory,
                episode,
                client,
                max_steps=max_steps,
                max_consecutive_rejections=max_consecutive_rejections,
                max_rejections=max_rejections,
                system_prompt=system_prompt,
                tolerant=tolerant,
                usage=episode_usage,
            )
    except EnvironmentServiceError as exc:
        trajectory.status = Status.ENV_ERROR
        trajectory.error = f"{type(exc).__name__}: {exc}"
    except obs.ObservationError as exc:
        trajectory.status = Status.OBSERVATION_ERROR
        trajectory.error = f"{type(exc).__name__}: {exc}"
    except ContextOverflowError as exc:
        # 必须排在 LLMError 之前：它是 LLMError 的子类。
        trajectory.status = Status.CONTEXT_OVERFLOW
        trajectory.error = f"{type(exc).__name__}: {exc}"
    except EmptyResponseError as exc:
        # 同样必须排在 LLMError 之前。重试已在客户端做过，走到这里说明反复空响应，
        # 是这一个回合的结局而非 infra 故障。
        trajectory.status = Status.EMPTY_RESPONSE
        trajectory.error = f"{type(exc).__name__}: {exc}"
    except LLMError as exc:
        trajectory.status = Status.LLM_ERROR
        trajectory.error = f"{type(exc).__name__}: {exc}"
    finally:
        # 失败的回合也要记用量：撞上下文超限的那些恰恰是压缩压力最大的样本，
        # 只记成功回合会把这个分布裁掉一头。
        trajectory.usage = episode_usage.snapshot()
    return trajectory


def _drive(
    trajectory: Trajectory,
    episode: Any,
    client: ChatClient,
    *,
    max_steps: int,
    max_consecutive_rejections: int,
    max_rejections: int,
    system_prompt: str,
    tolerant: bool,
    usage: Usage | None = None,
) -> None:
    reset_allowed, reset_blocked = obs.split_env_payload(episode.reset_result)
    trajectory.audit["reset_blocked"] = reset_blocked
    trajectory.audit["environment_version"] = reset_allowed.get("environment_version")

    state = obs.validate_state(reset_allowed.get("observation_state"))
    instruction = str(reset_allowed.get("instruction") or "")

    trajectory.messages = [
        {"role": "system", "content": system_prompt},
        # 初始页面一起给：否则模型的第一个动作是在没看到页面的情况下猜的。
        {"role": "user", "content": f"{instruction}\n\n{obs.render(state)}"},
    ]

    consecutive_rejections = 0
    while len(trajectory.steps) < max_steps:
        assistant = client.complete(trajectory.messages, tools.TOOL_SCHEMAS, usage=usage)
        calls = list(assistant.get("tool_calls") or [])

        # vLLM 的 hermes 解析器是全有或全无（源码 hermes_tool_parser.py:87-120）：正文里
        # 首块合法、尾部有垃圾（复读，或被 1024 token 上限腰斩），整条回复就一个
        # tool_call 都不返回。08-14 实测 grpo_v2 的 256 条标签轨迹里 206 条（80.5%）
        # 首块是合法的，也就是说这些回合里模型其实动了手，只是记账把它记成了没动。
        #
        # 兜底放在这一层而不是 llm.py：llm.py 是传输层，它如实转述服务端给的字段，那没
        # 有错；错的是**这里**把「没有 tool_calls」直接等同于「模型选择不动手」。
        if tolerant and not calls:
            recovered = recover_tool_calls(assistant.get("content"), tools.TOOL_NAMES)
            if recovered:
                calls = recovered
                trajectory.recovered_tool_calls += 1

        if not calls:
            trajectory.messages.append(
                {"role": "assistant", "content": assistant.get("content") or ""}
            )
            # 「被截断」和「只说话不动手」是两件事：前者是我们把 max_tokens 定在 1024
            # 造成的，后者是模型的决策。挤在同一个 no_tool_call 里就永远分不开，而
            # no_tool_call 恰好是 D2 里判 v2 输掉 6.29 pp 的那个指标。
            if tolerant and assistant.get("finish_reason") == "length":
                trajectory.truncated_replies += 1
                trajectory.status = Status.TRUNCATED
            else:
                trajectory.status = Status.NO_TOOL_CALL
            return

        # 每轮只执行第一个工具调用：后面的调用是基于同一个旧 observation 生成的，
        # 执行它们等于在已经变化的页面上盲点。多余的调用要从消息里删掉，
        # 否则 assistant 有 N 个 tool_call 而只有 1 个 tool 响应，对话就不合法了。
        dropped = calls[1:]
        call = calls[0]
        assistant_message = {
            "role": "assistant",
            "content": assistant.get("content") or "",
            "tool_calls": [call],
        }
        # thinking 模式的教师要求把 reasoning_content 回传（见 llm.echo_reasoning），
        # 所以原样留在消息里。它不会进训练样本：data/sft.py 的字段白名单会剥掉它，
        # 学生学的是动作而不是某个教师的思维链格式。
        if assistant.get("reasoning_content"):
            assistant_message["reasoning_content"] = assistant["reasoning_content"]

        try:
            name, arguments, call_id = _tool_call_fields(call)
            tools.check(name, arguments, state)
        except (tools.RejectedAction, ValueError, KeyError, TypeError) as exc:
            if isinstance(exc, tools.RejectedAction):
                reason, message = exc.reason, exc.message
            else:
                # 参数 JSON 坏了、缺 function 字段等，都按被拒处理并把原因告诉模型。
                reason = f"malformed_tool_call:{type(exc).__name__}"
                message = f"工具调用无法解析：{exc}"
            consecutive_rejections += 1
            trajectory.rejections.append(
                {
                    "at_step": len(trajectory.steps),
                    "tool_call": call,
                    "reason": reason,
                    "message": message,
                    "dropped_tool_calls": dropped,
                }
            )
            trajectory.messages.append(assistant_message)
            trajectory.messages.append(
                _tool_message(str(call.get("id") or ""), f"动作被拒绝，未执行。{message}")
            )
            if consecutive_rejections >= max_consecutive_rejections:
                trajectory.status = Status.REJECTION_LIMIT
                trajectory.error = f"连续 {consecutive_rejections} 次动作被拒"
                return
            if len(trajectory.rejections) >= max_rejections:
                trajectory.status = Status.REJECTION_LIMIT
                trajectory.error = f"累计 {len(trajectory.rejections)} 次动作被拒"
                return
            continue

        consecutive_rejections = 0
        action = tools.to_env_action(name, arguments)
        result = episode.interact(action)
        allowed, blocked = obs.split_env_payload(result.raw)
        state = obs.validate_state(allowed.get("observation_state"))
        rendered = obs.render(state)

        trajectory.steps.append(
            {
                "step": len(trajectory.steps),
                "tool_name": name,
                "arguments": arguments,
                "env_action": action,
                "observation": rendered,
                "done": result.done,
                "over": result.over,
                "dropped_tool_calls": dropped,
            }
        )
        trajectory.messages.append(assistant_message)
        trajectory.messages.append(_tool_message(call_id, rendered))

        if blocked:
            trajectory.audit.setdefault("step_blocked", []).append(
                {"step": len(trajectory.steps) - 1, "blocked": blocked}
            )

        if result.done:
            trajectory.status = Status.DONE
            trajectory.done = True
            trajectory.reward = result.reward
            trajectory.audit["terminal"] = blocked
            # 终局标签与可信度从 audit 提到顶层：下游要判断这条能不能进训练，不该
            # 被迫去翻一个名字就写着「不要读」的字段。audit 里的原文照旧保留。
            terminal = blocked or {}
            detail = terminal.get("reward_detail") or {}
            reward_type = detail.get("reward_type")
            trajectory.reward_type = str(reward_type) if reward_type else None
            # 环境把 `reward_valid` 同时放在终局 payload 顶层和 `reward_detail` 里。
            # 4,000 条已有轨迹里两处从未不一致，但两处都读、有一处说不可信就算不可信：
            # 这个字段唯一的作用就是拦下不可信的分，读漏一处等于白加。
            trajectory.reward_valid = (
                terminal.get("reward_valid", True) is not False
                and detail.get("reward_valid", True) is not False
            )
            if not trajectory.reward_valid:
                trajectory.reward = None
            return

        if result.over:
            # 环境侧 history 触顶。它不给 reward，只能当成用完预算。
            trajectory.status = Status.MAX_STEPS
            trajectory.error = "环境报告 over 但未 done"
            return

    trajectory.status = Status.MAX_STEPS
