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
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from ..environment import observation as obs
from ..environment import tools
from ..environment.pool import EnvironmentPool, EnvironmentServiceError
from .llm import ChatClient, ContextOverflowError, EmptyResponseError, LLMError
from .prompt import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

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
    reward: float | None = None
    done: bool = False
    error: str | None = None

    @property
    def env_steps(self) -> int:
        return len(self.steps)

    @property
    def infra_failure(self) -> bool:
        return self.status in INFRA_FAILURES

    def as_record(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_id": self.task_id,
            "attempt": self.attempt,
            "status": self.status,
            "done": self.done,
            "reward": self.reward,
            "env_steps": self.env_steps,
            "rejection_count": len(self.rejections),
            "messages": self.messages,
            "steps": self.steps,
            "rejections": self.rejections,
            "audit": self.audit,
            "error": self.error,
        }


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
) -> Trajectory:
    """跑一个回合。任何异常都收进 `Trajectory.status`，不向外抛。"""
    trajectory = Trajectory(task_id=task_id, attempt=attempt)
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
        assistant = client.complete(trajectory.messages, tools.TOOL_SCHEMAS)
        calls = list(assistant.get("tool_calls") or [])

        if not calls:
            trajectory.messages.append(
                {"role": "assistant", "content": assistant.get("content") or ""}
            )
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
            return

        if result.over:
            # 环境侧 history 触顶。它不给 reward，只能当成用完预算。
            trajectory.status = Status.MAX_STEPS
            trajectory.error = "环境报告 over 但未 done"
            return

    trajectory.status = Status.MAX_STEPS
