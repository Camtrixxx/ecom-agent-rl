"""批量驱动的不变量：续跑去重、失败分流、并发下的写盘。

`run_batch` 之前没有测试，代价是一个真实的 bug 溜到了实测阶段：基础设施失败被写进
主轨迹文件，于是续跑按 `(task_id, attempt)` 去重时永远跳过它们，而下游指标层把它们
当成模型的失败来算。这里的测试就是围着这条线写的。

不用真环境：`run_batch` 的职责是调度与写盘，回合内部的行为由 test_rollout.py 覆盖。
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from ecom_agent_rl.rollout.agent import Status, Trajectory
from ecom_agent_rl.rollout.batch import (
    completed_attempts,
    load_task_ids,
    run_batch,
)


class StubPool:
    """只提供 urls（用于推并发默认值），不真的开回合。"""

    def __init__(self, workers: int = 2) -> None:
        self.urls = [f"http://fake:{5700 + i}" for i in range(workers)]


class StubClient:
    def __init__(self) -> None:
        class _Usage:
            def snapshot(self) -> dict[str, int]:
                return {"calls": 0}

        self.usage = _Usage()


def outcome(task_id: int, attempt: int, status: str = Status.DONE,
            reward: float | None = 1.0) -> Trajectory:
    trajectory = Trajectory(task_id=task_id, attempt=attempt)
    trajectory.status = status
    trajectory.done = status == Status.DONE
    trajectory.reward = reward
    if status == Status.DONE:
        trajectory.audit["terminal"] = {
            "reward_detail": {"reward_type": "gold_purchase", "reward_valid": True}
        }
    else:
        trajectory.error = f"stub {status}"
    return trajectory


def driver(plan: dict[tuple[int, int], Trajectory], seen: list | None = None):
    """把 (task_id, attempt) 映射到预设结果，替掉真的 run_episode。"""
    def fake_run_episode(*, pool, client, task_id, attempt, **kwargs):
        if seen is not None:
            seen.append((task_id, attempt))
        return plan[(task_id, attempt)]

    return fake_run_episode


def batch(tmp_path: Path, monkeypatch, plan, task_ids, *, attempts=1,
          seen=None, **kwargs):
    monkeypatch.setattr(
        "ecom_agent_rl.rollout.batch.run_episode", driver(plan, seen)
    )
    return run_batch(
        pool=StubPool(), client=StubClient(), task_ids=task_ids,
        output=tmp_path / "out.jsonl", attempts=attempts, **kwargs
    )


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


# --- 基础设施失败的分流（这就是那个 bug） --------------------------------


@pytest.mark.parametrize(
    "status", [Status.ENV_ERROR, Status.LLM_ERROR, Status.OBSERVATION_ERROR]
)
def test_infra_failures_do_not_go_into_the_main_trajectory_file(
    tmp_path, monkeypatch, status
):
    """环境挂了不是模型的表现，写进主文件会被指标层当成失败率。"""
    plan = {(1, 0): outcome(1, 0), (2, 0): outcome(2, 0, status, reward=None)}
    summary = batch(tmp_path, monkeypatch, plan, [1, 2])

    written = lines(tmp_path / "out.jsonl")
    assert [r["task_id"] for r in written] == [1]
    assert summary["written"] == 1
    assert summary["infra_failed"] == 1


@pytest.mark.parametrize(
    "status", [Status.ENV_ERROR, Status.LLM_ERROR, Status.OBSERVATION_ERROR]
)
def test_infra_failures_are_archived_to_a_sidecar_file(tmp_path, monkeypatch, status):
    """不进主文件，但要留证据——否则查不了当时到底哪台挂了。"""
    plan = {(1, 0): outcome(1, 0, status, reward=None)}
    batch(tmp_path, monkeypatch, plan, [1])

    archived = lines(tmp_path / "out.failures.jsonl")
    assert len(archived) == 1
    assert archived[0]["status"] == status
    assert archived[0]["error"]


def test_a_retry_after_fixing_the_infrastructure_actually_reruns_the_episode(
    tmp_path, monkeypatch
):
    """这是分流的全部意义：'修好之后重跑即可续跑' 必须是真的。"""
    first = {(1, 0): outcome(1, 0), (2, 0): outcome(2, 0, Status.ENV_ERROR, reward=None)}
    batch(tmp_path, monkeypatch, first, [1, 2])

    seen: list = []
    second = {(1, 0): outcome(1, 0), (2, 0): outcome(2, 0)}
    summary = batch(tmp_path, monkeypatch, second, [1, 2], seen=seen)

    assert seen == [(2, 0)], "task 1 该跳过，task 2 该重试"
    assert summary["skipped"] == 1
    assert sorted(r["task_id"] for r in lines(tmp_path / "out.jsonl")) == [1, 2]


def test_model_failures_are_kept_because_they_are_real_outcomes(tmp_path, monkeypatch):
    """跑满步数、不调工具是模型的表现，必须留在主文件里计入指标。"""
    plan = {
        (1, 0): outcome(1, 0, Status.MAX_STEPS, reward=None),
        (2, 0): outcome(2, 0, Status.NO_TOOL_CALL, reward=None),
        (3, 0): outcome(3, 0, Status.REJECTION_LIMIT, reward=None),
    }
    summary = batch(tmp_path, monkeypatch, plan, [1, 2, 3])
    assert summary["written"] == 3
    assert summary["infra_failed"] == 0
    assert len(lines(tmp_path / "out.jsonl")) == 3


def test_an_infra_failure_aborts_the_batch(tmp_path, monkeypatch):
    """继续跑只是在稳定地生产垃圾。"""
    plan = {(i, 0): outcome(i, 0) for i in range(1, 6)}
    plan[(1, 0)] = outcome(1, 0, Status.ENV_ERROR, reward=None)
    summary = batch(tmp_path, monkeypatch, plan, [1, 2, 3, 4, 5], concurrency=1)
    assert summary["aborted"]
    assert "env_error" in summary["aborted"]
    assert summary["completed"] < summary["planned"]


def test_the_metrics_layer_sees_no_infra_noise_after_the_split(tmp_path, monkeypatch):
    """端到端：分流之后终局类型里不该再出现 no_terminal:env_error。"""
    from ecom_agent_rl.evaluation.metrics import load_outcomes, summarize

    plan = {(i, 0): outcome(i, 0) for i in range(1, 4)}
    plan[(3, 0)] = outcome(3, 0, Status.ENV_ERROR, reward=None)
    batch(tmp_path, monkeypatch, plan, [1, 2, 3], concurrency=1)

    summary = summarize(load_outcomes(tmp_path / "out.jsonl"))
    assert not any(k.startswith("no_terminal:env_error") for k in summary["reward_types"])
    assert summary["success_rate"]["mean"] == 1.0


# --- 续跑去重 -------------------------------------------------------------


def test_resume_skips_already_written_attempts(tmp_path, monkeypatch):
    plan = {(1, 0): outcome(1, 0), (1, 1): outcome(1, 1)}
    batch(tmp_path, monkeypatch, plan, [1], attempts=2)

    seen: list = []
    summary = batch(tmp_path, monkeypatch, plan, [1], attempts=2, seen=seen)
    assert seen == []
    assert summary["skipped"] == 2 and summary["planned"] == 0


def test_no_resume_reruns_everything(tmp_path, monkeypatch):
    plan = {(1, 0): outcome(1, 0)}
    batch(tmp_path, monkeypatch, plan, [1])
    seen: list = []
    batch(tmp_path, monkeypatch, plan, [1], seen=seen, resume=False)
    assert seen == [(1, 0)]


def test_resume_counts_attempts_per_task_not_just_tasks(tmp_path, monkeypatch):
    """采了 2 次要再采到 4 次时，只该补跑 attempt 2 和 3。"""
    plan = {(1, a): outcome(1, a) for a in range(4)}
    batch(tmp_path, monkeypatch, plan, [1], attempts=2)
    seen: list = []
    batch(tmp_path, monkeypatch, plan, [1], attempts=4, seen=seen)
    assert sorted(seen) == [(1, 2), (1, 3)]


def test_completed_attempts_tolerates_a_truncated_last_line(tmp_path):
    """进程被杀在半行是续跑要解决的问题，不是要报错的问题。"""
    path = tmp_path / "out.jsonl"
    path.write_text(
        json.dumps({"task_id": 1, "attempt": 0}) + "\n"
        + '{"task_id": 2, "attem',
        encoding="utf-8",
    )
    assert completed_attempts(path) == {(1, 0)}


def test_completed_attempts_on_a_missing_file_is_empty(tmp_path):
    assert completed_attempts(tmp_path / "nope.jsonl") == set()


# --- 调度 -----------------------------------------------------------------


def test_default_concurrency_is_the_worker_count_not_the_slot_capacity(
    tmp_path, monkeypatch
):
    """实测最优工作点是并发 = worker 数；按 capacity 会超配、尾延迟爆掉。"""
    captured: dict[str, int] = {}
    real_executor = __import__(
        "concurrent.futures", fromlist=["ThreadPoolExecutor"]
    ).ThreadPoolExecutor

    class SpyExecutor(real_executor):
        def __init__(self, max_workers=None, **kw):
            captured["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kw)

    monkeypatch.setattr("ecom_agent_rl.rollout.batch.ThreadPoolExecutor", SpyExecutor)
    plan = {(i, 0): outcome(i, 0) for i in range(1, 11)}
    batch(tmp_path, monkeypatch, plan, list(range(1, 11)))
    # StubPool 有 2 个 url，且 capacity（slot 倍数）会更大。
    assert captured["max_workers"] == 2


def test_concurrency_never_exceeds_the_planned_episode_count(tmp_path, monkeypatch):
    captured: dict[str, int] = {}
    real_executor = __import__(
        "concurrent.futures", fromlist=["ThreadPoolExecutor"]
    ).ThreadPoolExecutor

    class SpyExecutor(real_executor):
        def __init__(self, max_workers=None, **kw):
            captured["max_workers"] = max_workers
            super().__init__(max_workers=max_workers, **kw)

    monkeypatch.setattr("ecom_agent_rl.rollout.batch.ThreadPoolExecutor", SpyExecutor)
    batch(tmp_path, monkeypatch, {(1, 0): outcome(1, 0)}, [1], concurrency=32)
    assert captured["max_workers"] == 1


def test_concurrent_writes_produce_one_valid_json_per_line(tmp_path, monkeypatch):
    """多线程共写一个文件，没锁就会写出交错的坏行。"""
    plan = {(i, 0): outcome(i, 0) for i in range(200)}
    batch(tmp_path, monkeypatch, plan, list(range(200)), concurrency=16)
    records = lines(tmp_path / "out.jsonl")
    assert len(records) == 200
    assert sorted(r["task_id"] for r in records) == list(range(200))


def test_every_planned_episode_runs_exactly_once(tmp_path, monkeypatch):
    seen: list = []
    plan = {(t, a): outcome(t, a) for t in range(20) for a in range(3)}
    batch(tmp_path, monkeypatch, plan, list(range(20)), attempts=3,
          seen=seen, concurrency=8)
    assert sorted(seen) == sorted(plan)


def test_on_result_is_called_for_each_trajectory(tmp_path, monkeypatch):
    got: list[int] = []
    lock = threading.Lock()

    def collect(trajectory):
        with lock:
            got.append(trajectory.task_id)

    plan = {(i, 0): outcome(i, 0) for i in range(10)}
    batch(tmp_path, monkeypatch, plan, list(range(10)), on_result=collect)
    assert sorted(got) == list(range(10))


# --- 汇总 -----------------------------------------------------------------


def test_summary_reports_mean_reward_only_over_rewarded_episodes(tmp_path, monkeypatch):
    plan = {
        (1, 0): outcome(1, 0, reward=1.0),
        (2, 0): outcome(2, 0, reward=0.0),
        (3, 0): outcome(3, 0, Status.MAX_STEPS, reward=None),
    }
    summary = batch(tmp_path, monkeypatch, plan, [1, 2, 3])
    assert summary["rewarded_episodes"] == 2
    assert summary["mean_reward"] == 0.5


def test_status_counts_cover_every_completed_episode(tmp_path, monkeypatch):
    plan = {
        (1, 0): outcome(1, 0),
        (2, 0): outcome(2, 0, Status.MAX_STEPS, reward=None),
        (3, 0): outcome(3, 0, Status.NO_TOOL_CALL, reward=None),
    }
    summary = batch(tmp_path, monkeypatch, plan, [1, 2, 3])
    assert sum(summary["status_counts"].values()) == summary["completed"] == 3


def test_load_task_ids_reads_the_pool(tmp_path):
    path = tmp_path / "pool.jsonl"
    path.write_text(
        json.dumps({"task_id": 7, "domain": "Home"}) + "\n\n"
        + json.dumps({"task_id": 9, "domain": "Kids"}) + "\n",
        encoding="utf-8",
    )
    assert load_task_ids(path) == [7, 9]
