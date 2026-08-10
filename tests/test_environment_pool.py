"""EnvironmentPool 的租约与粘连语义。

不起真实服务：把 `_post` 换成假的，因为要验的是「回合内粘连到同一 worker」和
「租约必然归还」这两条不变量，它们与 HTTP 层无关。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from ecom_agent_rl.environment.pool import EnvironmentPool, EnvironmentServiceError


class FakeService:
    """记录每个 url 收到的请求，并模拟每 worker 的 env_idx 分配。"""

    def __init__(self, slots_per_worker: int = 4) -> None:
        self.slots_per_worker = slots_per_worker
        self.calls: list[tuple[str, dict]] = []
        self.live: dict[str, set[int]] = {}
        self.peak_live: dict[str, int] = {}
        self._lock = threading.Lock()

    def post(self, url: str, payload: dict) -> dict:
        with self._lock:
            self.calls.append((url, payload))
            action = payload.get("action")
            leased = self.live.setdefault(url, set())

            if action == "release_all":
                leased.clear()
                return {"message": "ok"}

            if action == "reset":
                free = next(
                    (i for i in range(self.slots_per_worker) if i not in leased), None
                )
                if free is None:
                    raise EnvironmentServiceError(f"{url}: no free slot")
                leased.add(free)
                self.peak_live[url] = max(self.peak_live.get(url, 0), len(leased))
                return {
                    "env_idx": free,
                    "instruction": f"task {payload['idx']}",
                    "observation_state": "page 1",
                }

            if action == "interact":
                return {"observation_state": "page 2", "over": False, "done": False}

            if action == "release_one":
                leased.discard(payload["env_idx"])
                return {"message": "released"}

            raise AssertionError(f"unexpected action {action}")


@pytest.fixture
def pool_and_service():
    service = FakeService()
    pool = EnvironmentPool(workers=4, slots_per_worker=4)
    pool._post = service.post  # type: ignore[method-assign]
    return pool, service


def test_capacity_is_workers_times_slots(pool_and_service):
    pool, _ = pool_and_service
    assert pool.capacity == 16
    assert len(pool.urls) == 4


def test_episode_requests_all_hit_one_worker(pool_and_service):
    """env_idx 只对分配它的进程有效，所以一个回合不能跨 worker。"""
    pool, service = pool_and_service
    with pool.episode(7) as episode:
        for _ in range(5):
            episode.interact("search[x]")

    urls = {url for url, _ in service.calls}
    assert len(urls) == 1, f"episode spanned multiple workers: {urls}"


def test_episode_releases_lease_on_success(pool_and_service):
    pool, service = pool_and_service
    with pool.episode(1) as episode:
        env_idx = episode.env_idx
    assert all(not leased for leased in service.live.values())
    released = [p for _, p in service.calls if p.get("action") == "release_one"]
    assert released == [{"action": "release_one", "env_idx": env_idx}]


def test_episode_releases_lease_when_body_raises(pool_and_service):
    """回合内抛异常也必须归还，否则 slot 会持续泄漏直到池子枯竭。"""
    pool, service = pool_and_service
    with pytest.raises(RuntimeError, match="boom"):
        with pool.episode(1):
            raise RuntimeError("boom")

    assert all(not leased for leased in service.live.values())
    assert any(p.get("action") == "release_one" for _, p in service.calls)


def test_concurrent_episodes_never_exceed_per_worker_slots(pool_and_service):
    """并发远超容量时应排队，而不是超发租约。"""
    pool, service = pool_and_service

    def run(idx: int) -> bool:
        with pool.episode(idx) as episode:
            episode.interact("search[x]")
            return True

    with ThreadPoolExecutor(max_workers=32) as executor:
        assert all(executor.map(run, range(64)))

    assert service.peak_live, "no episode ever leased a slot"
    assert max(service.peak_live.values()) <= pool._slots_per_worker
    assert all(not leased for leased in service.live.values())


def test_concurrent_episodes_spread_across_workers(pool_and_service):
    """轮询应该真的用上所有 worker，否则多进程等于白起。"""
    pool, service = pool_and_service

    barrier = threading.Barrier(4)

    def run(idx: int) -> None:
        with pool.episode(idx):
            barrier.wait(timeout=10)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run, range(4)))

    assert len(service.peak_live) == 4, f"only used {sorted(service.peak_live)}"


def test_reset_without_env_idx_is_an_error():
    pool = EnvironmentPool(workers=1, slots_per_worker=1)
    pool._post = lambda url, payload: {"instruction": "x"}  # type: ignore[method-assign]
    with pytest.raises(EnvironmentServiceError, match="no env_idx"):
        with pool.episode(0):
            pass


def test_slot_is_returned_even_if_reset_fails():
    """reset 失败也要放回信号量，否则连续失败会把池子锁死。"""
    pool = EnvironmentPool(workers=1, slots_per_worker=1)

    def failing_post(url, payload):
        raise EnvironmentServiceError("service down")

    pool._post = failing_post  # type: ignore[method-assign]
    for _ in range(3):
        with pytest.raises(EnvironmentServiceError):
            with pool.episode(0):
                pass
    # 信号量已满则再次 acquire 必须成功
    assert pool._slots[pool.urls[0]].acquire(blocking=False)


def test_urls_can_be_supplied_directly():
    pool = EnvironmentPool(urls=["http://h:1/api", "http://h:2/api"], slots_per_worker=2)
    assert pool.urls == ["http://h:1/api", "http://h:2/api"]
    assert pool.capacity == 4


def test_empty_url_list_is_rejected():
    with pytest.raises(ValueError, match="at least one worker"):
        EnvironmentPool(urls=[])
