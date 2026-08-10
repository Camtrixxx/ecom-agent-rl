"""EnvironmentPool 的租约与粘连语义。

不起真实服务：把 `_post` 换成假的，因为要验的是「回合内粘连到同一 worker」和
「租约必然归还」这两条不变量，它们与 HTTP 层无关。
"""

from __future__ import annotations

import socket
import threading
import time
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


def test_wait_until_ready_sends_no_requests(pool_and_service):
    """就绪探测必须只读，且走真实 `_probe`（不能 stub，否则测不到东西）。

    曾用 `release_all` 探活，而服务端会 `slot_pool.reset()` 清掉该 worker 上全部
    租约——并行 ablation 撞同一端口段时，会静默 reset 掉别的实验在飞的回合。
    """
    _, service = pool_and_service
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        pool = EnvironmentPool(base_port=port, workers=1, timeout=5.0)
        pool._post = service.post  # type: ignore[method-assign]
        pool.wait_until_ready(timeout=5.0)

    assert service.calls == [], f"probe sent requests: {service.calls}"


def test_probe_rejects_a_port_with_nothing_listening():
    """端口未监听即未就绪；`pack_api.py` 先建好 env 才 app.run()，所以监听即就绪。"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # 绑了但没 listen，connect 会被拒

    pool = EnvironmentPool(base_port=port, workers=1, timeout=1.0)
    with pytest.raises(EnvironmentServiceError, match="not ready"):
        pool.wait_until_ready(timeout=1.0)


def test_urls_can_be_supplied_directly():
    pool = EnvironmentPool(urls=["http://h:1/api", "http://h:2/api"], slots_per_worker=2)
    assert pool.urls == ["http://h:1/api", "http://h:2/api"]
    assert pool.capacity == 4


def test_empty_url_list_is_rejected():
    with pytest.raises(ValueError, match="at least one worker"):
        EnvironmentPool(urls=[])


def test_a_full_pool_takes_whichever_worker_frees_up_first():
    """过载时不能锁在轮到的那个 worker 上死等。

    `_pick` 是轮转，轮到谁就等谁；而那个 worker 可能正在跑一条 35 步长回合，别的
    worker 早就空了。这个 bug 不报错，只让尾部延迟变长——所以必须有测试守着。
    这里把池占满，只放开**最后**一个 worker 的 slot，看它能不能拿到。
    """
    service = FakeService(slots_per_worker=1)
    pool = EnvironmentPool(workers=3, slots_per_worker=1)
    pool._post = service.post  # type: ignore[method-assign]

    # 占满所有 worker。
    held = [pool._slots[url] for url in pool.urls]
    for slot in held:
        assert slot.acquire(blocking=False)

    freed = pool.urls[-1]
    result: list[str] = []

    def take() -> None:
        result.append(pool._acquire_slot())

    thread = threading.Thread(target=take, daemon=True)
    thread.start()
    time.sleep(0.05)  # 让它进入等待路径
    assert not result, "池已满时不该拿到 slot"

    pool._slots[freed].release()
    thread.join(timeout=2.0)
    assert result == [freed], "空出来的那个 worker 没被取到（可能死等在别的 worker 上）"
