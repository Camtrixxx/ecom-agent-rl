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
    """记录每个 url 收到的请求，并模拟每 worker 的 env_idx 分配。

    `release_one` 的两种 message 逐字照抄 `pack_api.py`：客户端靠它区分「真的回收了
    一个漏掉的租约」和「本来就是空的」，抄错了这里的测试就失去意义。
    """

    def __init__(
        self,
        slots_per_worker: int = 4,
        drop_reset_responses: int = 0,
        on_release=None,
    ) -> None:
        self.slots_per_worker = slots_per_worker
        # 模拟泄漏的根因：服务端在 reset 里已经 acquire 了 slot，响应却没能回到客户端。
        # 客户端因此永远不知道编号，那个 slot 再也没人归还得了。
        self.drop_reset_responses = drop_reset_responses
        self.on_release = on_release
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
                if self.drop_reset_responses > 0:
                    self.drop_reset_responses -= 1
                    raise EnvironmentServiceError(f"{url}: URLError: connection reset")
                return {
                    "env_idx": free,
                    "instruction": f"task {payload['idx']}",
                    "observation_state": "page 1",
                }

            if action == "interact":
                return {"observation_state": "page 2", "over": False, "done": False}

            if action == "release_one":
                env_idx = payload["env_idx"]
                if self.on_release is not None:
                    self.on_release(url, env_idx)
                if env_idx in leased:
                    leased.discard(env_idx)
                    return {"message": f"Environment {env_idx} has been released"}
                return {"message": f"Environment {env_idx} is already free"}

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


def test_a_lost_reset_response_is_reclaimed_and_the_episode_still_runs():
    """泄漏根因的端到端复现：服务端已 acquire、响应没回来，客户端拿不到编号。

    实测一轮长跑后 32 个 slot 漏了 25 个，有效容量掉到 7，表现是「并发没超也报
    env_error」。这里只给 1 个 slot，所以重试能成功**只可能**是因为那个漏掉的租约
    真的被收回来了——修复前这一路是死的。
    """
    service = FakeService(slots_per_worker=1, drop_reset_responses=1)
    pool = EnvironmentPool(workers=1, slots_per_worker=1)
    pool._post = service.post  # type: ignore[method-assign]
    url = pool.urls[0]

    with pool.episode(0) as episode:
        assert episode.env_idx == 0

    actions = [payload["action"] for _, payload in service.calls]
    assert actions.count("reset") == 2, f"没有重试: {actions}"
    assert service.live[url] == set()
    assert pool._owned[url] == set()


def test_reclaim_skips_slots_this_pool_is_using():
    """回收只碰自己不持有的编号。放掉在飞回合的 slot 会让两个回合共用一个 env，
    观测互相串台——比泄漏隐蔽得多，所以宁可漏收一轮。"""
    service = FakeService(slots_per_worker=4)
    pool = EnvironmentPool(workers=1, slots_per_worker=4)
    pool._post = service.post  # type: ignore[method-assign]
    url = pool.urls[0]

    with pool.episode(1) as episode:
        service.live[url].add(3)  # 注入一个孤儿
        assert pool._reclaim_orphans(url) == [3]
        assert episode.env_idx in service.live[url]
    assert service.live[url] == set()


def test_reclaim_is_locked_out_while_a_reset_is_in_flight():
    """服务端已分配、客户端还没登记的那个窗口里，回收必须被挡在外面。

    没有这把锁，`_owned` 这本账就有个天窗：回收线程看到一个刚分配、尚未登记的编号，
    会判成孤儿放掉。代价不是报错而是静默串数据，所以这条不变量单独钉住。
    """
    service = FakeService(slots_per_worker=2)
    pool = EnvironmentPool(workers=1, slots_per_worker=2)
    url = pool.urls[0]
    inside_reset = threading.Event()
    let_reset_finish = threading.Event()

    def slow_post(target: str, payload: dict) -> dict:
        if payload.get("action") == "reset":
            inside_reset.set()
            assert let_reset_finish.wait(timeout=5.0)
        return service.post(target, payload)

    pool._post = slow_post  # type: ignore[method-assign]
    done = threading.Event()

    def run() -> None:
        with pool.episode(0):
            done.wait(timeout=5.0)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert inside_reset.wait(timeout=5.0)
    assert not pool._reset_locks[url].acquire(blocking=False), "回收能挤进 reset 窗口"

    let_reset_finish.set()
    done.set()
    thread.join(timeout=5.0)
    assert not thread.is_alive()


def test_reclaim_never_steals_a_lease_under_concurrency():
    """并发跑回合的同时不断有 reset 丢响应，回收不能碰到任何在飞的租约。"""
    service = FakeService(slots_per_worker=2, drop_reset_responses=12)
    pool = EnvironmentPool(workers=4, slots_per_worker=2)

    in_flight: set[tuple[str, int]] = set()
    in_flight_lock = threading.Lock()
    stolen: list[tuple[str, int]] = []

    def on_release(url: str, env_idx: int) -> None:
        with in_flight_lock:
            if (url, env_idx) in in_flight:
                stolen.append((url, env_idx))

    service.on_release = on_release
    pool._post = service.post  # type: ignore[method-assign]

    def run(idx: int) -> None:
        try:
            with pool.episode(idx) as episode:
                key = (episode._url, episode.env_idx)
                with in_flight_lock:
                    in_flight.add(key)
                try:
                    episode.interact("search[x]")
                finally:
                    with in_flight_lock:
                        in_flight.discard(key)
        except EnvironmentServiceError:
            pass  # 丢响应的那几条本来就该失败，这里验的是别人不受牵连

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(run, range(96)))

    assert not stolen, f"回收放掉了在飞的租约: {stolen}"
    assert all(not owned for owned in pool._owned.values()), pool._owned
    # 回收是**按需**发生的：只有下一次 reset 撞墙才会触发，所以最后一批丢响应留下的
    # 租约还挂着。这是有意的——池子只在真的不够用时才去动服务端状态。显式 reconcile
    # 必须能把它们全部收干净，否则「按需」就变成了「永远收不回」。
    leftover = {url: sorted(v) for url, v in service.live.items() if v}
    assert leftover, "这批负载没造出泄漏，测试前提失效"
    pool.reconcile()
    assert all(not leased for leased in service.live.values()), service.live


def test_reconcile_reports_what_it_reclaimed():
    service = FakeService(slots_per_worker=4)
    pool = EnvironmentPool(workers=2, slots_per_worker=4)
    pool._post = service.post  # type: ignore[method-assign]
    first, second = pool.urls
    service.live[first] = {0, 2}
    service.live[second] = set()

    assert pool.reconcile() == {first: [0, 2], second: []}
    assert service.live[first] == set()


def test_a_failed_release_does_not_permanently_claim_the_slot():
    """归还请求本身失败时服务端可能仍持有租约，客户端必须销账，
    好让它落进下一次回收的范围——继续记着等于给这个 slot 永久豁免。"""
    service = FakeService(slots_per_worker=2)
    pool = EnvironmentPool(workers=1, slots_per_worker=2)
    url = pool.urls[0]

    def post(target: str, payload: dict) -> dict:
        if payload.get("action") == "release_one":
            raise EnvironmentServiceError("service down")
        return service.post(target, payload)

    pool._post = post  # type: ignore[method-assign]
    with pool.episode(0) as episode:
        leaked = episode.env_idx
    assert pool._owned[url] == set()

    pool._post = service.post  # type: ignore[method-assign]
    assert pool.reconcile()[url] == [leaked]


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
