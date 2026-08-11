"""跨多个 ShopSimulator 进程的客户端。

环境服务单进程被 GIL 锁死在 ~5 episodes/s（docs/environment-notes.md），扩展只能
靠多进程。但会话状态在进程内：`reset` 返回的 `env_idx` 只对分配它的那个进程有意义，
所以不能用无状态轮询——一个回合的所有请求必须打到同一个端口。

`EnvironmentPool.episode()` 负责这件事：进入时挑一个 worker 并锁定，退出时归还租约。
rollout、评测、教师采集三处都应通过它访问环境，这样并发上限只有一处定义。
"""

from __future__ import annotations

import itertools
import json
import logging
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

logger = logging.getLogger(__name__)

DEFAULT_BASE_PORT = 5700
DEFAULT_WORKERS = 8
DEFAULT_HOST = "127.0.0.1"

# `release_one` 对「本来就租着」和「本来就空着」返回不同的 message，这是服务端唯一
# 暴露租约状态的地方（`SlotLeasePool.free_slots()` 没有 HTTP 出口）。用它来数「实际
# 回收了几个」。措辞变了只会让计数偏保守（数成 0 → 不重试 → 退回旧行为报错），
# 不会误放在飞的 slot——放不放由 `_owned` 决定，与这里的解析无关。
_RELEASED_MARKER = "has been released"

# 环境服务在本机，必须绕开系统代理，否则请求会被转发出去而永远到不了 Flask。
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class EnvironmentServiceError(RuntimeError):
    """环境服务返回了 error，或请求无法完成。"""


@dataclass(frozen=True)
class Step:
    """一次 interact 的结果。"""

    observation: str
    over: bool
    done: bool
    raw: dict[str, Any]

    @property
    def reward(self) -> float | None:
        """回合未结束时环境不给 reward。评测侧须屏蔽此字段，见 blind guard。"""
        value = self.raw.get("reward")
        return float(value) if isinstance(value, (int, float)) else None


class EnvironmentPool:
    """把 N 个单进程环境服务当成一个池子用。

    线程安全：`episode()` 可被任意多个线程并发调用，最多 `len(urls) * slots_per_worker`
    个回合同时在飞。超出的调用会阻塞等待，而不是失败。
    """

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        base_port: int = DEFAULT_BASE_PORT,
        workers: int = DEFAULT_WORKERS,
        slots_per_worker: int = 4,
        timeout: float = 120.0,
        urls: list[str] | None = None,
        env_slots: int | None = None,
    ) -> None:
        if urls is None:
            urls = [
                f"http://{host}:{base_port + i}/api/shop_agent"
                for i in range(workers)
            ]
        if not urls:
            raise ValueError("environment pool needs at least one worker url")

        self._urls = list(urls)
        self._timeout = timeout
        self._slots_per_worker = slots_per_worker
        # 服务端每个 worker 的 slot 数（`SHOPSIM_ENV_SLOTS`）。回收孤儿租约要按索引
        # 逐个试 `release_one`，所以必须知道索引空间有多大。默认与客户端信号量同宽：
        # 两者不一致时客户端会超配或欠配，那是配置问题，不该由默认值掩盖。
        self._env_slots = slots_per_worker if env_slots is None else env_slots
        # 每个 worker 一个信号量，容量即该进程的 slot 数。租约不足时阻塞而非报错，
        # 因为上游的 rollout 批次通常远大于池子容量。
        self._slots = {
            url: threading.BoundedSemaphore(slots_per_worker) for url in self._urls
        }
        # 本池当前持有的 env_idx。回收孤儿时用它区分「别人在飞」和「服务端漏了」——
        # 少了这份账本就只能用 release_all，那会连在飞的回合一起清掉。
        self._owned: dict[str, set[int]] = {url: set() for url in self._urls}
        self._owned_lock = threading.Lock()
        # 每个 worker 一把 reset 锁，见 `_open_episode` 的推导。
        self._reset_locks = {url: threading.Lock() for url in self._urls}
        self._cycle = itertools.cycle(self._urls)
        self._cycle_lock = threading.Lock()

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    @property
    def capacity(self) -> int:
        """可同时在飞的回合数**硬上限**，不是推荐并发值。

        实测最优工作点是「并发 = worker 数」：单进程被 GIL 锁在 1 核，加 slot 不提升
        吞吐（见 docs/environment-notes.md）。按 capacity 设并发会超配 slots_per_worker
        倍，只是让请求在服务端排队。`batch.py` 因此默认取 `len(pool.urls)`。
        """
        return len(self._urls) * self._slots_per_worker

    # ---------------------------------------------------------------- transport

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with _OPENER.open(request, timeout=self._timeout) as response:
                parsed = json.load(response)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise EnvironmentServiceError(f"{url}: {type(exc).__name__}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise EnvironmentServiceError(f"{url}: malformed json: {exc}") from exc

        result = parsed.get("result")
        if not isinstance(result, dict):
            raise EnvironmentServiceError(f"{url}: unexpected payload: {parsed!r}")
        if "error" in result:
            raise EnvironmentServiceError(f"{url}: {result['error']}")
        return result

    # ------------------------------------------------------------------ health

    def _probe(self, url: str) -> None:
        """就绪探测：只连 TCP，不发请求。

        `pack_api.py` 先跑完 `initialize_environments()` 再 `app.run()`，所以
        「端口在监听」已经等价于「env 已就绪」，connect 成功即充分。

        这里不能用 `release_all` 探活：服务端会 `slot_pool.reset()`，清掉该 worker
        上**全部**租约。并行跑 ablation 时若两个实验撞上同一端口段，后启动的一方
        会静默 reset 掉前一方所有在飞回合。探测必须是只读的。
        """
        host, _, port = urllib.parse.urlsplit(url).netloc.rpartition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=self._timeout):
                return
        except (OSError, ValueError) as exc:
            raise EnvironmentServiceError(f"{url}: {type(exc).__name__}: {exc}") from exc

    def wait_until_ready(self, timeout: float = 600.0) -> None:
        """等所有 worker 就绪。初始化 env 期间端口未监听，必须先等。"""
        deadline = time.monotonic() + timeout
        pending = list(self._urls)
        last_error: Exception | None = None
        while pending and time.monotonic() < deadline:
            still_pending = []
            for url in pending:
                try:
                    self._probe(url)
                except EnvironmentServiceError as exc:
                    last_error = exc
                    still_pending.append(url)
            pending = still_pending
            if pending:
                time.sleep(2.0)
        if pending:
            raise EnvironmentServiceError(
                f"{len(pending)} worker(s) not ready after {timeout:.0f}s: "
                f"{pending} (last error: {last_error})\n"
                f"start them with: bash scripts/start_environment.sh"
            )

    # ----------------------------------------------------------------- episodes

    def _pick(self) -> str:
        with self._cycle_lock:
            return next(self._cycle)

    @contextmanager
    def episode(self, task_idx: int) -> Iterator["Episode"]:
        """租一个 slot 跑一个回合，退出时保证归还。

        选到的 worker 满了就换下一个；全满则阻塞在第一个上等待。
        """
        url = self._acquire_slot()
        try:
            env_idx, result = self._open_episode(url, task_idx)
            episode = Episode(self, url, env_idx, task_idx, result)
            try:
                yield episode
            finally:
                self._release_env(url, env_idx)
        finally:
            self._slots[url].release()

    def _open_episode(self, url: str, task_idx: int) -> tuple[int, dict[str, Any]]:
        """在 url 上 reset 出一个回合；撞上租约耗尽时先回收孤儿再重试一次。

        泄漏的成因：服务端在 `reset` 的处理里就 `slot_pool.acquire()`，而客户端要等
        响应回来才知道 `env_idx`。这中间任何一次超时、连接断开或 5xx，都会留下一个
        「服务端认为租出去了、没有任何人知道编号」的 slot，且永不回收——`release_one`
        只认编号。实测一轮长跑后 32 个 slot 漏掉 25 个（见 docs/environment-notes.md）。

        所以「要 slot」和「把编号记进 `_owned`」必须在同一把锁里做完。否则回收线程会
        看到一个刚被服务端分配、还没登记的编号，判成孤儿放掉——两个回合共用一个 env，
        观测互相串台，比泄漏隐蔽得多。锁的代价接近零：服务端被 GIL 锁在单核，同一个
        worker 的请求本来就是串行的。

        只重试一次：回收确实放出了 slot 才重试，否则原样抛出。真的容量不足时重试
        只是把 25 秒的服务端退避（`MAX_RETRIES=5` × `RETRY_DELAY_SECONDS=5`）翻倍。
        """
        with self._reset_locks[url]:
            try:
                return self._reset_once(url, task_idx)
            except EnvironmentServiceError as first_error:
                reclaimed = self._reclaim_orphans(url)
                if not reclaimed:
                    raise
                logger.warning(
                    "reset failed on %s (%s); reclaimed %d leaked slot(s) %s and retrying",
                    url,
                    first_error,
                    len(reclaimed),
                    sorted(reclaimed),
                )
            return self._reset_once(url, task_idx)

    def _reset_once(self, url: str, task_idx: int) -> tuple[int, dict[str, Any]]:
        """必须在 `_reset_locks[url]` 之下调用。"""
        result = self._post(url, {"action": "reset", "idx": task_idx})
        env_idx = result.get("env_idx")
        if env_idx is None:
            # 服务端已经 acquire 过了，但没告诉我们编号，归还不了。登记不下的孤儿
            # 留给下一次 `_reclaim_orphans` 收。
            raise EnvironmentServiceError(f"{url}: reset returned no env_idx")
        env_idx = int(env_idx)
        with self._owned_lock:
            self._owned[url].add(env_idx)
        return env_idx, result

    def _reclaim_orphans(self, url: str) -> list[int]:
        """回收「服务端认为租出去了、但本池并不持有」的 slot。必须持有该 url 的 reset 锁。

        逐个 `release_one` 而不是 `release_all`：后者是 `slot_pool.reset()`，会把该
        worker 上**全部**租约清掉，包括本池其他线程在飞的回合，以及并行 ablation 里
        别人的回合。跳过 `_owned` 里的编号，代价是漏收一轮，换来的是永远不会误放。

        前提是同一端口段上只有这一个客户端池（docs/environment-notes.md 的并发口径），
        否则别人持有的租约在这里会被当成孤儿。
        """
        with self._owned_lock:
            owned = set(self._owned[url])
        reclaimed: list[int] = []
        for idx in range(self._env_slots):
            if idx in owned:
                continue
            try:
                result = self._post(url, {"action": "release_one", "env_idx": idx})
            except EnvironmentServiceError as exc:
                logger.warning("failed to probe slot %s on %s: %s", idx, url, exc)
                continue
            if _RELEASED_MARKER in str(result.get("message", "")):
                reclaimed.append(idx)
        return reclaimed

    def measure_free_slots(self, url: str) -> list[int]:
        """量该 worker 现在实际还能租出几个 slot，拿到的当场还回去。

        刻意不走 `episode()`：那条路径在 reset 失败时会自动回收孤儿，等于体检顺手把病
        治了，于是永远量不到泄漏。这里要的是**现状**，修不修是 `reconcile()` 的事。

        必须一次试租满 `env_slots` 个。只试 1 个是测不出泄漏的——漏了 3 个、剩 1 个空位
        的 worker 照样会返回成功（docs/environment-notes.md 记的第一个诊断陷阱）。

        慢是正常的：满了的 worker 不快速失败，服务端 `MAX_RETRIES=5` ×
        `RETRY_DELAY_SECONDS=5` 会把请求挂 25 秒才回错误。调用方按「每个满 worker
        25 秒」估时，并且应该跨 worker 并行。
        """
        acquired: list[int] = []
        with self._reset_locks[url]:
            while len(acquired) < self._env_slots:
                try:
                    result = self._post(url, {"action": "reset", "idx": 0})
                except EnvironmentServiceError:
                    break
                env_idx = result.get("env_idx")
                if env_idx is None:
                    # 服务端已 acquire 却没给编号——正在制造一个新的孤儿。停手，
                    # 让它落进 `reconcile()` 的范围，而不是继续把池子掏空。
                    break
                acquired.append(int(env_idx))
            for env_idx in acquired:
                try:
                    self._post(url, {"action": "release_one", "env_idx": env_idx})
                except EnvironmentServiceError as exc:
                    logger.warning(
                        "probe failed to release env %s on %s: %s", env_idx, url, exc
                    )
        return acquired

    def reconcile(self) -> dict[str, list[int]]:
        """回收全池的孤儿租约，返回每个 url 收回了哪些编号。

        不在启动时自动跑：若同一端口段上真的有第二个客户端池，自动回收会把对方在飞的
        回合静默掐掉，而现在的表现是响亮的 `env_error` 中止。谁来跑、什么时候跑，交给
        `scripts/check_environment.py --reclaim` 显式决定。
        """
        found: dict[str, list[int]] = {}
        for url in self._urls:
            with self._reset_locks[url]:
                found[url] = self._reclaim_orphans(url)
        return found

    def _acquire_slot(self) -> str:
        """非阻塞地扫一圈找空闲 worker；都满了就轮询等**任意**一个空出来。

        不能在单个 worker 上死等：`_pick` 是轮转，轮到谁就等谁，而那个 worker 可能
        正在跑一条 35 步的长回合，同时别的 worker 早已空闲。表现是尾部延迟变长而
        不报任何错，很难查。默认并发 = worker 数时跑不到这条路径，但教师采集把并发
        调高就会踩到。

        用短超时轮询而不是 Condition：slot 是 BoundedSemaphore，跨 worker 等「任意
        一个」需要额外的同步原语，而这条路径本就是过载兜底，10ms 的轮询代价可忽略。
        """
        for _ in range(len(self._urls)):
            url = self._pick()
            if self._slots[url].acquire(blocking=False):
                return url
        while True:
            for _ in range(len(self._urls)):
                url = self._pick()
                if self._slots[url].acquire(timeout=0.01):
                    return url

    def _release_env(self, url: str, env_idx: int) -> None:
        try:
            self._post(url, {"action": "release_one", "env_idx": env_idx})
        except EnvironmentServiceError as exc:
            # 归还失败只影响后续该 worker 的可用 slot 量，不该掩盖回合本身的错误。
            logger.warning("failed to release env %s on %s: %s", env_idx, url, exc)
        finally:
            # 无论归还成功与否都要销账。归还失败时服务端可能仍持有租约，销账正是要
            # 让它落进下一次 `_reclaim_orphans` 的范围；继续记着反而是永久豁免。
            with self._owned_lock:
                self._owned[url].discard(env_idx)


class Episode:
    """一个已 reset 的回合，绑定在特定 worker 与 env_idx 上。"""

    def __init__(
        self,
        pool: EnvironmentPool,
        url: str,
        env_idx: int,
        task_idx: int,
        reset_result: dict[str, Any],
    ) -> None:
        self._pool = pool
        self._url = url
        self.env_idx = env_idx
        self.task_idx = task_idx
        self.reset_result = reset_result
        self.steps = 0

    @property
    def instruction(self) -> str:
        return self.reset_result.get("instruction", "")

    @property
    def observation(self) -> str:
        return self.reset_result.get("observation_state", "")

    def interact(self, response: str) -> Step:
        result = self._pool._post(
            self._url,
            {"action": "interact", "env_idx": self.env_idx, "response": response},
        )
        self.steps += 1
        return Step(
            observation=result.get("observation_state", ""),
            over=bool(result.get("over")),
            done=bool(result.get("done")),
            raw=result,
        )
