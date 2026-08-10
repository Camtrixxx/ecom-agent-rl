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
        # 每个 worker 一个信号量，容量即该进程的 slot 数。租约不足时阻塞而非报错，
        # 因为上游的 rollout 批次通常远大于池子容量。
        self._slots = {
            url: threading.BoundedSemaphore(slots_per_worker) for url in self._urls
        }
        self._cycle = itertools.cycle(self._urls)
        self._cycle_lock = threading.Lock()

    @property
    def urls(self) -> list[str]:
        return list(self._urls)

    @property
    def capacity(self) -> int:
        """可同时在飞的回合数上限。"""
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
            result = self._post(url, {"action": "reset", "idx": task_idx})
            env_idx = result.get("env_idx")
            if env_idx is None:
                raise EnvironmentServiceError(f"{url}: reset returned no env_idx")
            episode = Episode(self, url, int(env_idx), task_idx, result)
            try:
                yield episode
            finally:
                self._release_env(url, int(env_idx))
        finally:
            self._slots[url].release()

    def _acquire_slot(self) -> str:
        """非阻塞地扫一圈找空闲 worker；都满了就阻塞等一个。"""
        for _ in range(len(self._urls)):
            url = self._pick()
            if self._slots[url].acquire(blocking=False):
                return url
        url = self._pick()
        self._slots[url].acquire()
        return url

    def _release_env(self, url: str, env_idx: int) -> None:
        try:
            self._post(url, {"action": "release_one", "env_idx": env_idx})
        except EnvironmentServiceError as exc:
            # 归还失败只影响后续该 worker 的可用 slot 量，不该掩盖回合本身的错误。
            logger.warning("failed to release env %s on %s: %s", env_idx, url, exc)


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
