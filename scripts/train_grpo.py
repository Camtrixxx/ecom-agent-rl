#!/usr/bin/env python3
"""7B 全参 GRPO，采样与训练在同一个进程组里轮流做。

用法（必须用 accelerate 起）：
    bash scripts/train_grpo.sh

    # 只验链路
    bash scripts/train_grpo.sh --iterations 1 --tasks-per-iteration 4 --group-size 4

产物：
    <out-dir>/policy/           当前策略的 HF 权重，vLLM 直接 serve 这个目录
    <out-dir>/rollouts/iter*.jsonl  每轮的采样轨迹（原始，可复算奖励）
    <out-dir>/train_log.jsonl   每轮的回报、优势、loss、耗时
    <out-dir>/state.json        断点（--resume 从这里续）

## 结构：为什么是「一个进程组，两个阶段轮流」

采样要走 HTTP 打 vLLM，是单线程 IO 密集；训练是 6 卡 FSDP 集合通信。两者不能同时
用同一批卡，所以按轮交替：rank 0 采样（其余 rank 等在屏障上）→ 全员训练 → 导权重 →
重启 vLLM。采样结果通过**文件**广播给所有 rank：轨迹本来就要落盘留证据，用它当广播
通道比 `broadcast_object_list` 少一套序列化，也让每轮的输入可复现。

## 权重同步用「重启 vLLM」而不是 RPC 热更

热更（`collective_rpc("update_named_param")` 那条路）省掉每轮约 100 秒的重启，但要
求训练侧的参数名与 vLLM 内部布局逐个对上，装一套额外的 NCCL 通信组，而且**更新失败
是静默的**——服务照样响应，只是权重还是旧的，表现为「训练在动、采样分数不动」，极难
查。重启是幂等的：vLLM 从目录读权重，起来了就一定是新的。

这一轮跑不追求峰值效率，追求无人值守十几个小时不出事，所以选重启。每轮开销约
导权重 + 重启，占比在下面的日志里逐轮打出来，日后要换热更时有基准可比。

## loss：μ=1，所以没有 ratio 也没有 clip

每批数据只做一次梯度更新（`μ=1`），此时 `πθ = πold`，重要性比恒为 1、clip 恒不触发。
把这套机器写进来只会增加一次前向（算 old logprob）和一堆恒等于 1 的乘法，以及对应的
出错面。所以这里直接写它的极限形式：

    loss = Σ_{i,t} A_i · CE(θ)_{i,t} / Σ_{i,t} 1

`CE = -log πθ`，所以最小化它等价于最大化 `Σ A_i log πθ`——就是带组内基线的策略梯度。
好处是可以原样复用 SFT 那段分块 fp32 上采的 CE（vocab 152064，整体上采会炸显存），
只是把 `reduction="sum"` 换成 `"none"` 再按 token 权重加权。

**归一化按 token 数而不是按序列数**：按序列平均会让长回合的每个 token 权重更小，而
长回合恰恰是 `repeat_loop` 这类要压的负样本，等于给要罚的东西打折。

## 没有 KL 惩罚，也没有参考模型

省掉一个常驻的 7B（分到 6 卡上约 2.3G/卡）和每步一次额外前向（约 40% 的步时）。代价
是没有对 SFT 权重的显式约束，靠小学习率（1e-6）+ 梯度裁剪 + 每轮存档来兜。这是 DAPO
一系的做法，在从一个已经很强的 SFT 起点做短程 RL 时是常见取舍。

它不是免费的：策略可能漂。所以每轮都记录组均回报与终局类型分布，漂了在日志里是看得
见的（回报掉、`repeat_loop` 涨），而每轮的权重快照让回退到任意一轮都可行。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--init-model", type=Path,
                        default=ROOT / "outputs" / "models" / "sft",
                        help="起点权重，默认 SFT 的产物")
    parser.add_argument("--pool", type=Path,
                        default=ROOT / "data" / "task_pools" / "grpo_train.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "models" / "grpo")

    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--tasks-per-iteration", type=int, default=24, help="每轮抽几道题")
    parser.add_argument("--group-size", type=int, default=8, help="每题采几条，即 GRPO 的 G")
    parser.add_argument("--rollout-concurrency", type=int, default=32)
    parser.add_argument("--rollout-retries", type=int, default=3,
                        help="一轮采样被基础设施失败中止后重试几次（续跑，不重采已完成的）")

    parser.add_argument("--lr", type=float, default=1e-6, help="RL 阶段比 SFT 低一个量级")
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--invalid-reward", type=float, default=-1.0,
                        help="没走到终局的回报；默认严格低于 wrong_purchase 的 −0.85")
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--loss-chunk-tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--snapshot-every", type=int, default=5,
                        help="每 N 轮额外留一份不被覆盖的权重快照（0 表示不留）")
    parser.add_argument("--resume", action="store_true", help="从 <out-dir>/state.json 续跑")

    # vLLM 侧
    parser.add_argument("--vllm-gpus", default="1", help="给 vLLM 的卡，逗号分隔")
    parser.add_argument("--vllm-port", type=int, default=8180)
    parser.add_argument("--vllm-startup-timeout", type=float, default=900.0)
    parser.add_argument("--context-window", type=int, default=24576)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="RL 采样要比评测更散，否则组内没有方差可学")
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=1024)

    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-base-port", type=int, default=5700)
    parser.add_argument("--env-workers", type=int, default=8)
    parser.add_argument("--env-slots", type=int, default=4)
    return parser.parse_args()


# --------------------------------------------------------------------------- vLLM 生命周期


def port_is_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket() as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


# torchrun 注入的分布式变量。vLLM 是 `accelerate launch` 的孙进程，默认会把它们全部
# 继承过去，而其中 `TORCHELASTIC_USE_AGENT_STORE=True` 是致命的：torch 的
# `_create_c10d_store` 见到它就认定 TCPStore 由 elastic agent 提供，于是 **不启 daemon**
# ——vLLM 那个 world_size=1 的进程组只当 client 去连自己随机选的端口，那里没人监听，
# 于是死等到 600s 超时。日志里看到的是 "client socket has timed out"，和 vLLM 本身
# 毫无关系，独立跑 serve_model.sh 又完全正常，极难对上。
#
# RANK / WORLD_SIZE / MASTER_* 同理会让 vLLM 误判自己的拓扑，一并剥掉。
_TORCHRUN_ENV_PREFIXES = ("TORCHELASTIC_", "TORCH_NCCL_", "ACCELERATE_", "FSDP_")
_TORCHRUN_ENV_KEYS = frozenset({
    "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
    "GROUP_RANK", "GROUP_WORLD_SIZE", "ROLE_RANK", "ROLE_WORLD_SIZE", "ROLE_NAME",
    "MASTER_ADDR", "MASTER_PORT", "TORCH_DISTRIBUTED_DEBUG", "OMP_NUM_THREADS",
})


def clean_torchrun_env(source: Mapping[str, str]) -> dict[str, str]:
    """复制一份环境，剥掉 torchrun/accelerate 注入的分布式变量。"""
    return {
        key: value
        for key, value in source.items()
        if key not in _TORCHRUN_ENV_KEYS
        and not key.startswith(_TORCHRUN_ENV_PREFIXES)
    }


class VllmServer:
    """rank 0 独占的 vLLM 子进程。每轮换权重就是杀掉重起。

    必须 `start_new_session=True` 起进程组再 `killpg`：vLLM 会 fork 出 EngineCore
    子进程，只杀 shell 会留下占着显存的孤儿，下一轮起服务就会 OOM——而且报的是
    「显存不足」，看不出真因是上一轮没收干净。
    """

    def __init__(self, *, gpus: str, port: int, log_path: Path,
                 context_window: int, timeout: float) -> None:
        self.gpus = gpus
        self.port = port
        self.log_path = log_path
        self.context_window = context_window
        self.timeout = timeout
        self.process: subprocess.Popen[bytes] | None = None

    def start(self, model_dir: Path) -> float:
        if self.process is not None:
            raise RuntimeError("vLLM 已在运行，先 stop")
        self._wait_port_free()
        env = clean_torchrun_env(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.gpus
        env["LLM_PORT"] = str(self.port)
        env["MAX_MODEL_LEN"] = str(self.context_window)
        env["TP_SIZE"] = "1"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            ["bash", str(ROOT / "scripts" / "serve_model.sh"), str(model_dir)],
            cwd=str(ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self._wait_ready()
        return time.monotonic() - started

    def _wait_port_free(self) -> None:
        deadline = time.monotonic() + 120.0
        while port_is_open(self.port):
            if time.monotonic() > deadline:
                raise RuntimeError(f"端口 {self.port} 一直被占，上一轮的 vLLM 没收干净")
            time.sleep(2.0)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                tail = self._log_tail()
                raise RuntimeError(f"vLLM 启动即退出（码 {self.process.returncode}）：\n{tail}")
            if port_is_open(self.port):
                # 端口通了不等于能推理：权重可能还在加载。探一下 /v1/models。
                if self._models_ok():
                    return
            time.sleep(3.0)
        raise RuntimeError(f"vLLM {self.timeout}s 内没就绪：\n{self._log_tail()}")

    def _models_ok(self) -> bool:
        import urllib.error
        import urllib.request

        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(f"http://127.0.0.1:{self.port}/v1/models", timeout=5) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def _log_tail(self, lines: int = 30) -> str:
        if not self.log_path.exists():
            return "(无日志)"
        return "\n".join(
            self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
        )

    def stop(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=30)


# --------------------------------------------------------------------------- 采样


def iteration_tasks(
    all_tasks: Sequence[int], iteration: int, count: int, seed: int
) -> list[int]:
    """第 `iteration` 轮该跑哪几道题。

    先按 seed 把整池打散一次，再按轮次取连续切片并绕回。这样：同一轮里题目互不重复
    （重复会让两个「组」共享一道题，基线各算一份，等于给那道题双倍权重），跨轮尽量
    晚重复，而且只要 seed 与轮次相同就完全可复现——续跑不会换题。
    """
    order = list(all_tasks)
    random.Random(seed).shuffle(order)
    if not order:
        return []
    start = (iteration * count) % len(order)
    picked = [order[(start + i) % len(order)] for i in range(min(count, len(order)))]
    return picked


def sample_iteration(
    *, pool: Any, client: Any, task_ids: Sequence[int], output: Path,
    group_size: int, concurrency: int, retries: int, log: Any,
) -> dict[str, Any]:
    """跑一轮采样，带重试。返回最后一次的 summary。

    `run_batch` 撞上基础设施失败会中止整批（这是对的——继续跑只是在稳定地生产垃圾）。
    但无人值守时一次环境抖动不该终结整个训练，所以这里重试：已经落盘的回合不会重采，
    基础设施失败进 `.failures.jsonl` 也不占 `(task_id, attempt)`。

    `resume=True` 连**第一次**也要给。`_Writer` 是追加打开的，`resume=False` 并不会
    清空文件，只是不去读已完成的 `(task_id, attempt)`。于是整个进程崩掉后用 `--resume`
    重跑这一轮时，第一次调用会把上次已经落盘的回合再采一遍**追加**进去——同一个
    `(task_id, attempt)` 出现两条，组内基线被污染，而且看不出来。
    """
    from ecom_agent_rl.rollout.batch import run_batch

    summary: dict[str, Any] = {}
    for attempt in range(retries + 1):
        summary = run_batch(
            pool=pool, client=client, task_ids=list(task_ids), output=output,
            attempts=group_size, concurrency=concurrency, resume=True,
        )
        if not summary["aborted"]:
            return summary
        log(f"采样被中止（第 {attempt + 1} 次）：{summary['aborted']}")
        if attempt < retries:
            # 中止多半是 slot 泄漏把有效容量吃光了。这里回收是安全的，理由和
            # `check_environment.py --reclaim` 那个 caveat 正好相反：那边是新起的池子，
            # 账本是空的，别人的租约在它看来全是孤儿；这里用的是**同一个池对象**，
            # `_owned` 就是权威账本，而且此刻本池没有回合在飞。
            try:
                reclaimed = pool.reconcile()
                log(f"回收泄漏 slot：{reclaimed}")
            except Exception as exc:  # 回收失败不该盖住原始错误
                log(f"回收失败（继续重试）：{type(exc).__name__}: {exc}")
            time.sleep(10.0)
    return summary


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# --------------------------------------------------------------------------- 主流程


def main() -> None:
    args = parse_args()

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_constant_schedule_with_warmup

    sys.path.insert(0, str(ROOT / "scripts"))
    from train_sft import shard, shuffled, token_budget_batches  # noqa: E402
    from train_sft import save as save_weights  # noqa: E402

    from ecom_agent_rl.environment.pool import EnvironmentPool
    from ecom_agent_rl.environment.tools import TOOL_SCHEMAS
    from ecom_agent_rl.rollout.batch import load_task_ids
    from ecom_agent_rl.rollout.llm import ChatClient
    from ecom_agent_rl.training.grpo import (
        GroupStats, build_examples, collate, group_advantages, token_weights,
    )
    from ecom_agent_rl.training.sft_dataset import IGNORE_INDEX, RenderStats

    accelerator = Accelerator()
    set_seed(args.seed)
    main_process = accelerator.is_main_process

    def log(message: str) -> None:
        if main_process:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    if not args.init_model.exists():
        raise SystemExit(f"起点权重不存在: {args.init_model}")
    if not args.pool.exists():
        raise SystemExit(f"任务池不存在: {args.pool}")

    out_dir = args.out_dir
    policy_dir = out_dir / "policy"
    rollout_dir = out_dir / "rollouts"
    state_path = out_dir / "state.json"
    log_path = out_dir / "train_log.jsonl"
    if main_process:
        out_dir.mkdir(parents=True, exist_ok=True)
        rollout_dir.mkdir(parents=True, exist_ok=True)

    start_iteration = 0
    if args.resume and state_path.exists():
        start_iteration = int(json.loads(state_path.read_text(encoding="utf-8"))["iteration"])
        log(f"续跑：从第 {start_iteration} 轮开始")
    # 续跑时权重要从上一轮导出的 policy 目录起，不是从 SFT 起。
    load_from = policy_dir if (start_iteration > 0 and policy_dir.exists()) else args.init_model

    tokenizer = AutoTokenizer.from_pretrained(args.init_model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise SystemExit("tokenizer 没有 pad_token_id，无法成批")

    log(f"加载模型 {load_from}（attn={args.attn}）")
    model = AutoModelForCausalLM.from_pretrained(
        load_from, dtype=torch.bfloat16, attn_implementation=args.attn, use_cache=False
    )
    model.gradient_checkpointing_enable()

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    # 与 SFT 同样的两条：建在 prepare 之后，且不交给 prepare 包装（它会把内部计数
    # 每步推 num_processes 步）。RL 阶段用常数 lr——总步数由每轮的数据量决定，事先
    # 不知道，余弦退火没有可靠的 total_steps 可填。
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps
    )

    all_tasks = load_task_ids(args.pool)
    log(f"任务池 {len(all_tasks)} 题，每轮 {args.tasks_per_iteration} 题 × "
        f"{args.group_size} 条 = {args.tasks_per_iteration * args.group_size} 回合")

    env_pool = EnvironmentPool(
        host=args.env_host, base_port=args.env_base_port,
        workers=args.env_workers, slots_per_worker=args.env_slots,
    )
    server: VllmServer | None = None
    if main_process:
        env_pool.wait_until_ready(timeout=600.0)
        server = VllmServer(
            gpus=args.vllm_gpus, port=args.vllm_port,
            log_path=out_dir / "vllm.log",
            context_window=args.context_window, timeout=args.vllm_startup_timeout,
        )
        log(f"起 vLLM（GPU {args.vllm_gpus}，权重 {load_from}）")
        elapsed = server.start(load_from)
        log(f"vLLM 就绪，用时 {elapsed:.0f}s")

    client = ChatClient(
        base_url=f"http://127.0.0.1:{args.vllm_port}/v1",
        model="ecom-agent", api_key=os.environ.get("LLM_API_KEY"),
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens,
        context_window=args.context_window or None,
    )

    def forward_weighted(indices: Sequence[int], source: Sequence[dict[str, Any]]) -> tuple[Any, int]:
        """返回 (Σ A·CE, 动作 token 数)。

        分块上采 fp32 的理由与 SFT 完全一致（vocab 152064，整体 `.float()` 要 18.2G）。
        区别只在 `reduction="none"` 之后逐 token 乘优势权重再求和——权重在非动作
        token 上是 0，所以 padding 与 observation 不会贡献梯度。
        """
        batch = collate([source[i] for i in indices], pad_token_id=pad_token_id)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        weights = token_weights(labels, batch["advantages"].to(accelerator.device))

        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        flat_weights = weights.reshape(-1)

        total = flat_labels.numel()
        chunk = max(1, int(args.loss_chunk_tokens))
        loss = None
        for start in range(0, total, chunk):
            piece = torch.nn.functional.cross_entropy(
                flat_logits[start : start + chunk].float(),
                flat_labels[start : start + chunk],
                ignore_index=IGNORE_INDEX,
                reduction="none",
            )
            piece = (piece * flat_weights[start : start + chunk]).sum()
            loss = piece if loss is None else loss + piece
        if loss is None:
            loss = flat_logits.sum() * 0.0
        return loss, int((labels != IGNORE_INDEX).sum().item())

    history: list[dict[str, Any]] = []
    optimizer_steps = 0
    run_started = time.time()

    for iteration in range(start_iteration, args.iterations):
        iteration_started = time.time()
        task_ids = iteration_tasks(
            all_tasks, iteration, args.tasks_per_iteration, args.seed
        )
        rollout_path = rollout_dir / f"iter{iteration:03d}.jsonl"

        # ---- 阶段 1：采样（只有 rank 0 做，其余等在屏障上）----
        sample_seconds = 0.0
        rollout_summary: dict[str, Any] = {}
        if main_process:
            log(f"=== 第 {iteration} 轮：采样 {len(task_ids)} 题 × {args.group_size} 条")
            phase_started = time.time()
            rollout_summary = sample_iteration(
                pool=env_pool, client=client, task_ids=task_ids, output=rollout_path,
                group_size=args.group_size, concurrency=args.rollout_concurrency,
                retries=args.rollout_retries, log=log,
            )
            sample_seconds = time.time() - phase_started
            log(f"采样完成 {rollout_summary.get('written')} 条，{sample_seconds:.0f}s，"
                f"{rollout_summary.get('episodes_per_second')} ep/s")
        accelerator.wait_for_everyone()

        # ---- 阶段 2：优势与渲染（各 rank 各算一遍，纯 CPU，比广播便宜）----
        records = read_records(rollout_path)
        group_stats = GroupStats()
        render_stats = RenderStats()
        scored = group_advantages(
            records, invalid_reward=args.invalid_reward, stats=group_stats
        )
        examples = build_examples(
            scored, tokenizer, tools=TOOL_SCHEMAS,
            max_length=args.max_length, stats=group_stats, render_stats=render_stats,
        )
        if main_process:
            log(f"优势 {json.dumps(group_stats.to_dict(), ensure_ascii=False)} "
                f"渲染 {json.dumps(render_stats.to_dict(), ensure_ascii=False)}")

        batches = token_budget_batches(examples, args.tokens_per_batch)
        rank_batches = shard(
            shuffled(batches, args.seed, iteration),
            accelerator.process_index, accelerator.num_processes,
        )
        if len(rank_batches) < args.grad_accum:
            # 每卡不足一个累积组就没法凑出一步。跳过这一轮而不是崩——采样是有代价的，
            # 但这一轮的数据确实不够，硬跑会让某些 rank 空转在集合通信上。
            log(f"第 {iteration} 轮可用批数不足（每卡 {len(rank_batches)} < "
                f"grad_accum {args.grad_accum}），跳过更新")
            accelerator.wait_for_everyone()
            continue

        # ---- 阶段 3：更新 ----
        phase_started = time.time()
        model.train()
        step_losses: list[float] = []
        for start in range(0, len(rank_batches) - args.grad_accum + 1, args.grad_accum):
            group = rank_batches[start : start + args.grad_accum]
            local_tokens = sum(
                sum(1 for label in examples[i]["labels"] if label != IGNORE_INDEX)
                for indices in group for i in indices
            )
            step_tokens = accelerator.reduce(
                torch.tensor([local_tokens], device=accelerator.device, dtype=torch.float32),
                reduction="sum",
            ).item()
            if step_tokens == 0:
                continue
            loss_sum = 0.0
            for indices in group:
                loss, _ = forward_weighted(indices, examples)
                # 分母用全局动作 token 数：梯度是各卡求和的，这样等价于「对全世界的
                # 动作 token 取平均」，和 SFT 那边同一套口径。
                scaled = loss / step_tokens
                accelerator.backward(scaled)
                loss_sum += scaled.item()
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1
            step_losses.append(loss_sum)
        train_seconds = time.time() - phase_started

        # ---- 阶段 4：导权重并换掉 vLLM ----
        phase_started = time.time()
        # 先停服务再写权重，顺序不能反：从第 2 轮起 policy_dir 正是 vLLM 当前 serve
        # 的目录，而 safetensors 是 mmap 进来的。在它还开着的时候覆盖同名文件，映射
        # 页指向的内容会在它脚下被换掉——表现不是报错而是推理结果变成垃圾。
        if main_process and server is not None:
            server.stop()
        accelerator.wait_for_everyone()
        save_weights(accelerator, model, tokenizer, policy_dir, log)
        if args.snapshot_every and (iteration + 1) % args.snapshot_every == 0:
            save_weights(accelerator, model, tokenizer,
                         out_dir / f"iter{iteration:03d}", log)
        sync_seconds = time.time() - phase_started
        if main_process and server is not None:
            started_at = time.time()
            server.start(policy_dir)
            sync_seconds += time.time() - started_at
        accelerator.wait_for_everyone()

        if main_process:
            reward_types: dict[str, int] = {}
            for record in records:
                key = str(record.get("reward_type"))
                reward_types[key] = reward_types.get(key, 0) + 1
            record_entry = {
                "iteration": iteration,
                "optimizer_steps": optimizer_steps,
                "mean_reward": group_stats.to_dict()["mean_reward"],
                "loss": round(sum(step_losses) / len(step_losses), 6) if step_losses else None,
                "grad_norm": float(grad_norm) if step_losses else None,
                "lr": scheduler.get_last_lr()[0],
                "advantage_stats": group_stats.to_dict(),
                "render_stats": render_stats.to_dict(),
                "reward_types": dict(sorted(reward_types.items())),
                "seconds": {
                    "sample": round(sample_seconds, 1),
                    "train": round(train_seconds, 1),
                    "sync": round(sync_seconds, 1),
                    "total": round(time.time() - iteration_started, 1),
                },
                "rollout_usage": rollout_summary.get("usage"),
                "elapsed": round(time.time() - run_started, 1),
            }
            history.append(record_entry)
            log(json.dumps(record_entry, ensure_ascii=False))
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record_entry, ensure_ascii=False) + "\n")
            state_path.write_text(
                json.dumps({"iteration": iteration + 1}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if main_process:
        if server is not None:
            server.stop()
        metadata = {
            "schema_version": "ecom-grpo-training-v1",
            "hyperparams": {
                "init_model": str(args.init_model), "pool": str(args.pool),
                "iterations": args.iterations,
                "tasks_per_iteration": args.tasks_per_iteration,
                "group_size": args.group_size, "lr": args.lr,
                "warmup_steps": args.warmup_steps, "max_grad_norm": args.max_grad_norm,
                "invalid_reward": args.invalid_reward, "max_length": args.max_length,
                "tokens_per_batch": args.tokens_per_batch, "grad_accum": args.grad_accum,
                "temperature": args.temperature, "top_p": args.top_p,
                "world_size": accelerator.num_processes, "seed": args.seed,
                "kl_beta": 0.0, "inner_epochs": 1,
            },
            "optimizer_steps": optimizer_steps,
            "elapsed_seconds": round(time.time() - run_started, 1),
            "final": history[-1] if history else None,
        }
        (out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log(f"训练完成，权重与血缘 → {out_dir}")


if __name__ == "__main__":
    main()
