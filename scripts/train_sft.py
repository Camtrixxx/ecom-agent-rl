#!/usr/bin/env python3
"""7B 全参 SFT，8 卡 FSDP2。

用法（必须用 accelerate 起，单进程跑 7B 全参放不下）：
    bash scripts/train_sft.sh

    # 只验链路：几条样本跑 1 步
    bash scripts/train_sft.sh --limit 8 --max-train-steps 1

产物：
    <out-dir>/                  HF 格式权重，vLLM 可直接 serve
    <out-dir>/train_log.jsonl   每步的 loss / lr / 吞吐
    <out-dir>/metadata.json     血缘（数据 sha256、超参、渲染审计）

## 几个不显然的决定

**按 token 数分桶成批，而不是固定 batch size。** 样本长度从几千到三万多不等
（实测 p90 25776、max 35852）。固定 batch size 会让短样本批白等、长样本批 OOM；
按 token 预算分桶则每步的显存占用大致恒定。代价是每步样本数不定，所以 loss 要按
token 数加权平均，否则短样本批会被高估权重。

**loss 按全局有效 token 数归一，不是按批内。** 梯度累积时每个 micro-batch 的可训练
token 数差很多（4%-15%），按批内平均会让 token 少的 micro-batch 获得不成比例的权重。
先累加各 micro-batch 的 loss 之和，再除以这一步的总 token 数。

**不 resize embedding。** 我们没加特殊 token，vocab 保持 152064。动过 embedding 的
权重 vLLM 也能加载，但 tokenizer 与 config 不一致的风险不值得为省事承担。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="/data/heyuhang/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "sft" / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=None,
                        help="验证集 train.jsonl（build_sft_dataset.py 对 sft_val 的产物）")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "models" / "sft")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="7B 全参的常用量级；LoRA 才用 1e-4")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    # 覆盖实测 p90 25776；再往上是 max 35852 那条尾巴，丢掉比为它把窗口开到 40k 划算。
    parser.add_argument("--max-length", type=int, default=32768)
    # 每张卡每个 micro-batch 的 token 预算。32768 = 一条最长样本单独成批。
    parser.add_argument("--tokens-per-batch", type=int, default=32768)
    parser.add_argument("--grad-accum", type=int, default=4)
    # 算 loss 时一次上采成 fp32 的 token 数。vocab 152064 下每 4096 token 的 fp32
    # logits 约 2.3G，是显存与速度的折中；见 forward_loss。
    parser.add_argument("--loss-chunk-tokens", type=int, default=4096)
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 条（smoke 用）")
    parser.add_argument("--max-train-steps", type=int, default=None, help="提前停（smoke 用）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-each-epoch", action="store_true",
                        help="每个 epoch 存一份（默认只存最终权重）")
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"],
                        help="flash_attention_2 需要装 flash-attn；本机没装，默认 sdpa")
    return parser.parse_args()


@dataclass
class Hyperparams:
    """落进 metadata.json 的超参快照。"""

    model: str
    epochs: float
    lr: float
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float
    max_length: int
    tokens_per_batch: int
    grad_accum: int
    world_size: int
    seed: int
    attn: str


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_budget_batches(
    examples: Sequence[dict[str, Any]], tokens_per_batch: int
) -> list[list[int]]:
    """按 token 预算分桶，返回每批的样本下标。

    先按长度排序再分桶：同批长度接近，padding 浪费就小。批与批之间的顺序随后会被
    打散（见 `shuffled`），所以排序不会让训练看到「先短后长」的系统性顺序。

    padding 后的实际占用是 `批内最长 × 批内条数`，所以判据用它而不是长度之和——
    否则一条长样本配几条短的就会超预算。
    """
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    batches: list[list[int]] = []
    current: list[int] = []
    longest = 0
    for index in order:
        length = len(examples[index]["input_ids"])
        candidate_longest = max(longest, length)
        if current and candidate_longest * (len(current) + 1) > tokens_per_batch:
            batches.append(current)
            current, longest = [index], length
            continue
        current.append(index)
        longest = candidate_longest
    if current:
        batches.append(current)
    return batches


def shuffled(items: list[Any], seed: int, epoch: int) -> list[Any]:
    """每个 epoch 用不同但可复现的顺序。"""
    import random

    out = list(items)
    random.Random(f"{seed}:{epoch}").shuffle(out)
    return out


def shard(batches: list[Any], rank: int, world_size: int) -> list[Any]:
    """把批平均分给各 rank，并截到等长。

    截断是必须的：FSDP 的前反向是集合通信，某个 rank 少跑一批就会让其余 rank 永远
    等在 all-gather 上（表现为训练卡死而不是报错）。丢掉的量最多 world_size-1 批。
    """
    per_rank = len(batches) // world_size
    return [batches[i * world_size + rank] for i in range(per_rank)]


def val_rounds(batches: list[Any], rank: int, world_size: int) -> list[tuple[Any, float]]:
    """验证集的分卡方案：返回 [(批, 权重)]，长度对所有 rank 相同。

    和 shard() 的区别是**不丢样本**。shard() 截到等长是为了让各 rank 的 forward 次数
    相同（否则 FSDP 会卡在 all-gather 上），代价是丢掉尾部 len%world 批；训练侧无所谓
    （每 epoch 重新 shuffle），val 侧却致命——批是按长度排序的且不 shuffle，被丢的
    永远是最长的那几条，而且丢哪些取决于卡数，于是 val loss 不再跨卡数可比。

    这里改成 batches[rank::world]，再用**权重 0 的重复批**把各 rank 补到同样的轮数：
    forward 次数依然恒等，但补出来的那几次不进分子分母。于是每条样本恰好计一次，
    结果与 world_size 无关。
    """
    if world_size <= 0:
        raise ValueError(f"world_size 必须为正，得到 {world_size}")
    if len(batches) < world_size:
        raise ValueError(
            f"批数 {len(batches)} 少于进程数 {world_size}，无法给每个 rank 分到占位批"
        )
    mine = batches[rank::world_size]
    rounds = math.ceil(len(batches) / world_size)
    return [(mine[i], 1.0) if i < len(mine) else (mine[0], 0.0) for i in range(rounds)]


def steps_in_epoch(per_rank: int, grad_accum: int) -> int:
    """一个 epoch 内实际会走的 optimizer 步数。

    必须与训练循环 `range(0, per_rank - grad_accum + 1, grad_accum)` 同口径——那个
    循环只走**完整**的 grad_accum 组，所以是 floor 而不是 ceil。用 ceil 会让
    total_steps 大于所有 epoch 能走的步数之和，while 于是多跑一个残 epoch 去补齐，
    余弦调度在残 epoch 中途才退到 0。
    """
    return per_rank // grad_accum


def main() -> None:
    args = parse_args()

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    from ecom_agent_rl.training.sft_dataset import IGNORE_INDEX, collate, load_examples

    accelerator = Accelerator(gradient_accumulation_steps=args.grad_accum)
    set_seed(args.seed)
    main_process = accelerator.is_main_process

    def log(message: str) -> None:
        if main_process:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    if not args.train.exists():
        raise SystemExit(
            f"训练集不存在: {args.train}\n先跑 python scripts/build_sft_dataset.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        # Qwen2.5 有 pad_token（<|endoftext|>），这里只是不想在别的模型上静默出错。
        raise SystemExit("tokenizer 没有 pad_token_id，无法成批")

    log(f"渲染训练集 {args.train}")
    train_examples, train_stats = load_examples(
        args.train, tokenizer, max_length=args.max_length, progress=main_process
    )
    if args.limit:
        train_examples = train_examples[: args.limit]
    if not train_examples:
        raise SystemExit("训练集渲染后为空，检查 max_length 与数据格式")
    log(f"训练样本 {len(train_examples)}  审计 {json.dumps(train_stats.to_dict(), ensure_ascii=False)}")

    val_examples: list[dict[str, Any]] = []
    val_stats = None
    if args.validation and args.validation.exists():
        val_examples, val_stats = load_examples(
            args.validation, tokenizer, max_length=args.max_length, progress=main_process
        )
        if args.limit:
            val_examples = val_examples[: args.limit]
        log(f"验证样本 {len(val_examples)}")

    log(f"加载模型 {args.model}（attn={args.attn}）")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn, use_cache=False
    )
    # 长序列下激活值是显存主项，7B×32k 不开必 OOM。
    model.gradient_checkpointing_enable()

    batches = token_budget_batches(train_examples, args.tokens_per_batch)
    per_rank = len(shard(batches, 0, accelerator.num_processes))
    if per_rank == 0:
        raise SystemExit(
            f"批数 {len(batches)} 少于进程数 {accelerator.num_processes}；"
            "减小 --tokens-per-batch 或加样本"
        )
    # floor 而不是 ceil：下面的训练循环是
    #     range(0, len(rank_batches) - grad_accum + 1, grad_accum)
    # 也就是**只走完整的 grad_accum 组**，实际步数恒为 per_rank // grad_accum。用 ceil
    # 会让 total_steps 比所有 epoch 加起来能走的步数还大，于是 while 多跑一个残 epoch
    # 去补齐，而余弦调度在这个残 epoch 中途才走到 0（--save-each-epoch 还会多写一个
    # epoch 目录）。上次没暴露纯属整除的巧合：6 卡 per_rank=108、7 卡 92，都被 4 整除；
    # 8 卡 per_rank=81 就会踩——ceil 21 vs floor 20，total_steps 63 而 3 个 epoch 只有 60 步。
    steps_per_epoch = steps_in_epoch(per_rank, args.grad_accum)
    if steps_per_epoch == 0:
        # 训练循环一步都走不出来，而 while 只看 step < total_steps —— 会永远转下去。
        raise SystemExit(
            f"每卡 {per_rank} 批不足一个 grad_accum={args.grad_accum} 组，一步也走不了；"
            "减小 --grad-accum 或 --tokens-per-batch"
        )
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_train_steps:
        total_steps = min(total_steps, args.max_train_steps)
    log(f"{len(batches)} 批 / {accelerator.num_processes} 卡 → 每卡 {per_rank} 批，"
        f"{steps_per_epoch} 步/epoch，共 {total_steps} 步")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    # scheduler 建在 prepare 之后，且故意不交给 prepare 包装，两件事都是必需的：
    #
    # 不交给 prepare：它的包装器假定「dataloader 的 batch 被乘了 num_processes」，于是每
    # 调一次 step() 就把内部计数往前推 num_processes 步（accelerate/scheduler.py:75-76）。
    # 我们是自己分批分卡的，一步就该是一步。被包过会让 warmup 在第一步就走完、余弦半周期
    # 在 81/6≈13.5 步跑完然后开始振荡，训练结束时 lr 回到约 0.96e-5——等于完全没退火，
    # 而日志里的 loss 照样在降（实测 step 1 lr 已是 9.94e-6，step 13 掉到 3.6e-8 又爬回）。
    #
    # 建在 prepare 之后：prepare 返回的可能不是传进去的那个 optimizer 对象，scheduler 必须
    # 绑在真正被 step 的那个上，否则 lr 改了也不生效。
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "train_log.jsonl"
    if main_process:
        log_path.write_text("", encoding="utf-8")

    def forward_loss(indices: Sequence[int], source: Sequence[dict[str, Any]]) -> tuple[Any, int]:
        """返回 (loss 之和, 有效 token 数)。

        用 sum 而不是 mean：调用方要按全局 token 数归一（见模块 docstring），
        拿到 mean 就没法还原权重了。

        loss 分块算，因为 vocab 是 152064：一个 32k token 的批，logits 本身 bf16
        就要 9.1G，整体 `.float()` 再要 18.2G——这一个上采就够把 80G 卡打爆
        （实测 backward 里 "Tried to allocate 18.31 GiB" 而 GPU 只剩 14.4G）。
        按序列切块后同时只有一块被上采成 fp32，峰值降到约 1/N，数值上完全等价：
        reduction="sum" 对分块求和满足结合律。
        """
        batch = collate([source[i] for i in indices], pad_token_id=pad_token_id)
        batch = {k: v.to(accelerator.device) for k, v in batch.items()}
        outputs = model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
        )
        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        flat_logits = logits.reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)

        total = flat_labels.numel()
        chunk = max(1, int(args.loss_chunk_tokens))
        loss = None
        for start in range(0, total, chunk):
            piece = torch.nn.functional.cross_entropy(
                flat_logits[start : start + chunk].float(),
                flat_labels[start : start + chunk],
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            loss = piece if loss is None else loss + piece
        if loss is None:
            loss = flat_logits.sum() * 0.0
        return loss, int((labels != IGNORE_INDEX).sum().item())

    @torch.no_grad()
    def evaluate() -> float | None:
        """val loss。每条验证样本恰好计一次，结果与 world_size 无关。

        这里**不能**用 shard()。它为了对齐集合通信把各 rank 截到等长，丢掉尾部
        len(batches) % world_size 批。训练侧丢一点无所谓——每个 epoch 重新 shuffle，
        长期看每条都会被采到——但 val 侧不是：token_budget_batches 先按长度排序，
        val 又从不 shuffle，所以被丢的永远是排序后最靠后的那几批，也就是**最长的
        样本**，而且每个 epoch 丢的都是同一批。

        后果有二。一是偏：已公布的 0.39249 实际是 295 条上算的，缺的 3 条恰是最长
        的回合，而长回合是这个任务的主体。二是不可比：丢哪些取决于卡数（6 卡丢 3、
        7 卡丢 6、8 卡丢 7），两次不同卡数的运行是在两个不同的验证集上算 val loss，
        而没有任何东西会提示这件事。在「第 3 个 epoch 只降 0.0032」这种量级的比较上，
        验证集悄悄换掉 1-2% 的最长样本是不能接受的。

        改法：给每个 rank 发 batches[rank::world]，轮数取全局一致的
        ceil(len(batches)/world)。轮数不够的 rank 拿自己的第一批再跑一次但**权重 0**
        ——只为凑齐 all-reduce 的对端，不进分子也不进分母。各 rank 的 forward 次数
        因此恒等，FSDP 不会卡在 all-gather 上。
        """
        if not val_examples:
            return None
        all_batches = token_budget_batches(val_examples, args.tokens_per_batch)
        if not all_batches:
            return None
        # 每个 rank 至少要有一批才能拿来当零权重占位。只有「批数 < 卡数」时才不成立，
        # 那时 val 本身也没有意义了——明确报错好过静默算出一个错的数。
        try:
            rounds = val_rounds(
                all_batches, accelerator.process_index, accelerator.num_processes
            )
        except ValueError as exc:
            raise SystemExit(f"验证集分卡失败：{exc}；减小 --tokens-per-batch 或增加验证样本")
        model.eval()
        total = torch.zeros(2, device=accelerator.device)
        for indices, weight in rounds:
            loss, tokens = forward_loss(indices, val_examples)
            total += torch.tensor(
                [loss.item() * weight, tokens * weight], device=accelerator.device
            )
        total = accelerator.reduce(total, reduction="sum")
        model.train()
        return (total[0] / total[1]).item() if total[1] > 0 else None

    step = 0
    started = time.time()
    model.train()
    stop = False
    epoch = 0
    history: list[dict[str, Any]] = []

    def emit(record: dict[str, Any]) -> None:
        """进 history，同时落盘。

        val 记录以前只 append 不写盘（写盘发生在 log_every 那个分支里，val 走不到），
        于是 `grep val_loss train_log.jsonl` 是空的，三个 val 数只存在于人读日志里。
        要做 2 vs 3 epoch 的对比、要画 loss 曲线，第一步却是 grep 人读日志，这不对。
        """
        history.append(record)
        log(json.dumps(record, ensure_ascii=False))
        if main_process:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 训练前先量一次，给 val 曲线一个起点（epoch 0）。成本约 1-2 min，换来的是
    # 「第 1 个 epoch 降了多少」这个以前根本没有的数——原来的曲线从 epoch 1 才开始。
    baseline_val = evaluate()
    if baseline_val is not None:
        emit({"step": 0, "epoch": 0, "val_loss": round(baseline_val, 5)})

    while not stop and step < total_steps:
        rank_batches = shard(
            shuffled(batches, args.seed, epoch),
            accelerator.process_index,
            accelerator.num_processes,
        )
        for start in range(0, len(rank_batches) - args.grad_accum + 1, args.grad_accum):
            group = rank_batches[start : start + args.grad_accum]
            # 先数出这一步的全局 token 数，才能用它归一（各卡各 micro-batch 都不同）。
            local_tokens = sum(
                sum(1 for label in train_examples[i]["labels"] if label != -100)
                for indices in group
                for i in indices
            )
            step_tokens = accelerator.reduce(
                torch.tensor([local_tokens], device=accelerator.device, dtype=torch.float32),
                reduction="sum",
            ).item()
            if step_tokens == 0:
                continue

            loss_sum = 0.0
            for indices in group:
                loss, _ = forward_loss(indices, train_examples)
                # 除以全局 token 数：梯度是各卡求和的，所以这里的分母也用全局数，
                # 等价于「对全世界的有效 token 取平均」。
                scaled = loss / step_tokens
                accelerator.backward(scaled)
                loss_sum += scaled.item()

            grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0 or step == total_steps:
                reduced = accelerator.reduce(
                    torch.tensor([loss_sum], device=accelerator.device), reduction="sum"
                ).item()
                record = {
                    "step": step,
                    "epoch": round(epoch + (start + args.grad_accum) / max(1, len(rank_batches)), 3),
                    "loss": round(reduced, 5),
                    "lr": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm) if grad_norm is not None else None,
                    "tokens": int(step_tokens),
                    "elapsed": round(time.time() - started, 1),
                }
                emit(record)

            if step >= total_steps:
                stop = True
                break

        epoch += 1
        val_loss = evaluate()
        if val_loss is not None:
            emit({"step": step, "epoch": epoch, "val_loss": round(val_loss, 5)})
        if args.save_each_epoch and not stop:
            save(accelerator, model, tokenizer, args.out_dir / f"epoch{epoch}", log)

    save(accelerator, model, tokenizer, args.out_dir, log)

    if main_process:
        metadata = {
            "schema_version": "ecom-sft-training-v1",
            "hyperparams": asdict(Hyperparams(
                model=args.model, epochs=args.epochs, lr=args.lr,
                warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm, max_length=args.max_length,
                tokens_per_batch=args.tokens_per_batch, grad_accum=args.grad_accum,
                world_size=accelerator.num_processes, seed=args.seed, attn=args.attn,
            )),
            "provenance": {
                "train": str(args.train),
                "train_sha256": sha256_of(args.train),
                "validation": str(args.validation) if args.validation else None,
                "validation_sha256": (
                    sha256_of(args.validation)
                    if args.validation and args.validation.exists() else None
                ),
            },
            "render": {
                "train": train_stats.to_dict(),
                "validation": val_stats.to_dict() if val_stats else None,
            },
            "steps": step,
            "elapsed_seconds": round(time.time() - started, 1),
            # 拆成两项而不是一个 `final`：原来 final 取 history[-1]，而最后一条恰好
            # 总是 val 记录，于是最后一步的 train loss、lr、grad_norm 全被顶掉了。
            "final_train": next((r for r in reversed(history) if "loss" in r), None),
            "val_curve": [r for r in history if "val_loss" in r],
        }
        (args.out_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log(f"权重与血缘 → {args.out_dir}")


def save(accelerator: Any, model: Any, tokenizer: Any, out_dir: Path, log: Any) -> None:
    """存成 HF 格式，vLLM 能直接 serve。

    FSDP 把参数分片存在各卡上，必须先 all-gather 成完整 state dict 再写盘——
    `get_state_dict` 负责这件事。少了它存出来的是分片，vLLM 加载会失败。

    收尾那个 `wait_for_everyone` 不是保险，是必需的。gather 是 339 个逐参数的集合
    通信：非 0 号 rank 只要把自己的分片发出去就返回了，而 rank 0 还要把它们在 CPU 上
    拼成完整权重，慢得多。少了这道屏障，其余 rank 会直接走完 main() 退出、进程组被
    拆掉，rank 0 就永远等在没有对端的 gather 上——表现是训练完成后静默挂死，不报错
    （实测：6 rank 在 step 1 后瞬间只剩 1 个，py-spy 显示卡在 _gather_state_dict）。
    """
    accelerator.wait_for_everyone()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = accelerator.get_state_dict(model)
    # FSDP2 的混合精度在内部保留 fp32 主副本，gather 出来的就是 fp32：7B 存成 30G，
    # 且 config.json 会写 "dtype": "float32"，与我们训的 bf16 模型不是同一个东西。
    # 存盘前转回 bf16——精度上等价于前向用的权重，体积和加载时间都减半。
    if accelerator.is_main_process and state is not None:
        import torch as _torch

        state = {
            key: value.to(_torch.bfloat16)
            if isinstance(value, _torch.Tensor) and value.is_floating_point()
            else value
            for key, value in state.items()
        }
    unwrapped = accelerator.unwrap_model(model)
    unwrapped.save_pretrained(
        out_dir,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(out_dir)
        # save_pretrained 写 config.json 时按「模型当前参数」判定 dtype，而 FSDP 混合
        # 精度下活模型是 fp32，所以它会写 float32——和上面存成 bf16 的张量不一致。
        # vLLM 是照 config.json 决定加载精度的，这个字段错了会按 fp32 加载 bf16 权重。
        # 因此在写完之后按实际落盘的张量精度改正这一个字段。
        config_path = out_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["dtype"] = "bfloat16"
        config.pop("torch_dtype", None)
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        log(f"已存 {out_dir}")
    # 全员在此对齐后才离开 save，rank 0 的 gather 期间进程组保证还活着。
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
