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
    steps_per_epoch = max(1, math.ceil(per_rank / args.grad_accum))
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    if args.max_train_steps:
        total_steps = min(total_steps, args.max_train_steps)
    log(f"{len(batches)} 批 / {accelerator.num_processes} 卡 → 每卡 {per_rank} 批，"
        f"{steps_per_epoch} 步/epoch，共 {total_steps} 步")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

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
        if not val_examples:
            return None
        model.eval()
        val_batches = shard(
            token_budget_batches(val_examples, args.tokens_per_batch),
            accelerator.process_index,
            accelerator.num_processes,
        )
        total = torch.zeros(2, device=accelerator.device)
        for indices in val_batches:
            loss, tokens = forward_loss(indices, val_examples)
            total += torch.tensor([loss.item(), tokens], device=accelerator.device)
        total = accelerator.reduce(total, reduction="sum")
        model.train()
        return (total[0] / total[1]).item() if total[1] > 0 else None

    step = 0
    started = time.time()
    model.train()
    stop = False
    epoch = 0
    history: list[dict[str, Any]] = []

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
                history.append(record)
                log(json.dumps(record, ensure_ascii=False))
                if main_process:
                    with log_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if step >= total_steps:
                stop = True
                break

        epoch += 1
        val_loss = evaluate()
        if val_loss is not None:
            log(f"epoch {epoch} 验证 loss {val_loss:.5f}")
            history.append({"step": step, "epoch": epoch, "val_loss": round(val_loss, 5)})
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
            "final": history[-1] if history else None,
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
