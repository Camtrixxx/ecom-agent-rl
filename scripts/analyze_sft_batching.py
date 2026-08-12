#!/usr/bin/env python3
"""量一遍 SFT 的成批效率与 val 分片截断，为 docs/optimization-plan.md 提供实测数。

只用 tokenizer，不碰 GPU。每条样本只做一次整体渲染（拿总长度），不逐回合定位
assistant span——本脚本只关心长度与分批，不关心 mask。

    python scripts/analyze_sft_batching.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

MODEL = "/data/heyuhang/models/Qwen2.5-7B-Instruct"


def lengths_of(path: Path, tokenizer, max_length: int) -> list[int]:
    from ecom_agent_rl.training.sft_dataset import normalize_tool_arguments

    out = []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        messages = normalize_tool_arguments(row.get("messages") or [])
        if messages is None:
            continue
        text = tokenizer.apply_chat_template(
            messages, tools=list(row.get("tools") or []) or None,
            tokenize=False, add_generation_prompt=False,
        )
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n <= max_length:
            out.append(n)
    return out


def main() -> None:
    from transformers import AutoTokenizer
    from train_sft import shard, token_budget_batches

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    report: dict[str, object] = {}

    for name, path in [("train", ROOT / "data/sft/train.jsonl"),
                       ("validation", ROOT / "data/sft_val/train.jsonl")]:
        lens = lengths_of(path, tokenizer, 32768)
        examples = [{"input_ids": [0] * n} for n in lens]
        batches = token_budget_batches(examples, 32768)

        # 每批 padding 后的实际占用 = 批内最长 × 批内条数
        padded = sum(max(lens[i] for i in b) * len(b) for b in batches)
        content = sum(lens)

        entry: dict[str, object] = {
            "samples": len(lens),
            "content_tokens": content,
            "batches": len(batches),
            "padded_tokens": padded,
            "padding_waste_ratio": round(1 - content / padded, 4),
        }
        for world in (6, 7, 8):
            kept = shard(batches, 0, world)
            used_batches = len(kept) * world
            dropped = [b for b in batches[used_batches:]]
            dropped_samples = sum(len(b) for b in dropped)
            # 被丢的是排序后最靠后的批，也就是最长的样本
            dropped_lens = [lens[i] for b in dropped for i in b]
            entry[f"world{world}"] = {
                "batches_per_rank": len(kept),
                "batches_used": used_batches,
                "batches_dropped": len(dropped),
                "samples_dropped": dropped_samples,
                "dropped_share": round(dropped_samples / max(1, len(lens)), 4),
                "dropped_len_min": min(dropped_lens) if dropped_lens else None,
                "dropped_len_max": max(dropped_lens) if dropped_lens else None,
                "kept_len_max": max((lens[i] for b in kept for i in b), default=None),
            }
        report[name] = entry

    print(json.dumps(report, ensure_ascii=False, indent=2))
    (ROOT / "outputs/logs/sft_batching_analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
