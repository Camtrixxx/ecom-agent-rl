#!/usr/bin/env bash
# 起 8 卡 FSDP2 全参 SFT。参数原样透给 scripts/train_sft.py。
#
# 用法：
#     bash scripts/train_sft.sh                                  # 全量
#     bash scripts/train_sft.sh --limit 8 --max-train-steps 1     # smoke
#
# GPU 0 上有别人的常驻服务，默认用 1-7；要占满 8 张卡显式给 CUDA_VISIBLE_DEVICES。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 本机内核驱动只支持 CUDA 12.8，torch 是 cu130 轮子，靠 cuda-compat 补差。
# 少了这一行不会立刻报错——算子会静默返回全零，训练 loss 看着还在动。
. "${ROOT}/scripts/cuda_env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3,4,5,6,7}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"$CUDA_VISIBLE_DEVICES")"

# tokenizers 在 DataLoader fork 之后会警告并自行禁用并行；显式关掉少一堆噪音。
export TOKENIZERS_PARALLELISM=false
# 长序列 + FSDP 的显存碎片化明显，expandable_segments 能少掉几次 OOM。
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "FSDP2 全参 SFT：${NUM_GPUS} 卡（GPU ${CUDA_VISIBLE_DEVICES}）"

# 为什么是 FSDP2 而不是 ZeRO-3：本机没装 deepspeed，而装它要拖一串编译依赖；
# FSDP2 是 torch 自带的，分片粒度与 ZeRO-3 等价（参数+梯度+优化器状态全分片）。
#
# transformer_based_wrap + Qwen2DecoderLayer：按 decoder layer 分片。整模型一个
# 分片单元的话 all-gather 会把整个 7B 拉回单卡，等于没分片。
#
# SHARDED_STATE_DICT 存出来是分片文件，vLLM 加载不了；用 FULL_STATE_DICT 让
# get_state_dict 在主进程 all-gather 成完整权重。7B bf16 ≈ 15G，主机内存放得下。
exec "${ROOT}/.venv/bin/accelerate" launch \
  --num_processes "$NUM_GPUS" \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_version 2 \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
  --fsdp_reshard_after_forward true \
  --fsdp_state_dict_type FULL_STATE_DICT \
  --fsdp_cpu_ram_efficient_loading true \
  --fsdp_activation_checkpointing false \
  "${ROOT}/scripts/train_sft.py" "$@"
