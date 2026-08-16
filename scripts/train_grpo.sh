#!/usr/bin/env bash
# 7B 全参 GRPO，FSDP2。底座与 train_sft.sh 相同，差别只在卡的分配。
#
# 卡的分配：vLLM 独占 GPU_VLLM（默认 1），训练用 GPUS（默认 2-7，共 6 张）。两者必须
# 不相交——vLLM 按 gpu_memory_utilization 抢一大块显存并一直持有，和 FSDP 同卡必 OOM。
# GPU 0 上有别人的常驻服务，两边都不碰。
#
# **改 GPUS 的卡数不是无害的调度动作**：optimizer 的 global batch 是 grad_accum × 卡数，
# 6 卡是 12 批、4 卡是 8 批。已发布的 seed 42 用的就是这里的默认 6 卡，所以任何要和它
# 比较的 run（例如 R1 的 seed 43/44）都必须留着默认值——否则卡数就和被研究的变量混在
# 一起，而那恰好是那个实验唯一要量的东西。宁可让空闲卡闲着。
#
# 训练进程自己管 vLLM 的生死（每轮换权重要重启），所以**跑这个脚本前不要手动起
# vLLM**：端口被占的话 serve_model.sh 会直接退出，训练在第一轮就失败。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# driver 570 只支持到 CUDA 12.8 而 torch 是 cu130 轮子，靠 compat 库补。
# 训练入口漏了这一句会撞 "NVIDIA driver is too old"。
# shellcheck source=/dev/null
. "${ROOT}/scripts/cuda_env.sh"

# 显式把 venv 放到 PATH 前面，不依赖调用者有没有 activate。这不只是省事：训练进程
# 会 fork 出 serve_model.sh 去起 vLLM，子进程继承的是**这里**的 PATH——父进程靠
# 外部 activate 跑起来的话，子进程照样找不到 vllm，而失败要等到第一轮采样才暴露。
if [[ -x "${ROOT}/.venv/bin/accelerate" ]]; then
  export PATH="${ROOT}/.venv/bin:${PATH}"
fi

# 环境代码的内容锚：reward 变了而没人注意，这一轮训练就和之前的实验不可比，而日志里
# 看不出任何异常。6.5 小时的训练值得先花 0.2 秒校一下。
python3 "${ROOT}/scripts/hash_environment.py" --quiet

GPU_VLLM="${GPU_VLLM:-1}"
GPUS="${GPUS:-2,3,4,5,6,7}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${GPUS}")"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 环境池和 vLLM 都在本机，系统代理指向 127.0.0.1:7980，不绕开会把 loopback 也转出去。
export no_proxy='*'
export NO_PROXY='*'

echo "训练 GPU ${GPUS}（${NUM_GPUS} 卡），vLLM GPU ${GPU_VLLM}"

accelerate launch \
  --num_processes "${NUM_GPUS}" \
  --mixed_precision bf16 \
  --use_fsdp \
  --fsdp_version 2 \
  --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
  --fsdp_transformer_layer_cls_to_wrap Qwen2DecoderLayer \
  --fsdp_reshard_after_forward true \
  --fsdp_state_dict_type FULL_STATE_DICT \
  --fsdp_cpu_ram_efficient_loading true \
  --fsdp_activation_checkpointing false \
  scripts/train_grpo.py --vllm-gpus "${GPU_VLLM}" "$@"
