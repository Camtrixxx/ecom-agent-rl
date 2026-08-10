#!/usr/bin/env bash
# 用 vLLM 起一个 OpenAI 兼容服务，供 baseline 与评测调用。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/data/heyuhang/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ecom-agent}"
LLM_PORT="${LLM_PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.85}"

if [[ ! -d "${MODEL}" ]]; then
  echo "model directory not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${ROOT}/.venv/bin/vllm" ]]; then
  echo "vLLM is not installed in ${ROOT}/.venv" >&2
  exit 1
fi

# hermes 是 Qwen2.5 的工具调用格式；Qwen3.5 用的 qwen3_coder 在此不适用。
exec "${ROOT}/.venv/bin/vllm" serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${LLM_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
