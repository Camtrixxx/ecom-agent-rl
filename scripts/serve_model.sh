#!/usr/bin/env bash
# 用 vLLM 起一个 OpenAI 兼容服务，供 baseline 与评测调用。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-/data/heyuhang/models/Qwen2.5-7B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-ecom-agent}"
# 8000 是本机常见默认端口，已被别的服务占用；换到不易撞车的段。
LLM_PORT="${LLM_PORT:-8180}"
# 客户端按同一个数决定何时压缩历史（rollout/llm.py 的 DEFAULT_CONTEXT_WINDOW）。
# 改这里就要改那里，否则要么白压缩、要么照样撞 400。
# 模型本身支持 32768，但 24576 与参考实现一致，便于两边的数直接对比。
MAX_MODEL_LEN="${MAX_MODEL_LEN:-24576}"
TP_SIZE="${TP_SIZE:-1}"
GPU_UTIL="${GPU_UTIL:-0.85}"

# GPU 0 上有别人的常驻服务，默认落到它会与之争显存。GPU 1-7 空闲，从 1 起。
# 多卡时按 TP_SIZE 连续取；需要指定具体卡就直接传 CUDA_VISIBLE_DEVICES。
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  first_gpu="${FIRST_GPU:-1}"
  gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
  if [[ "${gpu_count}" -gt 0 && $((first_gpu + TP_SIZE)) -gt "${gpu_count}" ]]; then
    echo "TP_SIZE=${TP_SIZE} from GPU ${first_gpu} needs $((first_gpu + TP_SIZE)) GPUs," \
      "but only ${gpu_count} exist." >&2
    echo "占满全部 ${gpu_count} 张卡请显式设 CUDA_VISIBLE_DEVICES（注意 GPU 0 上有常驻服务）。" >&2
    exit 1
  fi
  devices="${first_gpu}"
  for ((i = 1; i < TP_SIZE; i++)); do
    devices+=",$((first_gpu + i))"
  done
  export CUDA_VISIBLE_DEVICES="${devices}"
fi

# 端口被占时 vLLM 要加载完权重才失败，先在这里拦掉。
# 探测放在子 shell 里，fd 随子 shell 退出自动关闭——在父 shell 里 `exec 3>&-`
# 关一个未打开的 fd 会让非交互 shell 直接退出，连报错都来不及打。
if (exec 3<>"/dev/tcp/127.0.0.1/${LLM_PORT}") 2>/dev/null; then
  echo "port ${LLM_PORT} is already in use; set LLM_PORT to a free port" >&2
  exit 1
fi
if [[ ! -d "${MODEL}" ]]; then
  echo "model directory not found: ${MODEL}" >&2
  exit 1
fi
if [[ ! -x "${ROOT}/.venv/bin/vllm" ]]; then
  echo "vLLM is not installed in ${ROOT}/.venv" >&2
  exit 1
fi

# 本机内核驱动只支持 CUDA 12.8，而 vLLM 0.25.1 依赖的 torch 是 cu130 轮子，
# 靠 cuda-compat 提供的 user-mode driver 补差。见 scripts/cuda_env.sh。
. "${ROOT}/scripts/cuda_env.sh"

# torch 的 CUDA 构建须不高于**生效的** driver 版本。注意比较对象是
# `cuDriverGetVersion`（加载 compat 后的实际能力），不是 `nvidia-smi` 报的内核模块
# 版本——后者恒为 12.8，用它比会把正确配置误判成错误。
#
# 这个检查值得留着：装错了 `torch.cuda.is_available()` 照样返回 True（它不真正建
# CUDA context），要等 vLLM 加载完权重、走到 `gpu_worker.init_device()` 才炸
# "driver is too old"，那时已经烧掉几分钟。
read -r torch_cuda driver_cuda <<<"$(
  LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" "${ROOT}/.venv/bin/python" - <<'PY' 2>/dev/null || true
import ctypes
try:
    import torch
    torch_cuda = torch.version.cuda or ""
except Exception:
    torch_cuda = ""
try:
    v = ctypes.c_int()
    ctypes.CDLL("libcuda.so.1").cuDriverGetVersion(ctypes.byref(v))
    driver_cuda = f"{v.value // 1000}.{v.value % 1000 // 10}"
except Exception:
    driver_cuda = ""
print(torch_cuda, driver_cuda)
PY
)"
if [[ -n "${torch_cuda}" && -n "${driver_cuda}" ]]; then
  # 版本号按 major.minor 数值比较，12.10 > 12.8 而字符串比较会反过来。
  if awk -v t="${torch_cuda}" -v d="${driver_cuda}" 'BEGIN {
        split(t, a, "."); split(d, b, ".")
        exit !(a[1] > b[1] || (a[1] == b[1] && a[2] > b[2]))
      }'; then
    echo "torch is built for CUDA ${torch_cuda} but the effective driver only supports ${driver_cuda}." >&2
    echo "CUDA compat 库可能没生效或版本不够，见 docs/environment-notes.md。" >&2
    exit 1
  fi
fi

echo "serving ${MODEL##*/} on port ${LLM_PORT}, GPU ${CUDA_VISIBLE_DEVICES}"

# hermes 是 Qwen2.5 的工具调用格式；Qwen3.5 用的 qwen3_coder 在此不适用。
exec "${ROOT}/.venv/bin/vllm" serve "${MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --port "${LLM_PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_UTIL}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
