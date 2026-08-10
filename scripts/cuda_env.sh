#!/usr/bin/env bash
# CUDA forward compatibility：让 CUDA 13 的 torch/vLLM 跑在 driver 570 上。
#
# 用法（source，不是执行）：
#   . "$(dirname "${BASH_SOURCE[0]}")/cuda_env.sh"
#
# 背景：本机内核驱动是 570.172.08，只支持到 CUDA 12.8，而 vLLM 0.25.1 依赖的
# torch==2.11.0 是 cu130 轮子。三条路里只有这一条走得通：
#
#   1. 换 cu128 的 torch —— 死路。vLLM 0.25.1 的预编译算子链接 libcudart.so.13，
#      硬塞 cu13 的库进 LD_LIBRARY_PATH 后能 import、不报错，但 `torch.ops._C.rms_norm`
#      **静默返回全零**。这种失败不会中止服务，只会让模型输出垃圾。
#   2. 升级内核驱动 —— 需要 root，且这是共用机器，GPU 0 上有别人的常驻服务。
#   3. cuda-compat（本文件）—— NVIDIA 官方支持的 forward compatibility：
#      数据中心卡（A800 属于）可以用新版 user-mode driver 配旧内核模块。
#      纯用户态，不动内核模块，对机器上其他人零影响。
#
# compat 库从 NVIDIA 的 el8 仓库取 `cuda-compat-13-0`，解包到 CUDA_COMPAT_DIR。
# 本机没有 rpm/cpio/bsdtar，解包过程见 docs/environment-notes.md。

CUDA_COMPAT_DIR="${CUDA_COMPAT_DIR:-/data/heyuhang/cudacompat/root/usr/local/cuda-13.0/compat}"

if [[ ! -f "${CUDA_COMPAT_DIR}/libcuda.so.1" ]]; then
  echo "找不到 CUDA compat 库: ${CUDA_COMPAT_DIR}/libcuda.so.1" >&2
  echo "见 docs/environment-notes.md 的「CUDA forward compatibility」一节。" >&2
  return 1 2>/dev/null || exit 1
fi

# compat 的 libcuda 必须排在系统库之前才会被优先加载。
export LD_LIBRARY_PATH="${CUDA_COMPAT_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
