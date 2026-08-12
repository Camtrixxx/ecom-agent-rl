#!/usr/bin/env bash
# 教师轨迹采集：把 .env 里的 TEACHER_* 映射成 run_rollout.py 认的 LLM_*，然后按顺序
# 跑 val 与 train 两个池子。
#
# 用法：
#     bash scripts/collect_teacher.sh                    # val 然后 train
#     bash scripts/collect_teacher.sh sft_val            # 只跑一个
#
# 为什么用脚本而不是直接敲命令：这是共用机器，api key 不能进命令行（`ps` 里所有
# 用户可见）也不该进 shell history。key 只从 .env 读，`set -a` 让它进环境变量。
#
# 为什么并发默认取 worker 数：环境是单进程 Flask，被 GIL 锁在 1 核，加 slot 不提升
# 吞吐（docs/environment-notes.md）。实测并发 24 会把 slot 耗尽、报 env_error，而
# env_error 属于 INFRA_FAILURES，会中止整批采集。别调高。
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f .env ]]; then
    echo "缺 .env（从 .env.example 复制填写）" >&2
    exit 1
fi

# .env 里的端口/并发是**默认值**，不是硬性配置：调用方可以先 export 覆盖，用于和另一
# 批 rollout 并行时错开端口段（同端口段并行会互抢 slot → env_error → 整批中止）。
# 必须在 source 之前存下来——`set -a; source .env` 会无条件覆盖同名变量。
_override_base_port="${SHOPSIM_BASE_PORT:-}"
_override_workers="${SHOPSIM_WORKERS:-}"

set -a
# shellcheck disable=SC1091
source .env
set +a

[[ -n "${_override_base_port}" ]] && SHOPSIM_BASE_PORT="${_override_base_port}"
[[ -n "${_override_workers}" ]] && SHOPSIM_WORKERS="${_override_workers}"

export LLM_BASE_URL="${TEACHER_BASE_URL:?.env 缺 TEACHER_BASE_URL}"
export LLM_MODEL="${TEACHER_MODEL:?.env 缺 TEACHER_MODEL}"
export LLM_API_KEY="${TEACHER_API_KEY:?.env 缺 TEACHER_API_KEY}"

WORKERS="${SHOPSIM_WORKERS:-8}"
BASE_PORT="${SHOPSIM_BASE_PORT:-5700}"
# 教师窗口远大于本地 vLLM 的 24576，长回合不必早早压缩历史。
CONTEXT_WINDOW="${TEACHER_CONTEXT_WINDOW:-65536}"

POOLS=("$@")
if [[ ${#POOLS[@]} -eq 0 ]]; then
    POOLS=(sft_val sft_train)
fi

mkdir -p outputs/teacher

for name in "${POOLS[@]}"; do
    echo "=== $name  并发 $WORKERS  窗口 $CONTEXT_WINDOW  $(date '+%F %T')"
    python scripts/run_rollout.py \
        --pool "data/task_pools/${name}.jsonl" \
        --out "outputs/teacher/${name}.jsonl" \
        --attempts 1 \
        --concurrency "$WORKERS" \
        --env-base-port "$BASE_PORT" \
        --env-workers "$WORKERS" \
        --context-window "$CONTEXT_WINDOW"
    echo "=== $name 完成 $(date '+%F %T')"
done
