#!/usr/bin/env bash
# GRPO 的留出集评测：policy(iter039) 与 iter019 各 500 题 × 4 次，口径与
# baseline / SFT 逐字一致。判定规则见 docs/eval-preregistration.md（先写后跑）。
#
# 两次 rollout **必须串行**：EnvironmentPool 的 BoundedSemaphore 是每个客户端池
# 各自一份的，同端口段并行跑两个池会各自以为独占 4 个租约，服务端那个 worker 实际
# 只有 4 个 → env_error → 属于 INFRA_FAILURES → 整批中止。两边都会死。
#
# 两个 vLLM 反而可以并行起（不同 GPU、不同端口），省掉中间那次 4 分钟加载等待。
#
# 断点续跑安全：重跑同一条命令按 (task_id, attempt) 去重；env_error 进
# .failures.jsonl 不占 attempt。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

POOL=data/task_pools/evaluation.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
TARGET=$((500 * ATTEMPTS))
LOG=outputs/logs/grpo_eval.log

# 代理会把回环也绕进去（http_proxy 指向 127.0.0.1:7980），环境池和 vLLM 都在本机。
export no_proxy='*' NO_PROXY='*'

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

serve() {  # serve <模型目录> <端口> <GPU> <日志>
  CUDA_VISIBLE_DEVICES="$3" LLM_PORT="$2" nohup bash scripts/serve_model.sh "$1" \
    > "$4" 2>&1 &
  echo $!
}

wait_ready() {  # wait_ready <端口> <超时秒>
  local port="$1" deadline=$((SECONDS + $2))
  while ((SECONDS < deadline)); do
    if curl -sf --noproxy '*' "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

rollout() {  # rollout <端口> <输出> <说明>
  local port="$1" out="$2" name="$3"
  log "=== ${name}: rollout 开始（端口 ${port} → ${out}）"
  LLM_BASE_URL="http://127.0.0.1:${port}/v1" LLM_MODEL=ecom-agent \
    .venv/bin/python scripts/run_rollout.py \
      --pool "$POOL" --out "$out" \
      --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
      >> "$LOG" 2>&1
  local rc=$?
  local n; n=$(wc -l < "$out" 2>/dev/null || echo 0)
  log "=== ${name}: rollout exit=${rc}，已有 ${n}/${TARGET} 条"
  # 中止不丢数据，直接原地重试一次；仍不满就往下走，报告照出、行数如实记。
  if ((n < TARGET)); then
    log "=== ${name}: 未跑满，重试一次（续跑，只补缺失的 attempt）"
    LLM_BASE_URL="http://127.0.0.1:${port}/v1" LLM_MODEL=ecom-agent \
      .venv/bin/python scripts/run_rollout.py \
        --pool "$POOL" --out "$out" \
        --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
        >> "$LOG" 2>&1
    n=$(wc -l < "$out" 2>/dev/null || echo 0)
    log "=== ${name}: 重试后 ${n}/${TARGET} 条"
  fi
}

log "###### GRPO 评测开始（k=${ATTEMPTS}，目标每模型 ${TARGET} 条）"

# 先量一次容量：slot 泄漏会表现成「并发没超也报 env_error」，事后才发现要重跑。
.venv/bin/python scripts/check_environment.py >> "$LOG" 2>&1 \
  || log "!! 环境池有泄漏，先跑 check_environment.py --reclaim 再来"

PID_POLICY=$(serve "${ROOT}/outputs/models/grpo/policy" 8180 1 outputs/logs/vllm_grpo_policy.log)
PID_I019=$(serve "${ROOT}/outputs/models/grpo/iter019" 8181 2 outputs/logs/vllm_grpo_iter019.log)
log "vLLM 启动中：policy pid=${PID_POLICY} (GPU1:8180)，iter019 pid=${PID_I019} (GPU2:8181)"

cleanup() {
  log "停 vLLM（父进程会带走 EngineCore 子进程）"
  kill "$PID_POLICY" "$PID_I019" 2>/dev/null
  sleep 15
  kill -9 "$PID_POLICY" "$PID_I019" 2>/dev/null
}
trap cleanup EXIT

if ! wait_ready 8180 900; then log "!! policy 的 vLLM 900s 内没就绪"; exit 1; fi
log "policy 就绪"

rollout 8180 outputs/rollouts/grpo.jsonl "policy(iter039)"

if ! wait_ready 8181 900; then log "!! iter019 的 vLLM 900s 内没就绪"; exit 1; fi
log "iter019 就绪"

rollout 8181 outputs/rollouts/grpo_iter019.jsonl "iter019"

# --- 报告：只读 jsonl，不需要模型服务 -------------------------------------
log "###### 出报告"
report() {  # report <轨迹> <对照或 -> <输出>
  local extra=()
  [[ "$2" != "-" ]] && extra=(--baseline "$2")
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "$1" "${extra[@]}" \
    --pool "$POOL" --json "$3" >> "$LOG" 2>&1
  log "报告 $3 exit=$?"
}

# 主比较（预注册）：GRPO vs SFT
report outputs/rollouts/grpo.jsonl outputs/rollouts/sft.jsonl \
       outputs/rollouts/grpo_vs_sft.report.json
# 次要：GRPO vs baseline，以及 GRPO 自身的分层报告
report outputs/rollouts/grpo.jsonl outputs/rollouts/baseline.jsonl \
       outputs/rollouts/grpo_vs_baseline.report.json
# 探索性：后半程有没有白跑
report outputs/rollouts/grpo_iter019.jsonl outputs/rollouts/sft.jsonl \
       outputs/rollouts/iter019_vs_sft.report.json
report outputs/rollouts/grpo.jsonl outputs/rollouts/grpo_iter019.jsonl \
       outputs/rollouts/policy_vs_iter019.report.json

# 乘性分解：预注册预测 1 说增益必须落在「条件买对率」而非「走到终局率」。
.venv/bin/python scripts/decompose_success.py \
  outputs/rollouts/sft.jsonl outputs/rollouts/grpo.jsonl >> "$LOG" 2>&1
log "分解 exit=$?"

log "###### 完成：grpo=$(wc -l < outputs/rollouts/grpo.jsonl) iter019=$(wc -l < outputs/rollouts/grpo_iter019.jsonl)"
