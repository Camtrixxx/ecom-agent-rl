#!/usr/bin/env bash
# E1：把 grpo_val（200 题）当 dev set，离线评 SFT + 8 个 GRPO 快照。
#
# 目的是回答「iter039 是不是最好的那个」。这个问题不需要重训：8 个快照都完整躺在盘上，
# 每个 15G、HF 格式齐全，vLLM 可直接 serve。改训练循环加 in-loop val 要重跑 6.56h 才能
# 验证改动有没有用，先做便宜的那个。
#
# 口径与 eval_grpo.sh 逐字一致（temperature 0.7 / top_p 0.95 / max_tokens 1024 /
# max_steps 35 / 窗口 24576 / k=4），**只换池子**。换池子是有意的：500 题池是 test set，
# 在它上面选 checkpoint 再在它上面报差值，就是在 test set 上做模型选择。
#
# 分辨率先算清楚：n=200、k=4 配对半宽约 ±3.2 pp（同血缘检查点相关性更高，实测 n=498
# 时 ±1.76 pp，按 √(498/200) 缩放约 ±2.8 pp）。而 iter019 → iter039 的真实差值是
# +2.63 pp。所以这套配置**能**回答「是不是提前见顶」，**不能**给相邻快照排名。
# 若前三名挤在 3 pp 内就当并列，对并列的补 k=8——这条规则写在跑之前。
#
# rollout 必须串行（EnvironmentPool 的 BoundedSemaphore 是每个客户端池各一份，同端口段
# 并行跑两个池会各自以为独占租约 → env_error → 整批中止）。vLLM 则在两张卡上交替预热，
# 把每次约 3 分钟的加载藏进上一个模型的 rollout 里。
#
#     bash scripts/eval_checkpoints.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

POOL=data/task_pools/grpo_val.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
ENV_BASE_PORT="${ENV_BASE_PORT:-5700}"
ENV_WORKERS="${ENV_WORKERS:-8}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
OUTDIR=outputs/rollouts/ckpt_eval
LOG=outputs/logs/eval_checkpoints.log

# 两张卡交替：A 在跑 rollout 时 B 预热下一个模型。
GPU_A="${GPU_A:-1}"; PORT_A=8190
GPU_B="${GPU_B:-2}"; PORT_B=8191

export no_proxy='*' NO_PROXY='*'

mkdir -p "$OUTDIR" outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MODELS=(
  "sft:${ROOT}/outputs/models/sft"
  "iter004:${ROOT}/outputs/models/grpo/iter004"
  "iter009:${ROOT}/outputs/models/grpo/iter009"
  "iter014:${ROOT}/outputs/models/grpo/iter014"
  "iter019:${ROOT}/outputs/models/grpo/iter019"
  "iter024:${ROOT}/outputs/models/grpo/iter024"
  "iter029:${ROOT}/outputs/models/grpo/iter029"
  "iter034:${ROOT}/outputs/models/grpo/iter034"
  "iter039:${ROOT}/outputs/models/grpo/iter039"
)

# 环境代码内容锚：硬闸门。环境代码变了意味着这批数字和 baseline/SFT/GRPO 那几批不在
# 同一口径上，继续跑只会产出一份看起来正常、实际不可比的报告。
if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  log "!! 环境代码与锚不一致，评测中止"
  exit 1
fi

serve() {  # serve <模型目录> <端口> <GPU> <日志>
  CUDA_VISIBLE_DEVICES="$3" LLM_PORT="$2" nohup bash scripts/serve_model.sh "$1" \
    > "$4" 2>&1 &
  echo $!
}

wait_ready() {  # wait_ready <端口> <超时秒>
  local port="$1" deadline=$((SECONDS + $2))
  while ((SECONDS < deadline)); do
    curl -sf --noproxy '*' "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}

stop_server() {  # stop_server <pid>
  [[ -z "${1:-}" ]] && return 0
  kill "$1" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null
  sleep 3
}

done_count() { wc -l < "$1" 2>/dev/null || echo 0; }

rollout() {  # rollout <端口> <输出> <名字>
  local port="$1" out="$2" name="$3"
  # 一次中止不丢数据：基础设施失败进 .failures.jsonl 不占 attempt，重跑按
  # (task_id, attempt) 续跑。所以原地重试最多 3 次，只会往前推进。
  for attempt in 1 2 3; do
    local n; n=$(done_count "$out")
    ((n >= TARGET)) && break
    log "  ${name}: 第 ${attempt} 次 rollout，当前 ${n}/${TARGET}"
    LLM_BASE_URL="http://127.0.0.1:${port}/v1" LLM_MODEL=ecom-agent \
      .venv/bin/python scripts/run_rollout.py \
        --pool "$POOL" --out "$out" \
        --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
        --env-base-port "$ENV_BASE_PORT" --env-workers "$ENV_WORKERS" \
        >> "$LOG" 2>&1
    local after; after=$(done_count "$out")
    log "  ${name}: ${n} → ${after}/${TARGET}"
    # 零进展说明不是 slot 抖动，别空转。
    ((after <= n)) && { log "  ${name}: 零进展，放弃重试"; break; }
    sleep 20
  done
}

log "###### E1 开始：${#MODELS[@]} 个模型 × ${TARGET} 回合（grpo_val k=${ATTEMPTS}）"

cur_pid=""; cur_port=""; cur_gpu=""
next_pid=""; next_port=""; next_gpu=""

cleanup() { log "收尾：停 vLLM"; stop_server "$cur_pid"; stop_server "$next_pid"; }
trap cleanup EXIT

for i in "${!MODELS[@]}"; do
  entry="${MODELS[$i]}"; name="${entry%%:*}"; path="${entry#*:}"
  out="${OUTDIR}/${name}.jsonl"

  if [[ ! -d "$path" ]]; then log "!! ${name}: 目录不存在 ${path}，跳过"; continue; fi
  if (( $(done_count "$out") >= TARGET )); then
    log "=== ${name}: 已有 $(done_count "$out")/${TARGET} 条，跳过"
    continue
  fi

  # 第一个模型没有预热好的服务，现起；之后每轮开头 next_* 已经在加载了。
  if [[ -z "$cur_pid" ]]; then
    cur_gpu="$GPU_A"; cur_port="$PORT_A"
    cur_pid=$(serve "$path" "$cur_port" "$cur_gpu" "outputs/logs/vllm_ckpt_${name}.log")
    log "=== ${name}: vLLM 启动 pid=${cur_pid} GPU${cur_gpu}:${cur_port}"
  fi

  if ! wait_ready "$cur_port" 900; then
    log "!! ${name}: vLLM 900s 未就绪，跳过"
    stop_server "$cur_pid"; cur_pid=""
    continue
  fi
  log "=== ${name}: 就绪，开始 rollout"

  # 在本模型 rollout 期间，把下一个模型加载到另一张卡上。
  if (( i + 1 < ${#MODELS[@]} )); then
    nentry="${MODELS[$((i + 1))]}"; nname="${nentry%%:*}"; npath="${nentry#*:}"
    nout="${OUTDIR}/${nname}.jsonl"
    if [[ -d "$npath" ]] && (( $(done_count "$nout") < TARGET )); then
      if [[ "$cur_gpu" == "$GPU_A" ]]; then next_gpu="$GPU_B"; next_port="$PORT_B";
      else next_gpu="$GPU_A"; next_port="$PORT_A"; fi
      next_pid=$(serve "$npath" "$next_port" "$next_gpu" "outputs/logs/vllm_ckpt_${nname}.log")
      log "    （预热 ${nname} pid=${next_pid} GPU${next_gpu}:${next_port}）"
    fi
  fi

  rollout "$cur_port" "$out" "$name"

  stop_server "$cur_pid"
  cur_pid="$next_pid"; cur_port="$next_port"; cur_gpu="$next_gpu"
  next_pid=""; next_port=""; next_gpu=""
done

log "###### rollout 全部结束，出报告"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"; out="${OUTDIR}/${name}.jsonl"
  [[ -s "$out" ]] || continue
  # 每个都对 SFT 配对：SFT 是共同基线，配对消掉题目难度方差。
  extra=(); [[ "$name" != "sft" ]] && extra=(--baseline "${OUTDIR}/sft.jsonl")
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "$out" "${extra[@]}" --pool "$POOL" \
    --json "${OUTDIR}/${name}.report.json" >> "$LOG" 2>&1
  log "报告 ${name} exit=$? 行数=$(done_count "$out")"
done

log "###### E1 完成"
