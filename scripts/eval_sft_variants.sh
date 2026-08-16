#!/usr/bin/env bash
# E2：四份 SFT 权重在 grpo_val（200 题，k=4）上互比，回答「第 3 个 epoch 值不值」。
#
#   sft          已发布的那份（6 卡、3 epoch）——锚点，用来看重训本身的 run 间抖动
#   sft_e3       S4 重训的 3 epoch
#   sft_e2       S4 独立的 2 epoch（余弦在 2 个 epoch 内退到 0）
#   sft_e3/epoch2  sft_e3 跑到第 2 个 epoch 末的权重（余弦**没有**退到 0）
#
# 关键的一对是 sft_e3/epoch2 vs sft_e2：两者见过一样多的数据，差别只在退火。这个差
# 就是退火的贡献；而 sft_e3 vs sft_e3/epoch2 是第 3 个 epoch 的贡献。分开量，别混。
#
# 分辨率同 E1：n=200、k=4 半宽约 ±3.2 pp，而 val loss 上第 3 个 epoch 只降 0.0032。
# 大概率四者挤在噪声里——**「测不出差异」在这里就是结论**（说明 2 epoch 够用，能省
# 三分之一训练时间），不是失败，也不是加采样的理由。若前两名差值 > 3 pp 才补 k=8。
# 这条规则写在跑之前。
#
# 用池子而非 val loss 做判定：val loss 低不等于任务成功率高，而报告里报的是成功率。
#
#     bash scripts/eval_sft_variants.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

POOL=data/task_pools/grpo_val.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
ENV_BASE_PORT="${ENV_BASE_PORT:-5700}"
ENV_WORKERS="${ENV_WORKERS:-8}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
OUTDIR=outputs/rollouts/sft_variants
LOG=outputs/logs/eval_sft_variants.log

GPU_A="${GPU_A:-1}"; PORT_A=8192
GPU_B="${GPU_B:-2}"; PORT_B=8193

export no_proxy='*' NO_PROXY='*'
mkdir -p "$OUTDIR" outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

MODELS=(
  "sft:${ROOT}/outputs/models/sft"
  "sft_e3:${ROOT}/outputs/models/sft_e3"
  "sft_e2:${ROOT}/outputs/models/sft_e2"
  "sft_e3_ep2:${ROOT}/outputs/models/sft_e3/epoch2"
)

if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  log "!! 环境代码与锚不一致，评测中止"
  exit 1
fi

serve() { CUDA_VISIBLE_DEVICES="$3" LLM_PORT="$2" nohup bash scripts/serve_model.sh "$1" > "$4" 2>&1 & echo $!; }

wait_ready() {
  local port="$1" deadline=$((SECONDS + $2))
  while ((SECONDS < deadline)); do
    curl -sf --noproxy '*' "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 10
  done
  return 1
}

stop_server() {
  [[ -z "${1:-}" ]] && return 0
  kill "$1" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$1" 2>/dev/null || return 0; sleep 1; done
  kill -9 "$1" 2>/dev/null; sleep 3
}

done_count() { wc -l < "$1" 2>/dev/null || echo 0; }

# 判「模型能不能评」要看权重落没落盘，不能看目录在不在。train_sft.py 一开跑就先建
# out-dir 写 train_log.jsonl（train_sft.py:292），权重要到最后 save() 才出现；GRPO 同理。
# 这个脚本自己踩过：08-13 06:30 我趁 GPU 空闲提前起了它去评三份已就绪的权重，而
# sft_e2 还在训——目录已存在，守卫放行，vLLM 去加载一个没有权重的目录，卡满 900 s
# 超时才退，白烧 43 分钟 GPU，最后 sft_e2 还是得串行评。预热那一路尤其要防，它是静默的。
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }

rollout() {
  local port="$1" out="$2" name="$3"
  for attempt in 1 2 3; do
    local n; n=$(done_count "$out")
    ((n >= TARGET)) && break
    log "  ${name}: 第 ${attempt} 次 rollout，当前 ${n}/${TARGET}"
    LLM_BASE_URL="http://127.0.0.1:${port}/v1" LLM_MODEL=ecom-agent \
      .venv/bin/python scripts/run_rollout.py \
        --pool "$POOL" --out "$out" --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
        --env-base-port "$ENV_BASE_PORT" --env-workers "$ENV_WORKERS" >> "$LOG" 2>&1
    local after; after=$(done_count "$out")
    log "  ${name}: ${n} → ${after}/${TARGET}"
    ((after <= n)) && { log "  ${name}: 零进展，放弃重试"; break; }
    sleep 20
  done
}

log "###### E2 开始：${#MODELS[@]} 份权重 × ${TARGET} 回合"

cur_pid=""; cur_port=""; cur_gpu=""
next_pid=""; next_port=""; next_gpu=""
cleanup() { log "收尾：停 vLLM"; stop_server "$cur_pid"; stop_server "$next_pid"; }
trap cleanup EXIT

for i in "${!MODELS[@]}"; do
  entry="${MODELS[$i]}"; name="${entry%%:*}"; path="${entry#*:}"
  out="${OUTDIR}/${name}.jsonl"
  if ! has_weights "$path"; then log "!! ${name}: 权重不存在 ${path}，跳过"; continue; fi
  if (( $(done_count "$out") >= TARGET )); then log "=== ${name}: 已完成，跳过"; continue; fi

  if [[ -z "$cur_pid" ]]; then
    cur_gpu="$GPU_A"; cur_port="$PORT_A"
    cur_pid=$(serve "$path" "$cur_port" "$cur_gpu" "outputs/logs/vllm_var_${name}.log")
    log "=== ${name}: vLLM 启动 pid=${cur_pid} GPU${cur_gpu}:${cur_port}"
  fi
  if ! wait_ready "$cur_port" 900; then
    log "!! ${name}: vLLM 900s 未就绪，跳过"; stop_server "$cur_pid"; cur_pid=""; continue
  fi
  log "=== ${name}: 就绪，开始 rollout"

  if (( i + 1 < ${#MODELS[@]} )); then
    nentry="${MODELS[$((i + 1))]}"; nname="${nentry%%:*}"; npath="${nentry#*:}"
    if has_weights "$npath" && (( $(done_count "${OUTDIR}/${nname}.jsonl") < TARGET )); then
      if [[ "$cur_gpu" == "$GPU_A" ]]; then next_gpu="$GPU_B"; next_port="$PORT_B";
      else next_gpu="$GPU_A"; next_port="$PORT_A"; fi
      next_pid=$(serve "$npath" "$next_port" "$next_gpu" "outputs/logs/vllm_var_${nname}.log")
      log "    （预热 ${nname} pid=${next_pid} GPU${next_gpu}:${next_port}）"
    fi
  fi

  rollout "$cur_port" "$out" "$name"
  stop_server "$cur_pid"
  cur_pid="$next_pid"; cur_port="$next_port"; cur_gpu="$next_gpu"
  next_pid=""; next_port=""; next_gpu=""
done

log "###### 出报告（都对已发布的 sft 配对，sft 自己不配对）"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"; out="${OUTDIR}/${name}.jsonl"
  [[ -s "$out" ]] || continue
  extra=(); [[ "$name" != "sft" ]] && extra=(--baseline "${OUTDIR}/sft.jsonl")
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "$out" "${extra[@]}" --pool "$POOL" \
    --json "${OUTDIR}/${name}.report.json" >> "$LOG" 2>&1
  log "报告 ${name} exit=$? 行数=$(done_count "$out")"
done

# 关键的那一对单独出一份：退火的贡献。
if [[ -s "${OUTDIR}/sft_e2.jsonl" && -s "${OUTDIR}/sft_e3_ep2.jsonl" ]]; then
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "${OUTDIR}/sft_e2.jsonl" --baseline "${OUTDIR}/sft_e3_ep2.jsonl" \
    --pool "$POOL" --json "${OUTDIR}/anneal.report.json" >> "$LOG" 2>&1
  log "报告 退火(e2 vs e3_ep2) exit=$?"
fi

log "###### E2 完成"
