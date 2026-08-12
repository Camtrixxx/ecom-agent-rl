#!/usr/bin/env bash
# E1 的补采：把并列的三个检查点从 k=4 补到 k=8。
#
# 触发的是 eval_checkpoints.sh 里跑之前就写死的那条：「若前三名挤在 3 pp 内就当并列，
# 对并列的补 k=8」。k=4 的结果是 iter034 +8.98 / iter039 +8.38 / iter019 +6.34，
# 跨度 2.64 pp < 3 pp，规则触发。
#
# **补哪三个、锐化哪个比较，要分开想清楚：**
#
# 只把这三个补到 k=8、sft 仍是 k=4，那么「vs sft」那一列几乎不会变窄——配对差值的方差是
# var(model) + var(baseline) − 2cov，基线那一项没动。所以补采**不是**为了让 vs-sft 的区间
# 变窄，那需要连 sft 一起补（多花 20 分钟，且 sft 不在并列名单里，规则没要求）。
#
# k=8 真正锐化的是**并列三者之间的两两配对**——而这恰好就是并列要解决的问题。三者同血缘、
# 同一批题，相关性高，配对差值的方差比各自对 sft 的差值小得多。
#
# 跑之前先写下预测，免得事后挑着解释：
#   - iter034 vs iter039 差 0.60 pp，k=8 **大概率仍判并列**（半宽从 ±3.2 缩到约 ±2.3，
#     要分辨 0.6 pp 得几十倍样本）。这条几乎注定是「测不出差异」——那本身就是结论：
#     没有证据说该把已发布的 iter039 换成 iter034。
#   - iter019 vs iter034 差 2.64 pp，**这条可能分得出**。而它正是 E1 设计时说这套配置
#     「能回答」的那个问题：是不是提前见顶。
#
# 端口段沿用 5700：E1 已退出，且夜间链被改成要等本脚本消失才开 E2，不会撞。
# 单卡串行，不预热——只有三个模型，省下的 9 分钟不值得多一份并发风险。
#
#     bash scripts/topup_k8.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

POOL=data/task_pools/grpo_val.jsonl
ATTEMPTS=8                      # 从 4 补到 8：resume 按 (task_id, attempt) 跳过，
                                # attempt 0-3 已在盘上，这一轮只跑 4-7，同一个 jsonl 追加
CONCURRENCY="${CONCURRENCY:-16}"
ENV_BASE_PORT="${ENV_BASE_PORT:-5700}"
ENV_WORKERS="${ENV_WORKERS:-8}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
OUTDIR=outputs/rollouts/ckpt_eval
LOG=outputs/logs/topup_k8.log
GPU="${GPU:-1}"; PORT=8190

MODELS=(
  "iter019:${ROOT}/outputs/models/grpo/iter019"
  "iter034:${ROOT}/outputs/models/grpo/iter034"
  "iter039:${ROOT}/outputs/models/grpo/iter039"
)

export no_proxy='*' NO_PROXY='*'
mkdir -p "$OUTDIR" outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  log "!! 环境代码与锚不一致，补采中止"
  exit 1
fi

serve() { CUDA_VISIBLE_DEVICES="$GPU" LLM_PORT="$PORT" nohup bash scripts/serve_model.sh "$1" > "$2" 2>&1 & echo $!; }

wait_ready() {
  local deadline=$((SECONDS + $1))
  while ((SECONDS < deadline)); do
    curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
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

log "###### k=8 补采开始：${#MODELS[@]} 个并列检查点 → 每个 ${TARGET} 回合"

pid=""
trap 'log "收尾：停 vLLM"; stop_server "$pid"' EXIT

for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"; path="${entry#*:}"
  out="${OUTDIR}/${name}.jsonl"
  if (( $(done_count "$out") >= TARGET )); then log "=== ${name}: 已到 ${TARGET}，跳过"; continue; fi

  pid=$(serve "$path" "outputs/logs/vllm_topup_${name}.log")
  log "=== ${name}: vLLM 启动 pid=${pid} GPU${GPU}:${PORT}"
  if ! wait_ready 900; then
    log "!! ${name}: vLLM 900s 未就绪，跳过"; stop_server "$pid"; pid=""; continue
  fi

  for attempt in 1 2 3; do
    n=$(done_count "$out")
    ((n >= TARGET)) && break
    log "  ${name}: 第 ${attempt} 次，当前 ${n}/${TARGET}"
    LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" LLM_MODEL=ecom-agent \
      .venv/bin/python scripts/run_rollout.py \
        --pool "$POOL" --out "$out" --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
        --env-base-port "$ENV_BASE_PORT" --env-workers "$ENV_WORKERS" >> "$LOG" 2>&1
    after=$(done_count "$out")
    log "  ${name}: ${n} → ${after}/${TARGET}"
    ((after <= n)) && { log "  ${name}: 零进展，放弃重试"; break; }
    sleep 20
  done

  stop_server "$pid"; pid=""
done

log "###### 出报告（k=4 的那份已另存为 *.k4.report.json，不覆盖）"
for entry in "${MODELS[@]}"; do
  name="${entry%%:*}"; out="${OUTDIR}/${name}.jsonl"
  [[ -s "$out" ]] || continue
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "$out" --baseline "${OUTDIR}/sft.jsonl" --pool "$POOL" \
    --json "${OUTDIR}/${name}.report.json" >> "$LOG" 2>&1
  log "报告 ${name} exit=$? 行数=$(done_count "$out")"
done

# 并列三者的两两配对——这才是 k=8 要锐化的东西。
log "###### 并列三者两两配对"
for pair in "iter034:iter039" "iter034:iter019" "iter039:iter019"; do
  a="${pair%%:*}"; b="${pair#*:}"
  [[ -s "${OUTDIR}/${a}.jsonl" && -s "${OUTDIR}/${b}.jsonl" ]] || continue
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "${OUTDIR}/${a}.jsonl" --baseline "${OUTDIR}/${b}.jsonl" --pool "$POOL" \
    --json "${OUTDIR}/pair_${a}_vs_${b}.json" >> "$LOG" 2>&1
  log "配对 ${a} vs ${b} exit=$?"
done

log "###### k=8 补采完成"
