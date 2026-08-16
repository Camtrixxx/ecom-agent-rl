#!/usr/bin/env bash
# R1：单个 GRPO seed 的留出集评测，口径与 eval_grpo.sh 逐字一致，只换模型目录。
#
#     bash scripts/eval_seed.sh grpo_s43        # 评 outputs/models/grpo_s43/policy
#
# 判定规则见 docs/multiseed-preregistration.md（先写后跑）。这里只产出一个数：
# 该 seed 的 policy 对**同一份** SFT 轨迹（outputs/rollouts/sft.jsonl）的配对差值。
# 基线不重采是有意的——要量的是 GRPO 侧的 run 间散布，重采 SFT 会把 SFT 的采样噪声
# 混进来，而那一项在 eval-preregistration 里已经量过。
#
# 与 eval_grpo.sh 的差别只有两处：模型目录来自参数、只评一个模型（不评 iter019）。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

NAME="${1:?用法：bash scripts/eval_seed.sh <run 名，如 grpo_s43>}"
MODEL="${MODEL:-${ROOT}/outputs/models/${NAME}/policy}"
POOL=data/task_pools/evaluation.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
GPU="${GPU:-1}"
PORT="${PORT:-8180}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
OUT="outputs/rollouts/${NAME}.jsonl"
LOG=outputs/logs/eval_${NAME}.log

export no_proxy='*' NO_PROXY='*'
mkdir -p outputs/rollouts outputs/logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 判「模型能不能评」要看权重落没落盘，不能看目录在不在。train_sft.py 一开跑就先建
# out-dir 写 train_log.jsonl（train_sft.py:292），权重要到最后 save() 才出现；GRPO 同理。
# 拿目录当守卫，就会在训练中途放行，vLLM 去加载一个没有权重的目录，卡满 900 s 超时——
# 这不是假想：08-13 06:30 就这么白烧了 43 分钟 GPU。
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }

has_weights "$MODEL" || { log "!! 模型权重不存在（目录可能只是训练中途建的）：${MODEL}"; exit 1; }

# 硬闸门：环境代码变了就和 baseline/SFT/seed42 不在同一口径，跑了也不能比。
if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  python3 scripts/hash_environment.py >&2
  log "!! 环境代码与锚不一致，评测中止"
  exit 1
fi

.venv/bin/python scripts/check_environment.py >> "$LOG" 2>&1 \
  || log "!! 环境池有泄漏，先跑 check_environment.py --reclaim 再来"

CUDA_VISIBLE_DEVICES="$GPU" LLM_PORT="$PORT" nohup bash scripts/serve_model.sh "$MODEL" \
  > "outputs/logs/vllm_${NAME}.log" 2>&1 &
PID=$!
log "vLLM 启动 pid=${PID} GPU${GPU}:${PORT} ← ${MODEL}"

cleanup() {
  log "停 vLLM"
  kill "$PID" 2>/dev/null; sleep 15; kill -9 "$PID" 2>/dev/null
}
trap cleanup EXIT

deadline=$((SECONDS + 900))
until curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; do
  ((SECONDS < deadline)) || { log "!! vLLM 900s 未就绪"; exit 1; }
  sleep 10
done
log "vLLM 就绪，开始 rollout"

# 断点续跑安全：按 (task_id, attempt) 去重，基础设施失败进 .failures.jsonl 不占 attempt。
for attempt in 1 2 3; do
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  (( n >= TARGET )) && break
  log "第 ${attempt} 次 rollout，当前 ${n}/${TARGET}"
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" LLM_MODEL=ecom-agent \
    .venv/bin/python scripts/run_rollout.py \
      --pool "$POOL" --out "$OUT" \
      --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" >> "$LOG" 2>&1
  after=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  log "${n} → ${after}/${TARGET}"
  (( after <= n )) && { log "零进展，放弃重试"; break; }
  sleep 20
done

log "###### 出报告"
.venv/bin/python scripts/report_metrics.py \
  --trajectories "$OUT" --baseline outputs/rollouts/sft.jsonl \
  --pool "$POOL" --json "outputs/rollouts/${NAME}_vs_sft.report.json" >> "$LOG" 2>&1
log "报告 exit=$? 行数=$(wc -l < "$OUT" 2>/dev/null || echo 0)/${TARGET}"
