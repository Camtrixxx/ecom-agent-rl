#!/usr/bin/env bash
# 通用留出集评测：一份权重 → 500 题 × k=4 → 配对报告。口径与 eval_grpo.sh / eval_seed.sh
# 逐字一致（同池、同 k、同温度 0.7、同并发），差别只在模型路径与对照来自参数。
#
#     bash scripts/eval_model.sh sft_v2  outputs/models/sft_v2         outputs/rollouts/sft.jsonl
#     bash scripts/eval_model.sh grpo_v2 outputs/models/grpo_v2/policy outputs/rollouts/grpo.jsonl
#
# 为什么不复用 eval_seed.sh：它把权重路径写成 `outputs/models/<name>/policy`（GRPO 的布局，
# SFT 没有 policy 子目录），把对照写死成 outputs/rollouts/sft.jsonl。D2 要评的两份权重
# 一份是 SFT 布局、一份要对 GRPO-v1 配对，两处都不合。改那个脚本会动到 R1 的产出口径，
# 所以另起一个，逐字抄它的守卫。
#
# 第三个参数（对照）可省。省了就只出单侧报告，不出配对差值。
#
# **不要和 GRPO 训练并发跑。** GRPO 训练循环自己也用 EnvironmentPool，而 BoundedSemaphore
# 是每个 client pool 各一份、不跨进程——两边一起抢同一段环境端口就会 env_error，而
# env_error 是整批中止而不是单条失败。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

NAME="${1:?用法：bash scripts/eval_model.sh <名字> <权重目录> [对照轨迹]}"
MODEL="${2:?第二个参数是权重目录}"
BASELINE="${3:-}"

POOL=data/task_pools/evaluation.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
GPU="${GPU:-1}"
PORT="${PORT:-8180}"
# 两个评测要并行时，这两段都必须错开：vLLM 端口，和环境池的端口段。环境池的
# BoundedSemaphore 是每个 client pool 各一份、不跨进程，所以两个进程用同一段端口
# 会各自以为租约是自己的 → env_error → 整批中止（不是单条失败）。
ENV_BASE_PORT="${ENV_BASE_PORT:-5700}"
ENV_WORKERS="${ENV_WORKERS:-8}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
OUT="outputs/rollouts/${NAME}.jsonl"
LOG="outputs/logs/eval_${NAME}.log"

export no_proxy='*' NO_PROXY='*'
mkdir -p outputs/rollouts outputs/logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 判「模型能不能评」要看权重落没落盘，不能看目录在不在：train_sft.py 一开跑就先建 out-dir
# 写 train_log.jsonl（train_sft.py:292），权重要到最后 save() 才出现。拿目录当守卫就会在
# 训练中途放行，vLLM 去加载一个没有权重的目录，卡满 900 s 超时——08-13 06:30 这么白烧过
# 43 分钟 GPU。
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }

has_weights "$MODEL" || { log "!! 权重不存在（目录可能只是训练中途建的）：${MODEL}"; exit 1; }
[[ -n "$BASELINE" && ! -s "$BASELINE" ]] && { log "!! 对照轨迹不存在或为空：${BASELINE}"; exit 1; }

# 硬闸门：环境代码变了就和 baseline / SFT / GRPO-v1 不在同一口径，跑了也不能比。
if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  python3 scripts/hash_environment.py >&2
  log "!! 环境代码与锚不一致，评测中止"
  exit 1
fi

# 自愈：这一段环境端口没人在听，就自己起一个池。
#
# 为什么加：本脚本原来假定「有个长跑的环境服务在那儿」。这个假定对 5700 成立，对别的段
# 不成立——run_d2.sh:159 给阶段 G 的第二条腿传 ENV_BASE_PORT=5800，而 5800 段从来没有起
# 过服务（outputs/environment/ 里只有 worker-5700..5731）。不补的后果不是崩而是降级：那
# 条腿的 rollout 全程连不上环境，阶段 H 的 3 对 3 变 3 对 2，df 变 3 而 analyze_d2.py 的
# 临界值写死 2.78（t₃ = 3.18），打印的区间窄约 14%。
#
# 为什么只做加法、不回落到 5700：两个 rollout 抢同一段端口，EnvironmentPool 的
# BoundedSemaphore 又是每个 client pool 各一份、不跨进程，于是双方都以为租约是自己的
# → env_error → **整批中止**。回落会把阶段 G 两条腿一起弄坏，比只坏一条更糟。
#
# 端口已在听时整段是 no-op，所以对既有调用（一律 5700）没有任何行为改变。用 /dev/tcp 探
# TCP 而不是 curl 探 HTTP：健康路由的名字这里不需要知道，而 `curl -sf` 撞上 404 会把活着
# 的池误判成没起来，反而去起第二个池。
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&-; }
if port_open "$ENV_BASE_PORT"; then
  log "环境段 ${ENV_BASE_PORT} 已在监听，不起池"
else
  log "环境段 ${ENV_BASE_PORT} 没有服务，自己起 ${ENV_WORKERS} 个 worker（CPU-only，不碰 GPU）"
  SHOPSIM_WORKERS="$ENV_WORKERS" SHOPSIM_BASE_PORT="$ENV_BASE_PORT" \
    nohup bash scripts/start_environment.sh \
    > "outputs/logs/env_${ENV_BASE_PORT}.nohup" 2>&1 &
  log "  start_environment.sh pid=$!"
  # worker 逐个起，探最后一个端口：它开了说明整段都开了。start_environment.sh 自己会先
  # 校环境内容锚（--quiet），所以自起的池和长跑的池在同一口径上。
  ENV_LAST_PORT=$((ENV_BASE_PORT + ENV_WORKERS - 1))
  for _ in $(seq 1 60); do
    sleep 5
    port_open "$ENV_LAST_PORT" && break
  done
  if port_open "$ENV_BASE_PORT" && port_open "$ENV_LAST_PORT"; then
    log "  ${ENV_BASE_PORT}-${ENV_LAST_PORT} 就绪"
  else
    log "!! ${ENV_BASE_PORT} 段 300 s 内没起全，下面的 rollout 大概会连不上环境"
  fi
fi

# 必须把段传进去：check_environment.py 的 --base-port 默认是 5700，不传的话本条腿明明用
# 5900 段却去查 5700，查了个和自己无关的池子——多腿并行时（#28 用了 6 段）这个检查会
# 完全失去意义，而且永远不报错，看起来像"检查过了"。不加 --reclaim：只读，不能动别的
# 腿在飞的租约。
.venv/bin/python scripts/check_environment.py \
    --base-port "$ENV_BASE_PORT" --workers "$ENV_WORKERS" >> "$LOG" 2>&1 \
  || log "!! 环境段 ${ENV_BASE_PORT} 有泄漏，先跑 check_environment.py --base-port ${ENV_BASE_PORT} --reclaim 再来"

# 把本段环境池的**启动时刻**记进日志。08-16 06:40 自查发现各段的池年龄能差到 5 天
# （段 5700 起于 08-10、5900 起于当天），而 #28 按腿的顺序分段，结果两个被比较的组
# 系统性地拿到了不同年龄的池。那次偏置实测只有 −0.06 pp，但**它当时在任何产物里都查不到**
# ——只能靠 ps 现场抓，跑完就没了。一个查不到的变量没法在事后排除，所以这里把它落盘。
# 失败不阻断评测：这是审计信息，不是闸门。
pool_started() {
  local pid
  pid=$(ss -ltnp 2>/dev/null | awk -v p=":${ENV_BASE_PORT}\$" '$4 ~ p {print $NF}' \
        | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  [ -z "$pid" ] && { echo "查不到监听 ${ENV_BASE_PORT} 的进程"; return; }
  ps -o lstart= -p "$pid" 2>/dev/null | xargs || echo "pid ${pid} 已退"
}
log "环境段 ${ENV_BASE_PORT} 的池启动于：$(pool_started)"

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
log "vLLM 就绪，开始 rollout（目标 ${TARGET} 回合）"

# 断点续跑安全：按 (task_id, attempt) 去重，基础设施失败进 .failures.jsonl 不占 attempt。
for attempt in 1 2 3; do
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  (( n >= TARGET )) && break
  log "第 ${attempt} 次 rollout，当前 ${n}/${TARGET}"
  LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" LLM_MODEL=ecom-agent \
    .venv/bin/python scripts/run_rollout.py \
      --pool "$POOL" --out "$OUT" \
      --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
      --env-base-port "$ENV_BASE_PORT" --env-workers "$ENV_WORKERS" >> "$LOG" 2>&1
  after=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  log "${n} → ${after}/${TARGET}"
  (( after <= n )) && { log "零进展，放弃重试"; break; }
  sleep 20
done

log "###### 出报告"
extra=(); [[ -n "$BASELINE" ]] && extra=(--baseline "$BASELINE")
.venv/bin/python scripts/report_metrics.py \
  --trajectories "$OUT" "${extra[@]}" --pool "$POOL" \
  --json "outputs/rollouts/${NAME}.report.json" >> "$LOG" 2>&1
log "报告 exit=$? 行数=$(wc -l < "$OUT" 2>/dev/null || echo 0)/${TARGET}"
