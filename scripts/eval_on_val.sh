#!/usr/bin/env bash
# 在 grpo_val（200 题）上评任意一组权重，条目从命令行给。
#
#     nohup bash scripts/eval_on_val.sh v1_iter039:outputs/models/grpo/iter039 \
#       > outputs/logs/v1_val.nohup 2>&1 &
#
# 条目格式 `名字:权重目录[:温度]`，温度省略即 0.7。可以给多个，串行跑。
#
# ## 为什么另起一个而不是改 eval_grpo_v2_snapshots.sh
#
# 那个脚本的 MODELS 是写死的 v2 十条，而它的产出 outputs/rollouts/grpo_v2_snap/ 已经是
# 定稿数据。改它意味着「同一个文件名对应过两套模型列表」，以后回头看日志分不清哪次跑的
# 是哪一版。逐字抄它的守卫、换成参数化的入口，代价只是重复几十行。
#
# ## 口径与那个脚本逐字一致
#
# 同池（grpo_val 200 题）、同 k=4、同并发 16、同 35 步上限、同温度默认 0.7。所以这里出的
# 终局率可以和 grpo_v2_snap 那张曲线表直接并列。**成功率不行**——n=200 配对半宽约 ±3.2 pp，
# 分不出相邻快照，这个池子是用来选 checkpoint 的，不是用来报差值的。
#
# 不出 baseline 配对报告：调用方要什么基线各不相同，硬写一个反而容易配错。需要配对差就
# 自己再跑 report_metrics.py --baseline。
#
# **不能和任何别的 rollout 并发同一段环境端口**：EnvironmentPool 的 BoundedSemaphore 是
# 每个客户端池各一份、不跨进程，两个进程抢同一段 → env_error → 整批中止（不是单条失败）。
# 不同段（5700 / 5800 / …）并行是安全的，08-14 实测两条腿各 36 / 38 行/min。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

(( $# )) || { echo "用法：bash scripts/eval_on_val.sh 名字:权重目录[:温度] ..." >&2; exit 2; }

POOL=data/task_pools/grpo_val.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
ENV_BASE_PORT="${ENV_BASE_PORT:-5800}"
ENV_WORKERS="${ENV_WORKERS:-8}"
GPU="${GPU:-2}"
PORT="${PORT:-8193}"
OUTDIR="${OUTDIR:-outputs/rollouts/val_scan}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
LOG="${LOG:-outputs/logs/eval_on_val.log}"

export no_proxy='*' NO_PROXY='*'
mkdir -p "$OUTDIR" outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 硬闸门：环境代码变了就和已发布的曲线不在同一口径，跑了也不能比。
if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  python3 scripts/hash_environment.py >&2
  log "!! 环境代码与锚不一致，评测中止"
  exit 1
fi

# 判「能不能评」看权重落没落盘，不是看目录在不在——目录可能是训练中途建的空壳，
# vLLM 会卡满 900s 超时（08-13 06:30 白烧过 43 分钟）。
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }
# `2>/dev/null` 必须写在 `< "$1"` **之前**：重定向是从左到右生效的，文件不存在时报错的
# 是 shell 而不是 wc，写在后面的话那句 "No such file or directory" 在 stderr 还没被改道
# 时就已经打出来了。计数本身一直是对的（`|| n=0` 接住了退出码），只是日志里多一行看着
# 像出错的噪声。换个顺序就干净了。
done_count() { local n; n=$(wc -l 2>/dev/null < "$1") || n=0; echo "${n:-0}"; }

log "###### grpo_val 评测：$# 个条目 × ${TARGET} 回合，GPU${GPU}:${PORT} env段${ENV_BASE_PORT}"

for entry in "$@"; do
  IFS=: read -r name path temp <<< "$entry"
  temp="${temp:-0.7}"
  out="${OUTDIR}/${name}.jsonl"

  if ! has_weights "$path"; then log "!! ${name}: 权重不存在 ${path}，跳过"; continue; fi
  if (( $(done_count "$out") >= TARGET )); then
    log "=== ${name}: 已有 $(done_count "$out")/${TARGET}，跳过"; continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU" LLM_PORT="$PORT" nohup bash scripts/serve_model.sh "$path" \
    > "outputs/logs/vllm_val_${name}.log" 2>&1 &
  pid=$!
  log "=== ${name}: vLLM 启动 pid=${pid} temp=${temp} 权重=${path}"

  deadline=$((SECONDS + 900)); ready=0
  while ((SECONDS < deadline)); do
    curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
      && { ready=1; break; }
    sleep 10
  done
  if (( ! ready )); then
    log "!! ${name}: vLLM 900s 未就绪，跳过"
    kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
    continue
  fi

  for attempt in 1 2 3; do
    n=$(done_count "$out")
    (( n >= TARGET )) && break
    log "  ${name}: 第 ${attempt} 次 rollout，当前 ${n}/${TARGET}"
    LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" LLM_MODEL=ecom-agent \
      .venv/bin/python scripts/run_rollout.py \
        --pool "$POOL" --out "$out" \
        --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
        --temperature "$temp" \
        --env-base-port "$ENV_BASE_PORT" --env-workers "$ENV_WORKERS" >> "$LOG" 2>&1
    after=$(done_count "$out")
    log "  ${name}: ${n} → ${after}/${TARGET}"
    (( after <= n )) && { log "  ${name}: 零进展，放弃重试"; break; }
    sleep 20
  done

  kill "$pid" 2>/dev/null; sleep 15; kill -9 "$pid" 2>/dev/null
  sleep 5

  [[ -s "$out" ]] || continue
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "$out" --pool "$POOL" \
    --json "${OUTDIR}/${name}.report.json" >> "$LOG" 2>&1
  log "报告 ${name} exit=$? 行数=$(done_count "$out")/${TARGET}"
done

log "###### 结束。终局率速览："
.venv/bin/python - "$OUTDIR" "$@" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
print(f"{'名字':<16}{'轨迹':>6}{'终局率':>9}{'成功率':>9}{'条件买对':>10}{'标签率':>9}")
for entry in sys.argv[2:]:
    name = entry.split(":")[0]
    p, traj = d / f"{name}.report.json", d / f"{name}.jsonl"
    tag = tot = 0
    if traj.exists():
        with traj.open() as f:
            for line in f:
                tot += 1
                if "<tool_call>" in line:
                    tag += 1
    if not p.exists():
        print(f"{name:<16}{tot:>6}   （缺报告）")
        continue
    o = json.loads(p.read_text())["overall"]
    n = o["trajectories"]; st = o["statuses"]
    term = st.get("done", 0) / n if n else float("nan")
    # success_rate 是 {mean, ci_low, ci_high}，不是标量——直接当 float 用会 TypeError。
    succ = (o.get("success_rate") or {}).get("mean", float("nan"))
    print(f"{name:<16}{n:>6}{term:>9.4f}{succ:>9.4f}"
          f"{succ / term if term else float('nan'):>10.4f}{tag / tot if tot else float('nan'):>9.4f}")
PY
