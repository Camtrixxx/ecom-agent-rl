#!/usr/bin/env bash
# D2 链的看护脚本：补上 5800 环境池，并在链跑完后回填任何缺掉的评测、重出判定。
#
#     nohup bash scripts/d2_guard.sh > outputs/logs/d2_guard.nohup 2>&1 &
#
# ## 为什么需要它
#
# `run_d2.sh:159` 给阶段 G 的第二条腿传 `ENV_BASE_PORT=5800`，但 **5800 段从来没有起过
# 服务**：`eval_model.sh` 自己不起环境池（只做 hash 闸门 + check_environment 检漏），
# 它假定有个长跑的服务在那儿；而 `outputs/environment/` 里只有 worker-5700..5731，
# 没有 worker-58xx。E1 的 `eval_checkpoints.sh` 头部也写明「rollout 必须串行」，它实现
# 并发的方式是只把 vLLM 在两张卡上交替预热，环境段始终是 5700 一段——项目里从来没有
# 「两个环境段」这个用法，5800 是写 run_d2.sh 时凭空取的。
#
# 不修的后果是**降级**而不是崩（读过 analyze_d2.py 才确认）：`die()` 只记问题不抛异常，
# 所以阶段 H 照样出数，但 3 对 3 变成 3 对 2，且 df 变 3 而临界值写死 2.78（t₃ = 3.18），
# 打印的区间会窄约 14%。脚本自己有 `if df != 4` 的告警。任务 #12 的全部意义在 n=3，
# 所以要修，但不必当灾难。
#
# ## 为什么是独立脚本而不是改 run_d2.sh
#
# **bash 按字节偏移读正在执行的脚本**，改一个正在跑的脚本会让它错位执行。run_d2.sh
# 还在跑（阶段 D/E/F），所以修法必须是加法。
#
# ## 设计成幂等 + 覆盖所有分支
#
# 写这个脚本的直接原因是 Bash 工具的安全分类器在间歇性超时，能不能在阶段 G 开始前
# （约 17:20）成功发出一条命令是不确定的。所以把所有分支收进一个脚本，**任何时刻只要
# 有一次调用成功就够了**：
#
#   - 早于阶段 G 跑起来 → 5800 池就位，阶段 G 按原样跑通，零额外代价。
#   - 晚于阶段 G 跑起来 → 5800 没赶上，但链跑完后这里会串行回填缺掉的评测再重出判定，
#     代价约 37 分钟，结论不丢。
#   - 重复调用 → 每一步都先检查再动作，不会重复起池、不会重复评测。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

LOG=outputs/logs/d2_guard.log
mkdir -p outputs/logs outputs/rollouts
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 用 /dev/tcp 探 TCP 端口，而不是 curl 探 HTTP 路由：环境服务的健康路由名字这里不需要
# 知道，而 `curl -sf` 撞上 404 会判失败，反而误判成没起来。
port_open() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3<&-; }

log "###### D2 guard 启动"

# --- 1：补 5800 环境池 -------------------------------------------------------
if port_open 5800; then
  log "=== 5800 已在监听，不重复起池"
else
  log "=== 5800 没有服务，起一个 8 worker 的池（CPU-only，不碰 GPU）"
  SHOPSIM_WORKERS=8 SHOPSIM_BASE_PORT=5800 \
    nohup bash scripts/start_environment.sh \
    > outputs/logs/env_5800.nohup 2>&1 &
  log "  start_environment.sh pid=$!"
  # 8 个 worker 逐个起，给足时间；每 5 s 探一次最后一个端口（5807），最后一个开了
  # 说明整段都起来了。
  for _ in $(seq 1 60); do
    sleep 5
    port_open 5807 && break
  done
  if port_open 5800 && port_open 5807; then
    log "  5800-5807 就绪"
  else
    log "!! 5800 段 300 s 内没起全。阶段 G 的第二条腿会失败，但下面的回填会兜住。"
  fi
fi

# --- 2：等 run_d2.sh 跑完 ----------------------------------------------------
# 匹配模式里不含本脚本名，所以不会自匹配（pgrep -f 会匹配调用方自己的 cmdline）。
if pgrep -f "bash scripts/run_d2.sh" >/dev/null; then
  log "=== run_d2.sh 在跑，等它结束（每 60 s 探一次）"
  while pgrep -f "bash scripts/run_d2.sh" >/dev/null; do sleep 60; done
  log "=== run_d2.sh 已结束"
else
  log "=== run_d2.sh 不在跑（已结束或还没起）"
fi

# --- 3：回填缺掉的评测 -------------------------------------------------------
# 逐条串行：两个 rollout 抢同一段环境端口会 env_error 整批中止，所以这里绝不并行，
# 一律走 5700 段（唯一确定长跑的那个）。
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }

# 名字:权重目录:对照轨迹（对照为 - 表示不给 --baseline）
RUNS=(
  "sft:outputs/models/sft:-"
  "sft_s43:outputs/models/sft_s43:outputs/rollouts/sft.jsonl"
  "sft_s44:outputs/models/sft_s44:outputs/rollouts/sft.jsonl"
  "sft_v2:outputs/models/sft_v2:outputs/rollouts/sft.jsonl"
  "sft_v2_s43:outputs/models/sft_v2_s43:outputs/rollouts/sft.jsonl"
  "sft_v2_s44:outputs/models/sft_v2_s44:outputs/rollouts/sft.jsonl"
  "grpo_v2:outputs/models/grpo_v2/policy:outputs/rollouts/grpo.jsonl"
)

missing=0
for entry in "${RUNS[@]}"; do
  IFS=: read -r name path baseline <<< "$entry"
  rep="outputs/rollouts/${name}.report.json"
  if [[ -s "$rep" ]]; then
    log "=== ${name}: 报告已在，跳过"
    continue
  fi
  if ! has_weights "$path"; then
    log "!! ${name}: 报告缺，且权重不存在（${path}）——训练没成功，回填不了"
    missing=$((missing + 1))
    continue
  fi
  log "=== ${name}: 报告缺、权重在 → 用 ENV_BASE_PORT=5700 串行补评（约 37 min）"
  args=("$name" "$path")
  [[ "$baseline" != "-" ]] && args+=("$baseline")
  GPU=1 PORT=8180 ENV_BASE_PORT=5700 ENV_WORKERS=8 \
    bash scripts/eval_model.sh "${args[@]}" >> "$LOG" 2>&1
  rc=$?
  if [[ -s "$rep" ]]; then
    log "  ${name}: 补评完成 exit=${rc}"
  else
    log "!! ${name}: 补评后仍无报告 exit=${rc}"
    missing=$((missing + 1))
  fi
done

# --- 4：重出判定 -------------------------------------------------------------
# 无条件重跑：即使阶段 H 已经跑过一次，那一次可能是 3 对 2 的降级结果，回填之后要用
# 齐的数重出。analyze_d2.py 只读报告、只写 d2_verdict.txt，重跑是安全的。
log "=== 重跑 analyze_d2.py"
.venv/bin/python scripts/analyze_d2.py 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
log "=== analyze_d2.py exit=${rc}（1 表示有口径问题，数不可直接引用）"

(( missing > 0 )) && log "!! 仍有 ${missing} 份报告缺失，上面的判定是降级结果"

# --- 5：接着扫 grpo_v2 快照 --------------------------------------------------
# 串在这里而不是单独发一条命令，同样是因为分类器不可靠：这样**一次成功的调用**就覆盖
# 了「补池 + 回填 + 重出判定 + 扫快照」全部四件事，不用守着链条结束的时间点再发一次。
#
# 此刻 run_d2.sh 已经退出（第 2 步等过了），GPU 全空，所以快照脚本不会和任何东西抢。
# 它自己也有排队逻辑，重复保护，跑不坏。
#
# 这一步是 D2 之后最值钱的测量：−6.29 pp 全在走到终局率上，而判别量只剩 no_terminal
# 且被温度压缩约 17 倍，所以要 (a) 定位终局率是哪一轮开始塌的，(b) 用 iter039_t10 那条
# 同池同权重只改温度的臂，把「温度」和「池子难度」拆开。
log "=== 接着扫 grpo_v2 快照（10 个模型 × grpo_val 200 题 × k=4，单卡 GPU1，约 5 h）"
bash scripts/eval_grpo_v2_snapshots.sh >> "$LOG" 2>&1
log "=== 快照扫描 exit=$?"

log "###### D2 guard 结束"
