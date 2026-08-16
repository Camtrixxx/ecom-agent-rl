#!/usr/bin/env bash
# 在宽容重解析口径下把 D2 主判定重评一遍：v1 与 v2 都加兜底，两边同口径。
#
#     nohup bash scripts/run_tolerant_reeval.sh > outputs/logs/tolerant_reeval.nohup 2>&1 &
#
# ## 为什么必须两边都重评
#
# 08-14 14:06 量到 grpo_v2 的 256 条标签轨迹里 206 条（80.5%）宽容重解析能取到合法首块。
# 那意味着 D2 主判定的 −6.29 pp 里有一块是**推理侧的记账损耗**，不是策略差。
#
# 但只给 v2 加兜底再和 v1 的旧数比，是把兜底的收益单方面记到 v2 头上。v1 的标签率虽然
# 只有 0.25%（5 条/2000），不为零就不能假设它是零。**同口径**才是可比的最低要求，所以
# 这条链把三份权重全部重评：
#
#   grpo_v1_tol           v1 的产物（= iter039，规则判「保留」）
#   grpo_v2_tol           v2 已发布的产物（= iter039，规则判「回退」，留着做对照）
#   grpo_v2_iter034_tol   v2 按规则选出的产物
#
# ## 口径警告
#
# `ROLLOUT_TOLERANT_PARSE=1` 会改成功率：一部分 no_tool_call 变成正常步骤或 truncated。
# **这条链产出的数不能和 08-15 之前发布的任何数字并列**，只能三份互相比。轨迹记录里带
# `tolerant_parse: true`，日志第一屏也会打警告，两处都能自证口径。
#
# ## 卡与端口
#
# GPU1 / vLLM 8180 / 环境段 5700，与步数对照臂（已改到 GPU2 / 8181 / 5800）错开。两个
# rollout 抢同一段环境端口会 env_error → 整批中止，不是单条失败。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=outputs/logs/tolerant_reeval.log
mkdir -p outputs/logs outputs/rollouts
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

export ROLLOUT_TOLERANT_PARSE=1
export GPU="${GPU:-1}" PORT="${PORT:-8180}"
export ENV_BASE_PORT="${ENV_BASE_PORT:-5700}" ENV_WORKERS="${ENV_WORKERS:-8}"

rollout_done() { local n; n=$(wc -l < "$1" 2>/dev/null) || n=0; (( ${n:-0} >= 2000 )); }

# 先等 GPU1 上的严格口径评测收工。两件事都要等：抢同一段环境端口会整批中止，而
# 同一张卡起第二个 vLLM 会显存不足。
if pgrep -f "eval_model.sh grpo_v2_iter034" >/dev/null; then
  log "=== 严格口径的 iter034 评测还在跑，排队等它（同卡同端口段）"
  while pgrep -f "eval_model.sh grpo_v2_iter034" >/dev/null; do sleep 60; done
  log "=== 它结束了，等 90s 回收显存"
  sleep 90
fi

# 顺序有依赖：v1 先跑，因为后两份要拿它当配对基线。
# 条目格式 名字:权重目录:基线轨迹（`-` 表示不出配对报告）
ENTRIES=(
  "grpo_v1_tol:outputs/models/grpo/iter039:-"
  "grpo_v2_tol:outputs/models/grpo_v2/iter039:outputs/rollouts/grpo_v1_tol.jsonl"
  "grpo_v2_iter034_tol:outputs/models/grpo_v2/iter034:outputs/rollouts/grpo_v1_tol.jsonl"
)

log "###### 宽容口径重评：${#ENTRIES[@]} 份权重 × 2000 回合（500 题 × k=4，温度 0.7）"
log "      ROLLOUT_TOLERANT_PARSE=1 —— 这批数与已发布的严格口径数字不可直接比较"

for entry in "${ENTRIES[@]}"; do
  IFS=: read -r name path baseline <<< "$entry"
  out="outputs/rollouts/${name}.jsonl"

  if rollout_done "$out"; then
    log "=== ${name}: 已跑满，跳过"; continue
  fi
  if [[ "$baseline" != "-" && ! -s "$baseline" ]]; then
    log "!! ${name}: 配对基线 ${baseline} 不存在或为空——前一份没跑成，停在这里"
    log "   （不降级成无基线报告：那会产出一个看起来正常、但和主判定不可比的数）"
    exit 1
  fi

  args=("$name" "$path")
  [[ "$baseline" != "-" ]] && args+=("$baseline")
  log "=== ${name}: 开始（基线 ${baseline}）"
  bash scripts/eval_model.sh "${args[@]}" >> "outputs/logs/eval_${name}.nohup" 2>&1
  rc=$?
  log "=== ${name}: 结束 exit=${rc} 行数=$(wc -l < "$out" 2>/dev/null || echo 0)/2000"

  # 口径自证：跑满了却一条 tolerant_parse 标记都没有，说明环境变量没传进去，
  # 那这份数其实是严格口径的，名字却带 _tol —— 必须当成失败。
  if [[ -s "$out" ]] && ! head -1 "$out" | grep -q '"tolerant_parse": *true'; then
    log "!! ${name}: 轨迹里没有 tolerant_parse 标记，环境变量没生效，这份数名不副实"
    exit 1
  fi
done

log "###### 重评结束，出对照表"
.venv/bin/python scripts/score_tolerant_reeval.py 2>&1 | tee -a "$LOG"
