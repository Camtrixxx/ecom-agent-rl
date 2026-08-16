#!/usr/bin/env bash
# 等 #14 的 s44 评测开跑，然后在**同一个窗口**里重评一次 sft_v2，作为 s44 的漂移锚。
#
#     nohup bash scripts/arm_resample3.sh > outputs/logs/arm_resample3.nohup 2>&1 &
#
# ## 为什么要第三份锚
#
# 08-15 晚量到跨 session 漂移约 2 pp、符号随机（optimization-log 08-15 21:35）。步数对照臂
# 的两组评测日期是**整组错开**的（stepmatch 在 08-15/16 三个窗口，已发布的 sft_v2 三条都在
# 08-14），整组错开的漂移是偏置不是噪声，加多少个 seed 都平均不掉。s42 已经演示过一次：
# 对 08-14 的 sft_v2 是 −1.19 pp，对同窗口的 sft_v2_resample 是 +1.06 pp，**符号翻了**。
#
# s42 有 sft_v2_resample（20:37 那次）、s43 有 sft_v2_resample2（23:47 那次），只差 s44。
# 补上之后三个窗口各有一份同窗口锚，逐窗口作差再做 n=3，组间偏置就被消掉。判读规则写在
# docs/stepmatch-preregistration.md，**已经在看到 s43/s44 之前定稿**。
#
# ## 为什么是脚本 + 轮询，而不是定个点再手起
#
# s44 的训练结束时刻只能估（约 02:00），估早了撞 resample2 的端口段，估晚了错过窗口。
# 轮询「s44 的轨迹文件出现」是这件事的真实信号：文件第一行落盘 = rollout 真的开始了。
#
# ## 两道闸门，都是硬的
#
# 1. **5700 段不能有别的 rollout。** EnvironmentPool 的 BoundedSemaphore 每个客户端池各一份、
#    不跨进程，两个 rollout 抢同一段 → env_error → 整批中止（不是单条失败）。resample2 用的
#    就是 5700，所以必须等它退干净。s44 用 5800，和这里不冲突。
# 2. **GPU 1 不能有别的 vLLM。** GPU 0 是别人的常驻服务，不碰。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

NAME=sft_v2_resample3
WEIGHTS=outputs/models/sft_v2
BASELINE=outputs/rollouts/sft_v2_resample2.jsonl   # 配对基线取最近一次，同权重
TRIGGER=outputs/rollouts/sft_stepmatch_s44.jsonl
LOG=outputs/logs/arm_resample3.log

mkdir -p outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "=== 等待触发文件出现：${TRIGGER}"
waited=0
while [[ ! -s "$TRIGGER" ]]; do
  sleep 60
  waited=$((waited + 1))
  # 超过 5 h 就认为链条挂了或提前结束了，退出而不是无限等。
  if (( waited > 300 )); then
    log "!! 等了 5 h 还没看到 ${TRIGGER}，放弃。手动查 outputs/logs/d2_stepmatch.nohup"
    exit 1
  fi
  (( waited % 30 == 0 )) && log "    已等 ${waited} min"
done
log "=== s44 的 rollout 已开始（$(wc -l < "$TRIGGER") 行），准备起锚"

# 闸门 1：5700 段必须空。resample2 若还没退，就等它。
while pgrep -f "run_rollout.py.*--env-base-port 5700" >/dev/null; do
  log "    5700 段还有 rollout 在跑，等（两个进程抢同段会 env_error 整批中止）"
  sleep 60
done

# 闸门 2：8180 端口必须空（= GPU 1 上没有 vLLM 残留）。
#
# **不能写成「等所有 serve_model.sh 退」**——s44 自己的评测全程有一个 serve_model.sh 挂在
# GPU 2 上，那样等于等到 s44 评完才起，而这个脚本存在的全部意义就是和 s44 同窗口。
# 端口才是要抢的那个资源，查端口。CUDA_VISIBLE_DEVICES 是环境变量，pgrep 看不见，
# 所以也没法用「按 GPU 号 pgrep」来分辨两条腿。
while curl -sf --noproxy '*' "http://127.0.0.1:8180/v1/models" >/dev/null 2>&1; do
  log "    8180 上还有 vLLM 在听（GPU1 没空出来），等"
  sleep 30
done

if [[ ! -s "$BASELINE" ]]; then
  log "!! 配对基线 ${BASELINE} 不存在或为空，改用已发布的 sft_v2.jsonl（跨天，配对值要打折看）"
  BASELINE=outputs/rollouts/sft_v2.jsonl
fi

log "=== 启动 ${NAME}：权重 ${WEIGHTS}，基线 ${BASELINE}，GPU1:8180 env段5700"
GPU=1 PORT=8180 ENV_BASE_PORT=5700 \
  bash scripts/eval_model.sh "$NAME" "$WEIGHTS" "$BASELINE" \
  >> "outputs/logs/${NAME}.nohup" 2>&1
log "=== ${NAME} 结束 exit=$? 行数=$(wc -l 2>/dev/null < "outputs/rollouts/${NAME}.jsonl" || echo 0)"
