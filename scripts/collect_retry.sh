#!/usr/bin/env bash
# D1：重采 1,219 个零 accepted 任务，中止就重启，直到跑满或试满上限。
#
# 为什么要包这层循环：上一次教师采集被一次 `HTTP 400 Content Exists Risk` 中止过，
# 它归到 llm_error → INFRA_FAILURES → 整批中止。重启是无损的：基础设施失败进
# .failures.jsonl 不占 attempt，已完成的回合留在主文件里，重跑按 (task_id, attempt)
# 续跑。所以循环只会往前推进，不会重复采样。
#
# 端口段用 5800 而不是默认的 5700：5700 段留给同期跑的 E1 评测。同端口段并行跑两个
# 客户端池会各自以为独占租约 → env_error → 两边都死。
#
#     bash scripts/start_environment.sh   # 先起 5800 段（SHOPSIM_BASE_PORT=5800）
#     bash scripts/collect_retry.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

POOL_NAME="${POOL_NAME:-sft_train_retry}"
OUT="outputs/teacher/${POOL_NAME}.jsonl"
TARGET="${TARGET:-$(wc -l < "data/task_pools/${POOL_NAME}.jsonl")}"
MAX_ROUNDS="${MAX_ROUNDS:-15}"
LOG=outputs/logs/collect_retry.log

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

log "###### D1 重采开始：目标 ${TARGET} 条 → ${OUT}"

for ((r = 1; r <= MAX_ROUNDS; r++)); do
  n=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  if [ "$n" -ge "$TARGET" ]; then log "已达标 ${n}/${TARGET}，停止"; break; fi
  log "第 ${r} 轮，当前 ${n}/${TARGET}"

  SHOPSIM_BASE_PORT=5800 SHOPSIM_WORKERS=8 \
    bash scripts/collect_teacher.sh "$POOL_NAME" >> outputs/logs/collect_retry_run.log 2>&1
  rc=$?

  after=$(wc -l < "$OUT" 2>/dev/null || echo 0)
  log "第 ${r} 轮结束（exit=${rc}）：${n} → ${after}"
  # 一轮零进展说明不是偶发中止，是真出事了，别空转烧 API 额度。
  if [ "$after" -le "$n" ]; then log "本轮零进展，放弃重试"; break; fi
  sleep 30
done

log "###### D1 循环结束，最终 $(wc -l < "$OUT" 2>/dev/null || echo 0)/${TARGET}"
