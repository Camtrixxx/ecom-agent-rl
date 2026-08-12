#!/usr/bin/env bash
# S4：重训 SFT 两次，回答「第 3 个 epoch 值不值」。
#
#   a) sft_e3：3 epoch，按 epoch 存权重（epoch1/ epoch2/ 另存，最终权重在根）
#   b) sft_e2：独立的 2 epoch——余弦在 2 个 epoch 内退到 0，这才是「2-epoch 训练」，
#      而不是 3-epoch 曲线上的第 2 个 epoch。两者的差就是退火的贡献。
#
# 两次**串行**（都要占满同一批卡）。卡的分配：E1 评测占着 GPU 1-2（它在两张卡上交替
# 预热），所以这里用 GPU 3-7 五张。这一点必须记进结论里：per_rank = 648//5 = 129，
# steps_in_epoch = 129//4 = 32，3 个 epoch 共 96 步——和已发布的 SFT（7 卡）步数不同。
# S4 内部的三处比较（e3 vs e2 vs e3/epoch2）共用同一个 world_size，自身是自洽的；
# 且 S1 修好之后 val loss 已经与卡数无关。
#
#     bash scripts/run_s4.sh
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

GPUS="${GPUS:-3,4,5,6,7}"
VAL="${VAL:-data/sft_val/train.jsonl}"
LOG=outputs/logs/run_s4.log
mkdir -p outputs/logs

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

run() {  # run <out-dir> <epochs> [额外参数...]
  local out="$1" epochs="$2"; shift 2
  if [[ -f "${out}/metadata.json" ]]; then
    log "=== ${out} 已完成，跳过"
    return 0
  fi
  log "=== 开跑 ${out}（${epochs} epoch，GPU ${GPUS}）"
  CUDA_VISIBLE_DEVICES="$GPUS" bash scripts/train_sft.sh \
    --validation "$VAL" --out-dir "$out" --epochs "$epochs" "$@" \
    >> "outputs/logs/$(basename "$out").log" 2>&1
  local rc=$?
  log "=== ${out} 结束 exit=${rc}"
  return $rc
}

log "###### S4 开始"
run outputs/models/sft_e3 3 --save-each-epoch
run outputs/models/sft_e2 2
log "###### S4 结束"
