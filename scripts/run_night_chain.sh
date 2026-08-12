#!/usr/bin/env bash
# 把剩下的实验串成一条无人值守的链：E1/S4 跑完 → E2 → R1（两个 seed + 两次评测）。
#
# 为什么要串而不是各自 nohup：卡是硬约束。E2 要 GPU 1-2（E1 正占着），R1 要 GPU 1-7
# 全部（E2 正占着 1-2）。而 R1 的卡数不能为了填满空闲卡而改——4 卡的 global batch 是
# 8 批、6 卡是 12 批，若两个 seed 卡数不同，卡数就和 seed 混在一起，恰好毁掉这个实验
# 要量的 run 间方差。所以 R1 必须等 E2 让出 GPU 1-2，宁可让 GPU 3-7 空闲一段。
#
# 中间没有决策点：multiseed-preregistration.md 已经写死「两个 seed 都跑完，区间跨 0 就
# 降级报结论、不补第 4 个 seed」。所以无人值守是安全的——不会有需要人来拍板的岔路。
#
#     nohup bash scripts/run_night_chain.sh > outputs/logs/night_chain.nohup 2>&1 &
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=outputs/logs/night_chain.log
mkdir -p outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

wait_gone() {  # wait_gone <pgrep 模式> <说明>
  if ! pgrep -f "$1" >/dev/null; then log "  ${2}：已不在运行"; return 0; fi
  log "  等 ${2} 结束……"
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  log "  ${2} 已结束"
}

log "###### 夜间链启动"

# --- E2：等 E1 让出 GPU 1-2，且等 S4 两个 run 都出了权重 -----------------------
wait_gone "eval_checkpoints.sh" "E1"
# E1 判出并列后触发了预注册的 k=8 补采，它占着 GPU1 和端口段 5700——E2 两样都要，
# 同端口段并行两个客户端池会互抢租约 → env_error → 整批中止。所以必须等它退出。
wait_gone "topup_k8.sh" "E1 的 k=8 补采"
wait_gone "run_s4.sh" "S4"

missing=0
for d in outputs/models/sft_e3 outputs/models/sft_e2 outputs/models/sft_e3/epoch2; do
  [[ -d "$d" ]] || { log "!! 缺 ${d}"; missing=1; }
done
if ((missing)); then
  log "!! S4 的产物不齐，E2 会跳过缺的那几份——照跑，报告里如实少几行"
fi

log "=== E2 开始"
bash scripts/eval_sft_variants.sh >> outputs/logs/eval_sft_variants.nohup 2>&1
log "=== E2 结束 exit=$?"
python3 scripts/summarize_reports.py outputs/rollouts/sft_variants --halfwidth 3.2 \
  --order sft sft_e2 sft_e3_ep2 sft_e3 2>&1 | tee -a "$LOG"

# --- R1：两个 seed，卡数与已发布的 seed 42 一致（GPU_VLLM=1, GPUS=2-7 是默认值）---
ITERATIONS=40

done_iters() {  # done_iters <out-dir> —— 读 state.json 的轮数，没有就 0
  python3 - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
from pathlib import Path
path = Path(sys.argv[1]) / "state.json"
print(json.loads(path.read_text())["iteration"] if path.exists() else 0)
PY
}

for seed in 43 44; do
  out="outputs/models/grpo_s${seed}"
  # 不能拿 policy/ 的存在判断「训完了」——它每轮都被覆盖写，跑了一轮也有。看 state.json。
  n=$(done_iters "$out")
  if (( n >= ITERATIONS )); then
    log "=== seed ${seed}: 已跑满 ${n}/${ITERATIONS} 轮，跳过训练"
    continue
  fi
  extra=()
  if (( n > 0 )); then
    log "=== seed ${seed}: 已有 ${n}/${ITERATIONS} 轮，--resume 续跑"
    extra=(--resume)
  else
    log "=== seed ${seed}: GRPO 开始（约 6.56 h）"
  fi
  bash scripts/train_grpo.sh --seed "$seed" --out-dir "$out" --snapshot-every 0 \
    "${extra[@]}" >> "outputs/logs/grpo_s${seed}.log" 2>&1
  rc=$?
  after=$(done_iters "$out")
  log "=== seed ${seed}: GRPO 结束 exit=${rc}，轮数 ${n} → ${after}/${ITERATIONS}"
  # 预注册里写死了「不许丢 seed」：没跑满就按实际轮数如实报，不静默换一个新 seed。
  (( after < ITERATIONS )) && log "!! seed ${seed} 未跑满，结论里要写明实际轮数 ${after}"
done

# 评测放在两个训练之后：训练要 GPU 1-7，评测只要 1 张，先把贵的排完。
for seed in 43 44; do
  out="outputs/models/grpo_s${seed}"
  # 这里用 policy/ 是对的：评测只要有一份权重就能跑，轮数不足的按实际轮数如实报。
  if [[ ! -d "${out}/policy" ]]; then
    log "!! seed ${seed}: 没有 policy，跳过评测"
    continue
  fi
  log "=== seed ${seed}: 评测开始"
  bash scripts/eval_seed.sh "grpo_s${seed}" >> "outputs/logs/eval_grpo_s${seed}.nohup" 2>&1
  log "=== seed ${seed}: 评测结束 exit=$?"
done

log "###### 夜间链结束"
