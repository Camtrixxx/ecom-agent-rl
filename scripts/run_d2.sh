#!/usr/bin/env bash
# D2 的无人值守链：SFT-v2 → GRPO-v2（主判定）→ SFT 3 对 3 多种子（次判定）。
# 判定规则全部写在 docs/d2-preregistration.md，跑之前就定死了，链中没有需要人拍板的岔路。
#
#     nohup bash scripts/run_d2.sh > outputs/logs/d2_chain.nohup 2>&1 &
#
# ## 卡的账（这条链的形状完全由它决定）
#
# SFT 训练要 6 卡（GPU 2-7），评测只要 1 卡。两者不冲突：SFT 训练不碰环境池，评测不碰
# GPU 2-7。所以每个「档位」= 一个 SFT 训练 ∥ 一个评测，这是本机能同时开的最大并行度。
# GRPO 不同：它自己要 GPU 1 起 vLLM + GPU 2-7 训练，**独占**，而且它的训练循环也用
# EnvironmentPool——任何评测与它并行都会抢同一段环境端口，env_error 是整批中止。
#
# 顺序把用户明确要的那两件（SFT-v2、GRPO-v2）排在最前，多种子塞进 GRPO 之后的空档。
#
#   A  SFT-v2 seed42 训练（已在链外起好，这里只等）
#   B  评 sft_v2      ∥ 训 sft_s43（v1 数据）
#   C  GRPO-v2                                    ← 独占，主判定
#   D  评 grpo_v2     ∥ 训 sft_v2_s43
#   E  评 sft_s43     ∥ 训 sft_s44
#   F  评 sft_v2_s43  ∥ 训 sft_v2_s44
#   G  评 sft_s44     ∥ 评 sft_v2_s44            ← 无训练占卡，两个评测并行（错开端口段）
#   H  出 3 对 3 判定
#
# ## 幂等
#
# 每一档都先查产物：SFT 看权重文件落没落盘（不是看目录在不在——train_sft.py 一开跑就建
# 目录写 train_log.jsonl），评测看轨迹行数够不够，GRPO 看 state.json 的轮数。所以这个脚本
# 中途挂了可以原样重跑，已完成的档位会跳过。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

LOG=outputs/logs/d2_chain.log
mkdir -p outputs/logs outputs/rollouts
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

POOL=data/task_pools/evaluation.jsonl
TARGET=$(( $(wc -l < "$POOL") * 4 ))
GRPO_ITERS=40

has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }
rollout_done()  { local n; n=$(wc -l < "$1" 2>/dev/null || echo 0); (( n >= TARGET )); }
grpo_iters() {
  python3 - "$1" <<'PY' 2>/dev/null || echo 0
import json, sys
from pathlib import Path
p = Path(sys.argv[1]) / "state.json"
print(json.loads(p.read_text())["iteration"] if p.exists() else 0)
PY
}

wait_gone() {  # wait_gone <pgrep 模式> <说明>
  # pgrep -f 会匹配到发起它的这个 shell 自己的 cmdline，模式里避开脚本名本身。
  if ! pgrep -f "$1" >/dev/null; then log "  ${2}：已不在运行"; return 0; fi
  log "  等 ${2} 结束……"
  while pgrep -f "$1" >/dev/null; do sleep 60; done
  log "  ${2} 已结束"
}

# 六卡 SFT。卡数显式给：train_sft.sh 默认 7 卡（GPU 1-7），而已发布的 v1 是 6 卡，
# global batch = grad_accum 4 × 卡数，7 卡 28 / 6 卡 24。卡数一错，「数据量」就和「卡数」
# 缠在一起，而那恰好是这个实验唯一要量的东西。
train_sft() {  # train_sft <名字> <train.jsonl> <seed>
  local name="$1" data="$2" seed="$3" out="outputs/models/$1"
  if has_weights "$out"; then log "  [训练] ${name}: 权重已在，跳过"; return 0; fi
  log "  [训练] ${name}: 开始（data=${data} seed=${seed} 6 卡 GPU2-7）"
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_sft.sh \
    --train "$data" --validation data/sft_val/train.jsonl \
    --out-dir "$out" --seed "$seed" >> "outputs/logs/${name}.log" 2>&1
  local rc=$?
  log "  [训练] ${name}: 结束 exit=${rc} 权重=$(has_weights "$out" && echo 有 || echo 无)"
}

eval_model() {  # eval_model <名字> <权重目录> <对照轨迹> [GPU] [vLLM端口] [环境端口段]
  local name="$1" model="$2" base="$3" gpu="${4:-1}" port="${5:-8180}" envp="${6:-5700}"
  if rollout_done "outputs/rollouts/${name}.jsonl"; then
    log "  [评测] ${name}: 已跑满 ${TARGET}，跳过"; return 0
  fi
  if ! has_weights "$model"; then log "  [评测] ${name}: 权重不存在 ${model}，跳过"; return 1; fi
  log "  [评测] ${name}: 开始（GPU${gpu}:${port} env${envp}）"
  GPU="$gpu" PORT="$port" ENV_BASE_PORT="$envp" \
    bash scripts/eval_model.sh "$name" "$model" "$base" \
    >> "outputs/logs/eval_${name}.nohup" 2>&1
  log "  [评测] ${name}: 结束 exit=$? 行数=$(wc -l < "outputs/rollouts/${name}.jsonl" 2>/dev/null || echo 0)/${TARGET}"
}

log "###### D2 链启动"

# --- A：等链外那个 SFT-v2 训完 ------------------------------------------------
# 匹配 train_sft.py 而不是 train_sft.sh：sh 用 exec 换成了 accelerate，进程名里只剩 py。
wait_gone "train_sft.py" "链外的 SFT-v2 训练"
if ! has_weights outputs/models/sft_v2; then
  log "!! SFT-v2 没有权重，整条链的前提不成立，退出"
  exit 1
fi
log "=== A 完成：outputs/models/sft_v2 就绪"

# --- B：评 sft_v2 ∥ 训 sft_s43 -----------------------------------------------
log "=== B 开始"
eval_model sft_v2 outputs/models/sft_v2 outputs/rollouts/sft.jsonl &
pid_eval=$!
train_sft sft_s43 data/sft/train.jsonl 43 &
pid_train=$!
wait "$pid_eval" "$pid_train"
log "=== B 完成"

# --- C：GRPO-v2，独占 GPU 1-7 ------------------------------------------------
log "=== C 开始（GRPO-v2，约 6.5 h，独占）"
n=$(grpo_iters outputs/models/grpo_v2)
if (( n >= GRPO_ITERS )); then
  log "  GRPO-v2 已跑满 ${n}/${GRPO_ITERS}，跳过"
else
  extra=()
  (( n > 0 )) && { log "  已有 ${n} 轮，--resume 续跑"; extra=(--resume); }
  # 超参与已发布的 seed 42 逐项一致，只改 --init-model 和 --out-dir。
  # snapshot-every 留默认 5（seed 42 就是默认），存快照不影响训练，只多占盘。
  bash scripts/train_grpo.sh \
    --init-model outputs/models/sft_v2 \
    --out-dir outputs/models/grpo_v2 \
    "${extra[@]}" >> outputs/logs/grpo_v2.log 2>&1
  rc=$?
  after=$(grpo_iters outputs/models/grpo_v2)
  log "  GRPO-v2 结束 exit=${rc}，轮数 ${n} → ${after}/${GRPO_ITERS}"
  (( after < GRPO_ITERS )) && log "!! 未跑满，结论里要写明实际轮数 ${after}"
fi
log "=== C 完成"

# --- D：评 grpo_v2（主判定的数）∥ 训 sft_v2_s43 -------------------------------
log "=== D 开始"
eval_model grpo_v2 outputs/models/grpo_v2/policy outputs/rollouts/grpo.jsonl &
pid_eval=$!
train_sft sft_v2_s43 data/sft_v2/train.jsonl 43 &
pid_train=$!
wait "$pid_eval" "$pid_train"
log "=== D 完成（主判定的轨迹已齐）"

# --- E / F：把剩下两个训练和两个评测继续错开 ----------------------------------
log "=== E 开始"
eval_model sft_s43 outputs/models/sft_s43 outputs/rollouts/sft.jsonl &
pid_eval=$!
train_sft sft_s44 data/sft/train.jsonl 44 &
pid_train=$!
wait "$pid_eval" "$pid_train"
log "=== E 完成"

log "=== F 开始"
eval_model sft_v2_s43 outputs/models/sft_v2_s43 outputs/rollouts/sft.jsonl &
pid_eval=$!
train_sft sft_v2_s44 data/sft_v2/train.jsonl 44 &
pid_train=$!
wait "$pid_eval" "$pid_train"
log "=== F 完成"

# --- G：没有训练占卡了，两个评测并行。vLLM 卡/端口和环境端口段都错开 ----------
log "=== G 开始（两个评测并行）"
eval_model sft_s44    outputs/models/sft_s44    outputs/rollouts/sft.jsonl 1 8180 5700 &
pid_a=$!
eval_model sft_v2_s44 outputs/models/sft_v2_s44 outputs/rollouts/sft.jsonl 2 8181 5800 &
pid_b=$!
wait "$pid_a" "$pid_b"
log "=== G 完成"

# --- H：出 3 对 3 判定 --------------------------------------------------------
log "=== H：3 对 3 判定"
.venv/bin/python scripts/analyze_d2.py 2>&1 | tee -a "$LOG"
log "###### D2 链结束"
