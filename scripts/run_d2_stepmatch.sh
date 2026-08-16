#!/usr/bin/env bash
# D2 追加臂：把「更多数据」和「更多 optimizer step」拆开。
#
#     nohup bash scripts/run_d2_stepmatch.sh 42 43 44 > outputs/logs/stepmatch.nohup 2>&1 &
#
# ## 为什么加这一臂
#
# 阶段 B 量到 sft_v2 − sft_v1 = **+5.65 pp**，比预注册的预测（0~+3 pp）大 3 倍。但预注册
# 里就写明有一个无法消除的混淆：固定 3 epoch 下加数据必然加 step，v1 81 步、v2 102 步，
# 总 LR 积分 41.50 → 52.00（+25.3%）。而 E2 实测过这条通道：在**看过一样多数据**的前提下
# 积分差 +39.0% 值 6.71 pp。线性外推到 +25.3% ≈ **4.4 pp**——也就是观测到的 5.65 pp 里
# 大部分可能根本不是数据的功劳。
#
# **这一臂的决定是看过 +5.65 pp 之后才做的，和 03:15 那次分层追加不同**（那次是盲的）。
# 记在这里是因为这个区别影响可信度：它之所以仍然正当，是因为它是**一个新实验**，不是把
# 同一批数据换个算法重新解读——目标没有被移动，是新采了一组能回答归因问题的数。
#
# ## 怎么拆
#
# 对照臂 = **v1 的数据 + v2 的 LR 计划**：
#
#   train_sft.py:266  total_steps = max(1, int(steps_per_epoch * epochs))
#   v1 数据 steps_per_epoch = 27，取 epochs 3.78 → int(102.06) = **102 步**
#   warmup = int(102 × 0.03) = **3**，与 sft_v2 逐项相同；余弦同样在 102 步退到 0
#
# 于是 `sft_stepmatch` 与 `sft_v2` 的 **LR 轨迹完全相同、optimizer step 完全相同、
# 处理的 batch 数和 token 数完全相同**（102 步 × 4 grad_accum × 6 卡）。唯一的差别是
# **不同样本数 1781 vs 2152**——也就是数据多样性。
#
#   sft_v2 − sft_stepmatch  = 纯数据多样性效应（同算力、同 LR 积分）
#   sft_stepmatch − sft_v1  = 纯 LR 积分效应（同数据）
#   两者相加应约等于已测到的 +5.65 pp —— 这是一个自洽性检查，不是三个独立的数
#
# 注意 `sft_stepmatch` 把 v1 数据看了 3.78 遍而 v2 只看了自己的 3.0 遍。这是**故意**的：
# 要匹配的是算力和 LR 积分，而「同样的算力下，样本重复更多次」恰好就是「数据更少」在
# 固定算力下的表现形式。
#
# ## 浮点数那个坑
#
# epochs 必须写 **3.78**，不能写 3.7777777：int() 是截断，27 × 3.7777777 = 101.9999979
# → **101 步**，整条对照就废了。3.78 → 102.06 → 102。脚本跑起来后会去日志里核对
# 「共 102 步」，不对就立刻停，不让它默默训出一个错的对照。
#
# ## 卡
#
# 6 卡 GPU 2-7，与 v1/v2 全部 run 一致（global batch = grad_accum × 卡数）。
# **不要和 GRPO 训练并发**：GRPO 要 GPU 1-7 且它的训练循环也用 EnvironmentPool。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SEEDS=("$@")
(( ${#SEEDS[@]} )) || SEEDS=(42)

LOG=outputs/logs/stepmatch_chain.log
mkdir -p outputs/logs outputs/rollouts
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

EXPECT_STEPS=102
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }
# 这里 2>/dev/null 必须写在 < "$1" **前面**：bash 按从左到右建立重定向，写在后面时
# 输入重定向已经失败并把 "No such file or directory" 打到了当时仍是终端的 stderr。
# 语义上「文件不存在 = 0 行 = 没跑完」是对的（不是在掩盖状态），漏的只是那行噪声。
rollout_done() { local n; n=$(wc -l 2>/dev/null < "$1" || echo 0); (( n >= 2000 )); }

# 先确认没有 GRPO / SFT 在跑，否则抢卡
if pgrep -f "train_grpo.py|train_sft.py" >/dev/null; then
  log "!! 还有训练在跑（train_grpo.py 或 train_sft.py），不启动。等链跑完再来。"
  exit 1
fi

for seed in "${SEEDS[@]}"; do
  name="sft_stepmatch_s${seed}"
  out="outputs/models/${name}"

  if has_weights "$out"; then
    log "[训练] ${name}: 权重已在，跳过"
  else
    log "[训练] ${name}: 开始（v1 数据 + epochs 3.78 → ${EXPECT_STEPS} 步，6 卡 GPU2-7）"
    CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 bash scripts/train_sft.sh \
      --train data/sft/train.jsonl --validation data/sft_val/train.jsonl \
      --out-dir "$out" --seed "$seed" --epochs 3.78 \
      >> "outputs/logs/${name}.log" 2>&1
    log "[训练] ${name}: 结束 exit=$? 权重=$(has_weights "$out" && echo 有 || echo 无)"
  fi

  # 核对步数：这一臂的全部意义就在于步数等于 102，错了就没有对照价值
  got=$(tr '\r' '\n' < "outputs/logs/${name}.log" 2>/dev/null \
        | grep -aoE "共 [0-9]+ 步" | tail -1 | grep -aoE "[0-9]+")
  if [[ "${got:-0}" != "$EXPECT_STEPS" ]]; then
    log "!! ${name} 的步数是 ${got:-未知}，预期 ${EXPECT_STEPS}——对照不成立，停在这里"
    exit 1
  fi
  log "  步数核对通过：${got} 步"

  if rollout_done "outputs/rollouts/${name}.jsonl"; then
    log "[评测] ${name}: 已跑满，跳过"
  else
    # 对照给 sft_v2：这一臂要回答的就是「同算力下多出来的数据值多少」
    #
    # 卡与端口显式错开到 GPU2 / 8181 / 环境段 5800（默认是 GPU1 / 8180 / 5700）。理由：
    # 08-15 起 GPU1 + 5700 段被宽容口径的重评长期占着，而两个 rollout 进程抢同一段环境
    # 端口会 env_error → **整批中止**（EnvironmentPool 的 BoundedSemaphore 每个客户端池
    # 各一份、不跨进程）。GPU2 在这一刻是空的：本 seed 的训练已经结束，下一个 seed 还没
    # 开始，评测和训练在同一个 seed 内是串行的。
    GPU=2 PORT=8181 ENV_BASE_PORT=5800 \
      bash scripts/eval_model.sh "$name" "$out" outputs/rollouts/sft_v2.jsonl \
      >> "outputs/logs/eval_${name}.nohup" 2>&1
    log "[评测] ${name}: 结束 exit=$? 行数=$(wc -l < "outputs/rollouts/${name}.jsonl" 2>/dev/null || echo 0)/2000"
  fi
done

log "###### 对照臂跑完，seeds: ${SEEDS[*]}"
log "接下来：把 sft_stepmatch_* 和 sft_v2_* 做 n 对 n，差值就是纯数据效应"
