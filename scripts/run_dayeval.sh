#!/usr/bin/env bash
# #28（六份权重同窗口重评）+ #27（只改 concurrency）的执行链。
# 判读规则见 docs/dayeval-preregistration.md，**本脚本不做任何判定**，只负责把腿跑齐。
#
#     nohup bash scripts/run_dayeval.sh > outputs/logs/dayeval_chain.nohup 2>&1 &
#
# 为什么两条臂串行而不是九条腿一起：#27 的三条腿必须彼此同窗口且不能被 #28 的腿当负载搅进去
# （#27 查的就是负载/批构成），而且只有 GPU 1-7 可用（GPU 0 有别人的常驻服务，不碰）。
#
# **脚本一旦在跑就不能编辑**：bash 按字节偏移读正在执行的脚本，改一行会让它跳到错的位置。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=outputs/logs/dayeval_chain.log
mkdir -p outputs/logs outputs/rollouts
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

TARGET=$(( $(wc -l < data/task_pools/evaluation.jsonl) * 4 ))

# 「跑完了」判定：文件行数到目标。文件不存在 = 0 行 = 没跑完（语义正确，不是在掩盖状态）。
# 2>/dev/null 必须在 < "$1" 前面，否则输入重定向已经把报错打到终端了。
done_rows() { local n; n=$(wc -l 2>/dev/null < "$1" || echo 0); echo "$n"; }

# ---- 闸门 ----
if pgrep -f "train_grpo.py|train_sft.py" >/dev/null; then
  log "!! 还有训练在跑，不启动（会抢卡，而且 GRPO 训练自己也用 EnvironmentPool）"
  exit 1
fi
for p in 8180 8181 8182 8183 8184 8185; do
  if curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:${p}/v1/models" >/dev/null 2>&1; then
    log "!! 端口 ${p} 上还有 vLLM 在听，先清掉再来"
    exit 1
  fi
done
avail=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
if (( avail < 20 )); then
  log "!! /data 只剩 ${avail}G，九条腿约需 1.4G 但留 20G 余量，不启动"
  exit 1
fi
log "闸门通过：无训练在跑，8180-8185 空，/data 余 ${avail}G"

# ============================================================================
# 臂一 · #28：六份权重，同一个窗口，六条腿并行
# ============================================================================
# 配置逐字相同（k=4、温度 0.7、concurrency 16、同池），差别只有权重。
# 不传第三个参数（对照）→ 只出单侧报告；窗口内的两两配对交给 analyze_dayeval.py 现算，
# 免得在这里把某一个基线选择写死。
#
# GPU / vLLM 端口 / 环境段三者都必须错开：环境池的 BoundedSemaphore 每个 client pool 各一份、
# 不跨进程，两条腿共用一段端口会各自以为租约是自己的 → env_error → **整批中止**。
log "###### 臂一 #28：六份权重同窗口重评，六条腿并行"
i=0
for spec in \
  "dw_sft:outputs/models/sft" \
  "dw_sft_s43:outputs/models/sft_s43" \
  "dw_sft_s44:outputs/models/sft_s44" \
  "dw_sft_v2:outputs/models/sft_v2" \
  "dw_sft_v2_s43:outputs/models/sft_v2_s43" \
  "dw_sft_v2_s44:outputs/models/sft_v2_s44" ; do
  name="${spec%%:*}"; weights="${spec#*:}"
  gpu=$(( 1 + i ))
  port=$(( 8180 + i ))
  seg=$(( 5700 + i * 100 ))
  log "  起 ${name}：GPU${gpu}:${port} 段${seg} ← ${weights}"
  GPU="$gpu" PORT="$port" ENV_BASE_PORT="$seg" CONCURRENCY=16 \
    nohup bash scripts/eval_model.sh "$name" "$weights" \
    > "outputs/logs/dayeval_${name}.nohup" 2>&1 &
  i=$(( i + 1 ))
  sleep 20   # 错开 vLLM 启动，别让六个进程同时抢着从盘上读 7.6G 权重
done

log "  六条腿已起，等全部退出（每条约 45-70 min，并行下会更慢）"
wait
log "  六条腿都退了"

ok=0
for name in dw_sft dw_sft_s43 dw_sft_s44 dw_sft_v2 dw_sft_v2_s43 dw_sft_v2_s44; do
  n=$(done_rows "outputs/rollouts/${name}.jsonl")
  log "    ${name}: ${n}/${TARGET}"
  (( n >= TARGET )) && ok=$(( ok + 1 ))
done
log "  完整的腿 ${ok}/6"

if (( ok == 6 )); then
  log "###### 跑 #28 判定"
  .venv/bin/python scripts/analyze_dayeval.py 2>&1 | tee -a "$LOG"
else
  log "!! 只有 ${ok}/6 条腿完整，**不跑判定**：预注册写的是 n=3 对 n=3、t₄=2.776，"
  log "   缺腿会让 df 变成看完数据才定的东西。补齐缺的腿再手动跑 analyze_dayeval.py。"
fi

# ============================================================================
# 臂二 · #27：同权重同窗口，只改 concurrency
# ============================================================================
# 三条腿必须彼此同窗口，所以并行；而且必须等臂一全退，否则臂一的腿会当成额外负载
# 搅进来——这条臂查的正是负载/批构成。
log "###### 臂二 #27：sft_v2 同窗口三条腿，concurrency 8 / 16 / 32"
for p in 8180 8181 8182; do
  while curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:${p}/v1/models" >/dev/null 2>&1; do
    log "  端口 ${p} 上 vLLM 还没退（臂一的腿在收尾），等"
    sleep 30
  done
done

i=0
for c in 8 16 32; do
  name="cc_sft_v2_c${c}"
  gpu=$(( 1 + i )); port=$(( 8180 + i )); seg=$(( 5700 + i * 100 ))
  log "  起 ${name}：GPU${gpu}:${port} 段${seg} concurrency=${c}"
  GPU="$gpu" PORT="$port" ENV_BASE_PORT="$seg" CONCURRENCY="$c" \
    nohup bash scripts/eval_model.sh "$name" outputs/models/sft_v2 \
    > "outputs/logs/dayeval_${name}.nohup" 2>&1 &
  i=$(( i + 1 ))
  sleep 20
done
log "  三条腿已起（c=8 那条最慢，约 2-3 h；c=32 最快）"
wait
log "  三条腿都退了"

ok=0
for c in 8 16 32; do
  n=$(done_rows "outputs/rollouts/cc_sft_v2_c${c}.jsonl")
  log "    cc_sft_v2_c${c}: ${n}/${TARGET}"
  (( n >= TARGET )) && ok=$(( ok + 1 ))
done

if (( ok == 3 )); then
  log "###### 跑 #27 判定"
  .venv/bin/python scripts/analyze_concurrency.py 2>&1 | tee -a "$LOG"
else
  log "!! 只有 ${ok}/3 条腿完整，不跑判定。"
fi

log "###### 链结束"
