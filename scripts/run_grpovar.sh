#!/usr/bin/env bash
# #29（三份 GRPO 快照同窗口重评）的执行链。
# 判读规则见 docs/grpovar-preregistration.md，**本脚本不做任何判定**，只负责把腿跑齐。
#
#     nohup bash scripts/run_grpovar.sh > outputs/logs/grpovar_chain.nohup 2>&1 &
#
# 四条腿并行（GPU 1-4，vLLM 8180-8183，环境段 5700/5800/5900/6000）。GPU 0 上有别人的
# 常驻服务，不碰。约 1.2-1.5 h。
#
# **和 run_dayeval.sh 的唯一结构差别：跑前把四段环境池全杀掉重起。** 见下面那段注释。
#
# **脚本一旦在跑就不能编辑**：bash 按字节偏移读正在执行的脚本，改一行会让它跳到错的位置。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=outputs/logs/grpovar_chain.log
mkdir -p outputs/logs outputs/rollouts
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

SEGS=(5700 5800 5900 6000)
WORKERS=8

# ---- 闸门 ----
if pgrep -f "train_grpo.py|train_sft.py" >/dev/null; then
  log "!! 还有训练在跑，不启动（会抢卡，而且 GRPO 训练自己也用 EnvironmentPool）"
  exit 1
fi
for p in 8180 8181 8182 8183; do
  if curl -sf --noproxy '*' --max-time 3 "http://127.0.0.1:${p}/v1/models" >/dev/null 2>&1; then
    log "!! 端口 ${p} 上还有 vLLM 在听，先清掉再来"
    exit 1
  fi
done
avail=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
if (( avail < 20 )); then
  log "!! /data 只剩 ${avail}G，四条腿约需 0.6G 但留 20G 余量，不启动"
  exit 1
fi
# 三份 GRPO 权重 + 一份 SFT 必须都在，缺一条腿主判定就出不来（预注册写死 n=3）。
for w in outputs/models/grpo/policy outputs/models/grpo_s43/policy \
         outputs/models/grpo_s44/policy outputs/models/sft ; do
  [[ -f "$w/model.safetensors" || -f "$w/model.safetensors.index.json" ]] || {
    log "!! 权重不存在：${w}"; exit 1; }
done
log "闸门通过：无训练在跑，8180-8183 空，/data 余 ${avail}G，四份权重齐"

# ============================================================================
# 池龄统一（#30 第 1 项的第一次落地）
# ============================================================================
# 为什么要做这件事：#28 事后自查发现 run_dayeval.sh 按固定腿序分段，于是 v1 组拿到
# 5.4 天 / 3.2 天 / 全新的池，v2 组三段全新——**池龄和分组系统性地绑在一起**。那次实测
# 偏置只有 −0.06 pp、方向保守，判定没受影响，但界定它的数据来自另一条臂，纯属运气。
#
# 做法：把四段的 worker 全杀掉，让 eval_model.sh 的自愈逻辑（eval_model.sh:77-97）各起
# 一个新池。start_environment.sh 的 trap 会在任一 worker 退出时收掉整池（该脚本 :77-86），
# 所以杀 worker 就等于杀池，不用去找 supervisor 的 pid。
#
# 杀不掉不阻断：那样只是回到 #28 的状态（池龄不齐），比不跑这条臂强。但要**记进日志**，
# 好让事后能查——#28 那次的教训正是"一个查不到的变量没法在事后排除"。
seg_pids() {
  local base=$1 last=$(( $1 + WORKERS - 1 )) p
  for ((p = base; p <= last; p++)); do
    ss -ltnp 2>/dev/null | awk -v m=":${p}\$" '$4 ~ m {print $NF}' \
      | sed -n 's/.*pid=\([0-9]*\).*/\1/p'
  done | sort -u
}
log "###### 统一池龄：杀掉 ${SEGS[*]} 四段的现有池"
for seg in "${SEGS[@]}"; do
  pids=$(seg_pids "$seg")
  if [ -z "$pids" ]; then
    log "  段 ${seg}：本来就没在听，跳过"
    continue
  fi
  log "  段 ${seg}：杀 $(echo "$pids" | wc -l) 个 worker（$(echo "$pids" | tr '\n' ' ')）"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null
done
sleep 15
for seg in "${SEGS[@]}"; do
  left=$(seg_pids "$seg")
  if [ -n "$left" ]; then
    log "  !! 段 ${seg} 还有 worker 没退（$(echo "$left" | tr '\n' ' ')），"
    log "     再给 15 s，之后 -9"
    sleep 15
    left=$(seg_pids "$seg")
    # shellcheck disable=SC2086
    [ -n "$left" ] && kill -9 $left 2>/dev/null
  fi
done
sleep 5
for seg in "${SEGS[@]}"; do
  left=$(seg_pids "$seg")
  if [ -n "$left" ]; then
    log "  !! 段 ${seg} 仍有 worker 活着：池龄没能统一，这条臂带一个和 #28 同样的混杂。"
    log "     照跑，但判定要带这句限定。"
  else
    log "  段 ${seg} 已清空，等 eval_model.sh 起新池"
  fi
done

# ============================================================================
# 四条腿并行
# ============================================================================
# 配置逐字相同（k=4、温度 0.7、concurrency 16、同 500 题池），差别只有权重。
# 不传第三个参数（对照）→ 只出单侧报告；所有配对交给 analyze_grpovar.py 现算，
# 免得在这里把基线选择写死（主判定配旧 sft.jsonl，附带 2 配新 gv_sft，两个都要）。
#
# GPU / vLLM 端口 / 环境段三者都必须错开：EnvironmentPool 的 BoundedSemaphore 每个
# client pool 各一份、不跨进程，两条腿共用一段端口会各自以为租约是自己的 → env_error
# → **整批中止**。
log "###### 四条腿并行：三份 GRPO 快照 + 同窗口 SFT"
i=0
for spec in \
  "gv_grpo_s42:outputs/models/grpo/policy" \
  "gv_grpo_s43:outputs/models/grpo_s43/policy" \
  "gv_grpo_s44:outputs/models/grpo_s44/policy" \
  "gv_sft:outputs/models/sft" ; do
  name="${spec%%:*}"; weights="${spec#*:}"
  gpu=$(( 1 + i ))
  port=$(( 8180 + i ))
  seg=${SEGS[$i]}
  log "  起 ${name}：GPU${gpu}:${port} 段${seg} ← ${weights}"
  GPU="$gpu" PORT="$port" ENV_BASE_PORT="$seg" CONCURRENCY=16 \
    nohup bash scripts/eval_model.sh "$name" "$weights" \
    > "outputs/logs/grpovar_${name}.nohup" 2>&1 &
  i=$(( i + 1 ))
  sleep 20   # 错开 vLLM 启动，别让四个进程同时抢着从盘上读 7.6G 权重
done

log "  四条腿已起，等全部退出"
wait
log "  四条腿都退了"

TARGET=$(( $(wc -l < data/task_pools/evaluation.jsonl) * 4 ))
for name in gv_grpo_s42 gv_grpo_s43 gv_grpo_s44 gv_sft; do
  f="outputs/rollouts/${name}.jsonl"
  n=$(wc -l 2>/dev/null < "$f"); n=${n:-0}
  # grep -c 无匹配时退 1，所以不看退出码、只取 stdout（`|| echo 0` 会再追一个 0）。
  t=$(grep -c '"tolerant_parse"' "$f" 2>/dev/null); t=${t:-0}
  log "  ${name}: ${n}/${TARGET} 行，tolerant_parse ${t} 处"
done
log "###### 跑完。判定：.venv/bin/python scripts/analyze_grpovar.py"
