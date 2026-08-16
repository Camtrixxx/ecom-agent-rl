#!/usr/bin/env bash
# 找 grpo_v2 的「走到终局率」是哪一轮塌的，以及有没有一个更早的快照是健康的。
#
#     nohup bash scripts/eval_grpo_v2_snapshots.sh > outputs/logs/grpo_v2_snap.nohup 2>&1 &
#
# ## 为什么做这个，而不是先跑步数对照臂
#
# 08-14 12:05 的主判定：端到端 GRPO-v2 比 GRPO-v1 差 6.29 pp [−8.62, −3.96]。但按乘性
# 分解，**退化全在走到终局率**：
#
#     　　　　　　终局率   条件买对率   成功率
#     sft_v2      0.9945    0.6663     0.6626
#     grpo_v2     0.8615    0.7226     0.6225      终局 ×0.866，条件 ×1.084
#     grpo (v1)   0.9975    0.6855     0.6838      终局 ×1.003，条件 ×1.125
#
# grpo_v2 的**条件买对率 0.7226 是四个模型里最高的**（比 grpo_v1 高 3.71 pp）。反事实：
# 若保住 sft_v2 自己的终局率 0.9945，成功率会是 0.7186，反过来比 grpo_v1 高 3.5 pp。
# 也就是说 v2 数据很可能是有用的，只是被一个单一退化行为整个盖住了（no_tool_call
# 从 7 条涨到 272 条）。**先定位这个退化，再决定要不要否掉 v2 数据。**
#
# 训练日志里 no_terminal 一直只有 2-5%（temp 1.0），而评测是 13.6%（temp 0.7）。温度
# 更低反而更严重，符合「退化模式已经是 argmax 附近、低温把它放大」。所以必须在**评测
# 口径**下看这条曲线，训练日志看不出来。
#
# ## 池子用 grpo_val（200 题），不是 500 题评测池
#
# 沿用 E1 的规矩：**在哪个池上选 checkpoint，就不能在同一个池上报它的差值**，否则是在
# test set 上做模型选择。选出健康快照之后，再单独对 500 题池评一次出可比的数。
#
# 分辨率（E1 实测）：n=200、k=4 配对半宽约 ±3.2 pp。这**不够**给相邻快照的成功率排名，
# 但完全够看终局率——要找的是 0.99 → 0.86 这种 13 pp 级别的塌陷，不是几个点的差别。
#
# ## 卡与端口
#
# 默认单卡（GPU 1），不做 E1 那种双卡交替预热——为的是能和「步数对照臂」的 6 卡 SFT
# 训练同时跑。代价是每个模型多等约 3 分钟加载，9 个模型多约 27 分钟，可接受。
#
# 环境端口段用 **5700**，也就是唯一那个长跑的池子。原来写的是 5900「避开 run_d2.sh」，
# 那是错的：`eval_model.sh` / `eval_checkpoints.sh` 自己都不起环境池，段里没有服务在听
# 就只会把 `wait_until_ready` 的 600s 白等一遍，10 个模型等 100 分钟然后全跳过。
# 而「避开」这件事本来也不需要——下面已经硬性拒绝在有 run_rollout.py 时启动，rollout
# 已经串行化了，同一段端口不可能被两个进程同时抢。
#
# 但**不能和任何别的 rollout 并发**：EnvironmentPool 的 BoundedSemaphore 是每个客户端池
# 各一份、不跨进程，两个 rollout 进程抢同一段端口 → env_error → 整批中止。所以下面硬性
# 拒绝在有 run_rollout.py 时启动，而只允许和训练并存。
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT="$PWD"

POOL=data/task_pools/grpo_val.jsonl
ATTEMPTS="${ATTEMPTS:-4}"
CONCURRENCY="${CONCURRENCY:-16}"
ENV_BASE_PORT="${ENV_BASE_PORT:-5700}"
ENV_WORKERS="${ENV_WORKERS:-8}"
GPU="${GPU:-1}"
PORT="${PORT:-8192}"
TARGET=$(( $(wc -l < "$POOL") * ATTEMPTS ))
OUTDIR=outputs/rollouts/grpo_v2_snap
LOG=outputs/logs/grpo_v2_snap.log

export no_proxy='*' NO_PROXY='*'
mkdir -p "$OUTDIR" outputs/logs
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 条目格式 名字:权重目录:采样温度。
#
# sft_v2 是共同基线（配对消掉题目难度方差）。它也是 iter 曲线的起点：GRPO 的第 0 轮。
#
# 最后一条 `iter039_t10` 是**判定臂**，不是曲线的一部分：同一份权重、同一个池子，只把
# 温度从 0.7 换成 1.0（= GRPO 训练时的采样温度）。
#
# 为什么值得单独花 20 分钟：训练日志里 iter039 的 no_terminal 是 9/192 = **4.7%**（温度
# 1.0），而 500 题评测同一份权重是 **13.85%**（温度 0.7）。**降温反而让病更重，差 2.9 倍。**
# 格式坏掉的输出住在分布的尾巴上，降温应该让它变少；升温才让它变多。所以「kl_beta=0
# 导致格式漂移」这个假设**方向就是反的**。降温变重只有一种解释：不发工具调用已经是那些
# 状态下的**众数附近**行为，降温把它放大。那是策略众数真的移了，不是采样噪声。
#
# 但上面这个对比有一个混淆没法忽略：4.7% 出自**训练池**采的 24 题，13.85% 出自 500 题
# **评测池**，两个池的难度分布不同。而按难度分层看，no_tool_call 恰好是 11.0% / 10.5% /
# 14.6% / **27.2%**（1-2 / 3-4 / 5-7 / 8+），也就是说池子的难度构成本身就能造出几倍的
# 差别。**所以温度和池子现在缠在一起，光凭已有数据分不开。** 这条臂就是来拆它的：
# 同池同权重只改温度，如果 t=1.0 掉回 5% 附近 → 温度是放大器，训练期监控在结构上就看
# 不见这个病；如果 t=1.0 仍然十几个点 → 是池子差异，训练日志那 4.7% 本来就没有可比性。
# 两种结果都改变下一个 run 该怎么配，所以这一条比曲线上任何单个点都值钱。
#
# 注意：t=1.0 的数**不能**和已发布的任何成功率并列——评测口径一律 0.7。它只用来解释
# 这一个退化，标签里带 `_t10` 就是提醒。
MODELS=(
  "sft_v2:${ROOT}/outputs/models/sft_v2:0.7"
  "iter004:${ROOT}/outputs/models/grpo_v2/iter004:0.7"
  "iter009:${ROOT}/outputs/models/grpo_v2/iter009:0.7"
  "iter014:${ROOT}/outputs/models/grpo_v2/iter014:0.7"
  "iter019:${ROOT}/outputs/models/grpo_v2/iter019:0.7"
  "iter024:${ROOT}/outputs/models/grpo_v2/iter024:0.7"
  "iter029:${ROOT}/outputs/models/grpo_v2/iter029:0.7"
  "iter034:${ROOT}/outputs/models/grpo_v2/iter034:0.7"
  "iter039:${ROOT}/outputs/models/grpo_v2/iter039:0.7"
  "iter039_t10:${ROOT}/outputs/models/grpo_v2/iter039:1.0"
)

# 原来这两道闸门是直接 exit 1。改成**等**，理由是操作性的：Bash 工具的安全分类器在
# 间歇性超时，能不能在某个特定时刻成功发出命令是不确定的。改成等之后，这个脚本任何
# 时刻启动都安全——早了就自己排队，于是「现在launch一次」就能覆盖「链条跑完后再开始」，
# 不必守着某个时间点再发一次命令。
#
# 等的对象是**整条 run_d2.sh**，不是只等 run_rollout.py：链条的阶段 E/F 里评测只占约
# 1 h、训练占约 2.1 h，中间有约 1 h 没有 run_rollout.py 的空窗。若只等 run_rollout.py，
# 就会在空窗里起跑，然后在下一个阶段的评测开始时撞上去——而闸门只在启动时查一次，
# 撞上了不会自己退。所以必须等整条链。
#
# 设 SNAP_NO_WAIT=1 可以恢复「占着就直接退」的老行为。
WAIT_FOR="run_d2.sh|run_rollout.py|train_grpo.py"
if pgrep -f "$WAIT_FOR" >/dev/null; then
  if [[ -n "${SNAP_NO_WAIT:-}" ]]; then
    log "!! 有 ${WAIT_FOR} 在跑，且 SNAP_NO_WAIT=1，直接退出"
    exit 1
  fi
  log "=== 有 run_d2.sh / run_rollout.py / train_grpo.py 在跑，排队等它们结束"
  log "    （两个 rollout 抢同一段环境端口会 env_error 整批中止；GRPO 训练还独占 GPU 1-7）"
  waited=0
  while pgrep -f "$WAIT_FOR" >/dev/null; do
    sleep 60
    waited=$((waited + 1))
    (( waited % 30 == 0 )) && log "    已等 $((waited / 60)) h $((waited % 60)) min"
  done
  log "=== 占用者都结束了，开始扫快照"
  sleep 30   # 给 vLLM 的显存回收留一点余量
fi

# 硬闸门：环境代码变了就和 baseline / SFT / GRPO-v1 不在同一口径，跑了也不能比。
if ! python3 scripts/hash_environment.py >> "$LOG" 2>&1; then
  python3 scripts/hash_environment.py >&2
  log "!! 环境代码与锚不一致，评测中止"
  exit 1
fi

# 判「能不能评」看权重落没落盘，不是看目录在不在——目录可能是训练中途建的空壳，
# vLLM 会卡满 900s 超时（08-13 06:30 白烧过 43 分钟）。
has_weights() { [[ -f "$1/model.safetensors" || -f "$1/model.safetensors.index.json" ]]; }
# `2>/dev/null` 在 `< "$1"` 之前：文件不存在时报错的是 shell，写在后面时 stderr 还没
# 改道，日志里就会多一行 "No such file or directory"。计数一直是对的（`|| echo 0`），
# 只是噪声。
done_count() { wc -l 2>/dev/null < "$1" || echo 0; }

log "###### grpo_v2 快照扫描：${#MODELS[@]} 个模型 × ${TARGET} 回合（grpo_val k=${ATTEMPTS}）"

for entry in "${MODELS[@]}"; do
  IFS=: read -r name path temp <<< "$entry"
  out="${OUTDIR}/${name}.jsonl"

  if ! has_weights "$path"; then log "!! ${name}: 权重不存在 ${path}，跳过"; continue; fi
  if (( $(done_count "$out") >= TARGET )); then
    log "=== ${name}: 已有 $(done_count "$out")/${TARGET}，跳过"; continue
  fi

  CUDA_VISIBLE_DEVICES="$GPU" LLM_PORT="$PORT" nohup bash scripts/serve_model.sh "$path" \
    > "outputs/logs/vllm_snap_${name}.log" 2>&1 &
  pid=$!
  log "=== ${name}: vLLM 启动 pid=${pid} GPU${GPU}:${PORT} temp=${temp}"

  deadline=$((SECONDS + 900)); ready=0
  while ((SECONDS < deadline)); do
    curl -sf --noproxy '*' "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 \
      && { ready=1; break; }
    sleep 10
  done
  if (( ! ready )); then
    log "!! ${name}: vLLM 900s 未就绪，跳过"
    kill "$pid" 2>/dev/null; sleep 10; kill -9 "$pid" 2>/dev/null
    continue
  fi

  for attempt in 1 2 3; do
    n=$(done_count "$out")
    (( n >= TARGET )) && break
    log "  ${name}: 第 ${attempt} 次 rollout，当前 ${n}/${TARGET}"
    LLM_BASE_URL="http://127.0.0.1:${PORT}/v1" LLM_MODEL=ecom-agent \
      .venv/bin/python scripts/run_rollout.py \
        --pool "$POOL" --out "$out" \
        --attempts "$ATTEMPTS" --concurrency "$CONCURRENCY" \
        --temperature "$temp" \
        --env-base-port "$ENV_BASE_PORT" --env-workers "$ENV_WORKERS" >> "$LOG" 2>&1
    after=$(done_count "$out")
    log "  ${name}: ${n} → ${after}/${TARGET}"
    (( after <= n )) && { log "  ${name}: 零进展，放弃重试"; break; }
    sleep 20
  done

  kill "$pid" 2>/dev/null; sleep 15; kill -9 "$pid" 2>/dev/null
  sleep 5
done

log "###### rollout 结束，出报告"
for entry in "${MODELS[@]}"; do
  IFS=: read -r name _ _ <<< "$entry"
  out="${OUTDIR}/${name}.jsonl"
  [[ -s "$out" ]] || continue
  extra=(); [[ "$name" != "sft_v2" ]] && extra=(--baseline "${OUTDIR}/sft_v2.jsonl")
  .venv/bin/python scripts/report_metrics.py \
    --trajectories "$out" "${extra[@]}" --pool "$POOL" \
    --json "${OUTDIR}/${name}.report.json" >> "$LOG" 2>&1
  log "报告 ${name} exit=$? 行数=$(done_count "$out")"
done

# 主指标是**标签率**：content 里出现字面量 `<tool_call>` 的轨迹占比。
#
# 为什么不是成功率、也不是终局率：08-14 13:20 直接读轨迹定性了，272 条 no_tool_call 里
# 256 条（94.1%）是把工具调用写成了正文里的 Hermes 标签，而不是原生 tool_calls 字段。
# 所以标签率是这条因果链的**源头**；终局率是它经过 vLLM hermes 解析器之后的下游后果，
# 中间隔着「首块合法但被尾部垃圾整体带走」这一层损耗；成功率在 n=200 下根本分辨不出
# 相邻快照（配对半宽约 ±3.2 pp）。
log "###### 标签率 / 终局率曲线"
.venv/bin/python - "$OUTDIR" <<'PY' 2>&1 | tee -a "$LOG"
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
names = ["sft_v2"] + [f"iter{i:03d}" for i in (4, 9, 14, 19, 24, 29, 34, 39)] + ["iter039_t10"]
# `复读` 列比标签率本身更能分机制：grpo_v2 那 256 条标签轨迹的块数分布是**双峰**的。
# 08-14 14:06 用 scripts/analyze_tag_recovery.py 实测（不再是 grep 估）：
#
#     块数   1    2   3-4  5-9  10-29  30-99  100+
#     条数  48   88    4    2     17     97     0
#
# 即 136 条（53.1%）≤2 块，97 条（37.9%）在 30-99，中间几乎是空的。两个峰要修的东西不
# 一样，所以分开报：`少量` = 1-2 块，`复读` = ≥30 块。
#
# **口径已定死：一条轨迹的块数就等于一个回复的块数。** 源码 agent.py:233-238 一旦拿不到
# tool_calls 就立刻 return，所以标签只可能出现在终止那一条 assistant 消息里；实测
# `非终局标签` 四个模型全是 0，256/256 印证。（中途我一度以为 1024 那条论证作废，是算错
# 了——实测块长中位 45 字符 ≈ 15 token，反推成立：那些块是残缺短 JSON。别再往回写。）
#
# 背景率 0/6000（sft / grpo_v1 / sft_v2 三个模型的评测轨迹里 ≥3 块的一条都没有）同样是
# 轨迹级计的，跨臂比较同口径。
#
# **看 `少量` 列时记住**：那一峰里绝大多数是**能救回来的**——206/256（80.5%）宽容重解析能
# 取到合法首块。所以 `少量` 高不等于策略坏，它更多是 hermes 全有或全无 + 我们没兜底造成的
# 记账损失。真正反映策略退化的是 `复读` 列。
print(f"{'模型':<12}{'标签率':>8}{'标签数':>7}{'少量':>6}{'复读':>6}"
      f"{'终局率':>9}{'no_tool':>9}{'成功率':>9}{'条件买对':>10}{'env步':>7}")
for n in names:
    p = d / f"{n}.report.json"
    traj = d / f"{n}.jsonl"
    # 标签率直接数行：一条轨迹只要 content 里出现过标签就算一条，和 grep -c 同口径，
    # 这样这张表里的数可以和 optimization-log 里用 grep 量出来的训练期数字直接并列。
    tag = tot_lines = few = spew = 0
    if traj.exists():
        with traj.open() as f:
            for line in f:
                tot_lines += 1
                c = line.count("<tool_call>")   # `</tool_call>` 不含子串 `<tool_call>`，不会重复计
                if c:
                    tag += 1
                    if c <= 2:
                        few += 1
                    elif c >= 30:
                        spew += 1
    rate = tag / tot_lines if tot_lines else float("nan")
    if not p.exists():
        print(f"{n:<12}{rate:>8.4f}{tag:>7}{few:>6}{spew:>6}   （缺报告）"); continue
    o = json.loads(p.read_text())["overall"]
    tot = o["trajectories"]; st = o["statuses"]
    term = st.get("done", 0) / tot
    sr = o["success_rate"]["mean"]
    print(f"{n:<12}{rate:>8.4f}{tag:>7}{few:>6}{spew:>6}{term:>9.4f}"
          f"{st.get('no_tool_call', 0):>9}{sr:>9.4f}"
          f"{(sr / term if term else float('nan')):>10.4f}"
          f"{o['mean_env_steps']['mean']:>7.2f}")

# 预注册的判读（08-14 13:40 写下，14:05 细化，都在跑之前）：训练期（温度 1.0、训练池）
# 标签率全程平在 0.53%，而 iter039 在评测口径（温度 0.7、500 题池）是 12.8%，差 24 倍；
# v1 同口径反而是 ÷3.1。方向翻转说明主效应不是池子也不是温度，剩下的是交互项。
#
#   - t10 掉到 3% 附近  → 温度是放大器，训练温度下的 in-loop 监控在结构上看不见这个病。
#   - t10 仍是十几个点  → 放大器是池子难度，训练期那 0.53% 从一开始就没有可比性。
#
# 14:05 加的更细一层，也更该看：如果那 38% 的 `复读` 峰是**低温复读退化**（贪心重复是
# 低温的已知病，不是高温的），那么升到 1.0 时应当是 `复读` 列塌得比 `少量` 列多——即
# 不只是总标签率降，而是**双峰里那个高块数的峰优先消失**。若 t10 的 `复读` 基本不降而
# `少量` 降了，那复读就不是温度造的，得回到「策略本身学会了复读」，那要靠 iter 曲线看
# 它是哪一轮长出来的。这一条比总标签率的升降更有判别力，因为两个峰的修法不一样。
# 判读 C（08-14 14:20 写下）：训练 rollout 每一轮都存着（40 轮 × 192 条），所以「哪一轮
# 长出来的」这件事**在温度 1.0 下已经免费定位过了**——用「一条回复里 ≥3 个标签块」去扫，
# 40 轮里只有 iter039 命中（阈值放到 5 块、30 块答案一样，iter039 里 3 条 ≥30 块）。而
# checkpoint 只每 5 轮存一个，所以训练 rollout 的时间分辨率反而比本脚本的快照更细。
#
# 于是本脚本要回答的其实是另一个问题：**在评测温度、更大 n 下，这个「只有最后一轮」还成
# 立吗？** 训练 rollout 那边是 192 条/轮、温度 1.0（弱条件），可能根本看不见缓慢累积。
#   - iter004…034 全部贴着 sft_v2 的水平，iter039 单点跳到十几个点
#       → 与训练 rollout 一致：是最后几步突然长出来的。iter034 可以直接当产物用。
#   - 标签率从 iter024/029 就开始单调爬
#       → 训练期那个「只有 iter039」是 n=192 不够，退化一直在悄悄累积。
#         这会把结论从「checkpoint 选择就能救」改成「训练目标本身要加锚」。
print("\n预注册判读 A：iter039 vs iter039_t10 同池同权重只差温度。"
      "t10≈3% → 温度是放大器；t10 仍十几个点 → 池子难度是放大器。")
print("预注册判读 B：若复读峰是低温退化，t10 应当是「复读」列塌得比「少量」列多；"
      "若「复读」不降，则是策略学会了复读。"
      "（尺度参考，14:06 实测更新：温度 1.0 训练期复读 3/192=1.6%，温度 0.7 评测"
      " 97/2000=4.85%，但 3/192 的 95% 区间约 0.3%-4.6%，所以预期是「明显低但不为零」"
      "而非归零。另：复读峰的形状已量过——众数块占比中位 0.444、去重率中位 0.167，"
      "即在少数变体间循环、由一个块主导，不是逐字复读也不是重试不同参数。）")
print("预注册判读 C：iter004…034 贴着 sft_v2、iter039 单点跳 → 最后几步突然长出来，"
      "iter034 可直接当产物；若从 iter024/029 就单调爬 → 一直在累积，"
      "结论要从「选 checkpoint 就够」改成「训练目标要加锚」。")
PY

log "###### 完成"
