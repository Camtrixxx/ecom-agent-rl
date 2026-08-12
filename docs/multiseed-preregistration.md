# 多 seed GRPO 预注册

**写于开跑之前**，2026-08-13。这是一份**新的**预注册，不是对
[eval-preregistration.md](eval-preregistration.md) 的修订——那份末尾写着「本文件此后
不再改动」，这条要守住。多 seed 是一个新的假设检验，判定规则必须在看到 seed 43 / 44
的数字之前落定。

## 问题：+7.41 pp 的区间里少了一整项方差

已发布的结论是 GRPO policy vs SFT 配对差值 **+7.41 pp [+5.40, +9.51]**。那个区间由
`report_metrics.py` 对 498 道题做配对 bootstrap 得到，它回答的是：

> 同一个训练好的策略，在同一批题上**重采一遍**，差值会浮动多少。

它**不**回答：

> 换一次训练 run，差值会浮动多少。

后者在本项目里有一个具体、且不小的来源。`iteration_tasks`
（`scripts/train_grpo.py`）按 seed 把 3,000 题的池子打散一次，再按轮次取连续切片：
40 轮 × 24 题 = **960 道题，占池子 32%，且不绕回**。换 seed 等于换掉训练见过的那
三分之一题目。这不是「同一实验的随机抖动」，是两个见过不同数据的 run。

所以 between-run 方差**应当**显著大于 within-run 采样方差。现在报的区间完全没有这一项，
读者会把 [+5.40, +9.51] 误读成「重跑一次也会落在这里面」。本次实验就是去量那个缺失项。

## 口径

训练与评测口径**与已发布的 run 逐字一致**，只改 seed 与产物目录：

| 项 | 值 |
|---|---|
| 起点 | `outputs/models/sft`（**同一份** SFT 权重，三个 run 共用） |
| 轮数 / 每轮题数 / G | 40 / 24 / 8 |
| lr / warmup / max_grad_norm | 1e-6 / 5 / 1.0 |
| 训练温度 | 1.0 |
| seed | 43、44（42 是已发布的那个 run） |
| `--snapshot-every` | **0**（见下） |
| 评测池 / k | `data/task_pools/evaluation.jsonl` 500 题 / k=4 |
| 评测 temperature / top_p / max_tokens | 0.7 / 0.95 / 1024 |
| max_steps / 窗口 / concurrency | 35 / 24576 / 16 |
| 配对基线 | **同一份** `outputs/rollouts/.../sft` 轨迹，不重采 |

`--snapshot-every 0`：每个 run 的 `policy/` + 8 个中间快照 = 135 G，两个 run 就是 270 G。
这两个 run 的用途是量 run 间方差，不做 checkpoint 选择，中间快照留着没用。E1
（`eval_checkpoints.sh`）已经在 seed 42 的快照上回答了「是不是提前见顶」，不需要在
另外两个 seed 上重复。

**基线只评一次。** 三个 delta 都对同一份 SFT 轨迹配对——这是有意的：要量的是 GRPO
run 间的方差，把 SFT 也重采会把 SFT 的采样噪声混进来，而那一项已经在
eval-preregistration 里量过了。代价是三个 delta 之间存在共同的基线误差，因此它们
**不完全独立**；下面的 t 区间描述的是「GRPO 这一侧的 run 间散布」，这一点先写明。

**「seed」的实际含义。** vLLM 的采样本身没有 seed，所以即使传同一个 seed 也不会
逐比特复现。这里的 seed 是「一次独立的训练 run」的标签——而这恰好就是要测的东西，
不是缺陷。

## 主统计量与判定规则

三个 run 各自对同一份 SFT 得到一个配对差值 d₄₂、d₄₃、d₄₄（d₄₂ = +7.41 pp 已知）。

**主统计量：三者的均值 d̄。** 区间用 n=3 的 t 分布：

```
d̄ ± t₂,₀.₉₇₅ × s / √3 ,  t₂,₀.₉₇₅ = 4.30
```

**先把这个区间会有多宽算清楚**，免得事后觉得「不够显著」就想加 seed：

| 三个 delta 的标准差 s | 半宽 = 4.30 × s / √3 |
|---|---|
| 0.5 pp | ±1.24 pp |
| 1.0 pp | ±2.48 pp |
| 2.0 pp | ±4.97 pp |
| 3.0 pp | ±7.45 pp |

也就是说 **3 个 seed 只能给一个粗界，给不了紧区间**。这是 n=3 的固有代价，不是意外。

判定：

1. **t 区间不跨 0** → 结论升级为「GRPO 的提升在 run 间稳定，均值 d̄，区间 […]」，
   **停在 3 个 seed**。已发布的 +7.41 pp 保留，并注明它是三个 run 之一。
2. **t 区间跨 0** → 结论降级为「在 3 个 run 的分辨率下，无法确认 GRPO 的提升在 run 间
   稳定」。**不补第 4 个 seed。** 同时如实报出三个 delta 的原始值和 s。

规则 2 是这份预注册最重要的一条。「不显著就加 seed 直到显著」会把假阳性率从 5% 抬到
20% 以上，而它在过程中看起来完全合理。若 s 大到区间跨 0，那个信息本身就是结论——
它说明单 run 的 +7.41 pp 不足以支撑「GRPO 有效」，而这正是当初该做多 seed 的原因。

**不许丢 seed。** 崩掉或没跑满 40 轮的 run 按**实际轮数**如实报，不静默替换成一个新
seed。若某个 run 因基础设施原因（OOM、vLLM 起不来、环境服务挂）在第 k 轮中止，
优先 `--resume` 续跑；确实无法续跑时，该 run 以其最后一个完整轮次的权重参与评测，
并在结果里写明轮数不足。

## 预注册的次要预测

写在看到数字之前，用来区分「解释」和「事后编故事」：

1. **s 会落在 1–3 pp 区间。** 依据：换 seed 换掉 32% 的训练题，而训练期
   `gold_purchase` 从 56.2% 涨到 71.6%（15.4 pp 的训练侧变化）；留出集上的 run 间散布
   应当是它的一个小分数，但不会小到 0.5 pp 以下。若 s < 0.5 pp，说明 GRPO 的增益几乎
   与见过哪些题无关——那是一个比预期更强的结论，要单独核对是不是三个 run 塌到了同一个解。
2. **三个 delta 全为正。** 若出现负的 delta，则「GRPO 有效」这个结论本身要重写，
   无论均值是多少。
3. **`early_abstain` 在三个 run 上都保持 0。** 与 eval-preregistration 第 4 条同源：
   若某个 run 把弃买推回来，即使成功率上去也要单独报。

## 复现

```bash
# 两次串行，各 ~6.56 h，占 GPU 1-7
bash scripts/train_grpo.sh --seed 43 --out-dir outputs/models/grpo_s43 --snapshot-every 0
bash scripts/train_grpo.sh --seed 44 --out-dir outputs/models/grpo_s44 --snapshot-every 0

# 各自 500 题 k=4，与同一份 SFT 轨迹配对
bash scripts/eval_seed.sh grpo_s43
bash scripts/eval_seed.sh grpo_s44
```

合计 ~14.5 h wall-clock（13.1 h 训练 + 1.4 h 评测）。

---

**本文件此后不再改动。** 结果写进 [roadmap.md](roadmap.md)。
