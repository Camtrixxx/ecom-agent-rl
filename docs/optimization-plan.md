# 优化执行计划

**写于 2026-08-12**，在阶段 D 收尾之后。范围：SFT 基线自身的可优化点，加上此前提出的
三条建议（补 GRPO seed、按 epoch 存 checkpoint、把 grpo_val 接进循环）的核对与修正。

阅读顺序：先看 [结论摘要](#结论摘要)，再看 [三条已有建议的核对](#三条已有建议的核对)——
其中两条的做法要改。SFT 自身的问题在 [SFT 基线的实测问题](#sft-基线的实测问题)。

本文档所有数字都标了出处。凡是**估算**而非实测的，都显式写了「估」字；凡是**没验证过**
的猜测，都写在 [不做的事](#不做的事) 里并给出核算过程，而不是留在正文里当成待办。

## 结论摘要

| # | 项 | 类别 | 成本 | 依赖 |
|---|---|---|---|---|
| **S1** | `evaluate()` 的分片截断丢掉最长的 val 样本 | 正确性 | 改代码，0 GPU | — |
| **S2** | val 曲线没进 `train_log.jsonl`，`metadata.final` 被 val 记录顶掉 | 可观测性 | 改代码，0 GPU | — |
| **S3** | `steps_per_epoch` 用 ceil 而循环用 floor，某些卡数下多跑一个残epoch | 正确性 | 改代码，0 GPU | — |
| **E1** | 用 grpo_val 当 dev set，**离线**评已存的 8 个快照 | 协议 + 实验 | ~3.5 h，1 卡 | — |
| **D1** | 重采 1,219 个零 accepted 任务，SFT 数据估计 +21~31% | 数据 | ~1.6 h，**0 GPU** | — |
| **S4** | 重训 SFT：3-epoch 按 epoch 存 + **独立**的 2-epoch run | 实验 | ~2.4 h，7 卡 | S1 S3 |
| **E2** | epoch 数与 checkpoint 一律在 grpo_val 上选，不用 val loss、不用 500 题池 | 协议 | 0 | E1 |
| **R1** | 补 2 个 GRPO seed + 一份**新的**预注册 | 实验 | ~14.5 h，7 卡 | 新预注册 |
| **S5** | 训练卡数 6 → 7（上次被占，现在空着） | 吞吐 | 免费，估省 15% | S1 S3 |
| **D2** | 若采用 D1 的数据 → SFT 基线变了 → GRPO 必须重跑 | 决策点 | +7.3 h | D1 S4 |

三条已有建议里，**「重训 SFT 存 checkpoint」和「把 grpo_val 接进训练循环」的做法都要改**，
理由分别是 lr 退火混淆（[见 S4](#s4重训-sft两个-run-而不是一个)）和「快照已经在盘上了」
（[见 E1](#e1-用-grpo_val-当-dev-set先离线评已存的-8-个快照)）。补 seed 那条成本核算要修正，
而且需要先补一份预注册。

一句话的优先级判断：**S1/S2/S3 + E1 应该先做**（改代码 + 3.5 h 单卡），因为它们是后面
所有比较的前提；D1 可以立刻并行开跑（不占 GPU、且不承诺任何后续动作）；R1 最贵但收益
最明确，放最后。

## 三条已有建议的核对

### 建议 1：补 2 个 GRPO seed

**判断：该做，理由成立。** 现在的 +7.41 pp [+5.40, +9.51] 是**题内采样噪声**的区间——
`report_metrics.py` 对 498 道题做配对 bootstrap，量的是「同一个训练好的策略，重采一遍
会差多少」。它不含**训练 run 之间**的方差，而后者在本项目里有一个具体且不小的来源：

`iteration_tasks`（`scripts/train_grpo.py:260`）按 seed 把 3,000 题的池子打散一次，再按
轮次取连续切片。40 轮 × 24 题 = **960 道题，占池子 32%，且不绕回**。换 seed 等于换掉
训练用的那三分之一题目。这不是「同一实验的随机抖动」，是两个见过不同数据的 run——
between-run 方差应当显著大于 within-run 采样方差，而现在报的区间完全没有这一项。

**成本要修正。** 一次 GRPO 是 6.56 h（`outputs/logs/grpo_full.log`，40 轮），占 7 张卡
（GPU 1-7）。两个 seed 是 **13.1 h wall-clock**，按卡时算是 ~92 GPU·h，不是 13 GPU·h。
外加每个 seed 一次 500 题 k=4 评测：2,000 回合 ÷ 0.57 ep/s ≈ 38 min，加 vLLM 起服务 3 min
（实测见 `outputs/logs/grpo_eval.nohup`），两个 seed 共 **~1.4 h**。合计 **~14.5 h**。

**必须先补一份预注册，不能改旧的那份。** `docs/eval-preregistration.md` 末尾写着「本文件
此后不再改动」，这条要守住。多 seed 是一个**新的**假设检验，判定规则要在看到第二、第三个
seed 的数字之前写下来，至少包含：

- 合并方式：报 3 个 seed 各自 vs 同一个 SFT 的配对差值，主统计量是**三者的均值**，
  区间用 n=3 的 t 分布（t₂,₀.₉₇₅ = 4.30）。
- 提前算清楚这个区间会有多宽，免得事后觉得「不够显著」就想加 seed：若三个 delta 的
  标准差是 1.0 pp，区间半宽 = 4.30 × 1.0 / √3 = **±2.5 pp**；标准差 2.0 pp 则 ±5.0 pp。
  也就是说 **3 个 seed 只能给一个粗界，给不了紧区间**——这是 n=3 的固有代价，写进
  预注册就不会在事后被当成意外。
- 不许丢 seed：崩掉或没跑满 40 轮的 run 按实际轮数如实报，不静默替换成一个新 seed。
- seed 取 43 / 44（42 已用）。注意 vLLM 采样本身没有 seed，所以即使同 seed 也不会
  逐比特复现——这里的「seed」实际含义是「一次独立的训练 run」，而这正是要测的东西。

### 建议 2：重训 SFT 并按 epoch 存 checkpoint

**判断：该做，但做法要改，而且 `--save-each-epoch` 这个开关已经存在了。**

`scripts/train_sft.py:73` 已有 `--save-each-epoch`，上次只是没加这个参数。所以「想对比
2 vs 3 epoch 只能重跑」的原因不是缺功能，是缺一次带参数的运行。

真正的问题是：**从 3-epoch run 里切出来的 epoch-2 checkpoint 不能代表「2-epoch 训练」**。
余弦调度是按 `total_steps=81` 铺的，走到第 54 步（第 2 个 epoch 结束）时
`lr = 2.616e-06`（`outputs/models/sft/train_log.jsonl` step 54），根本没退火完。一个真正
的 2-epoch run 会把余弦在 54 步内退到 0，末段的低 lr 精修完全没发生过。拿前者当后者比，
比的是「退火完成」vs「退火中途」，不是「2 epoch vs 3 epoch」。

所以要跑**两个 run**，不是一个：详见 [S4](#s4重训-sft两个-run-而不是一个)。

另外，**判定不能用 val loss**：第 3 个 epoch 只降 0.0032 这个数，和成功率之间没有可用的
换算关系。我们真正在意的指标是成功率，而 SFT → GRPO 那一段的经验正说明 loss 与能力
可以脱钩（roadmap 里 scheduler 那个 bug 就是「lr 全错而 loss 照样在降」）。判定要在
grpo_val 上按成功率做，见 [E2](#e2-选择一律在-grpo_val-上做)。

### 建议 3：把 grpo_val 接进训练循环

**判断：目标对，但顺序要倒过来——先别改训练循环。**

核对结果：`grpo_val` 确实从未被用过，全仓引用只有 `build_task_pools.py:45`（生成它）和
roadmap 里提了一句。200 题、与其余四个池 task_id 零重叠（本次重新核验，见
[附录](#附录实测数据来源)）。

但「iter039 是最后一个而不是最好的一个」这个问题**不需要重训就能回答**：8 个快照
（iter004/009/014/019/024/029/034/039）都完整躺在 `outputs/models/grpo/`，每个 15 G，
HF 格式齐全（`model.safetensors` + config + tokenizer），vLLM 可直接 serve。磁盘还剩 705 G。

先离线评这 8 个，成本 ~3.5 h 单卡；改训练循环则要重跑一次 6.56 h 的训练才能验证改动
有没有用。**先做便宜的那个**，见 [E1](#e1-用-grpo_val-当-dev-set先离线评已存的-8-个快照)。

in-loop 的 val 仍然值得加，但它的价值是**给下一次 run 提供早停依据**，不是回答本次的
选择问题。而且要先知道 in-loop val 在 k=1 下够不够用——不够的话加进去只是徒增 wall-clock：
200 题 k=1 的半宽约 ±6.5 pp（按预注册那套公式外推），而 iter019 → iter039 的真实差值
是 +2.63 pp。**k=1 分辨不出来。** 所以 in-loop 版本的正确配置至少是 k=2（±4.6 pp，
每次 ~12 min）甚至 k=4（±3.2 pp，每次 ~23 min），且只在 `snapshot_every` 命中的轮次做
（否则选出来的轮次没有权重可用）。8 次 × k=4 = +3.1 h，给 6.56 h 的 run 加 47%。

这笔账要不要付，取决于 E1 的结果：若 E1 发现成功率在 iter024 附近就见顶，那 in-loop
早停能省下后半程的 3 h 以上，值得付；若 E1 发现确实是单调爬到 iter039，那 in-loop val
只是保险，配 k=2、每 5 轮一次即可。**E1 是这个决定的输入。**

实现位置（等真要做时）：`scripts/train_grpo.py:573-575`，`server.start(policy_dir)` 之后。
那一刻 vLLM 加载的正是刚写盘的 post-update 权重，也正是 `snapshot_every` 命中时存成
`iter{iteration}` 的那份——val 与快照名天然对齐，不存在 off-by-one。环境池直接复用
`env_pool`，不必另起端口。

## SFT 基线的实测问题

### S1：`evaluate()` 的分片截断丢掉最长的 val 样本

`scripts/train_sft.py:292-296` 的 val 走的是同一个 `shard()`，而 `shard()`
（`scripts/train_sft.py:144`）为了对齐 FSDP 集合通信**把各 rank 截到等长**，丢掉尾部
`len(batches) % world_size` 批。训练侧丢一点无所谓（每 epoch 重新 shuffle，长期看都被
采到），**但 val 侧不是**：`token_budget_batches` 先按长度排序，val 又不 shuffle，
所以被丢的永远是排序后最靠后的那几批——**最长的样本**，每个 epoch 丢的都是同一批。

实测（`scripts/analyze_sft_batching.py`，298 条 val 渲染成 111 批）：

| world_size | 每 rank 批数 | 丢批 | 丢样本 | 占比 | 被丢样本长度 | 保留的最长 |
|---|---|---|---|---|---|---|
| 6（**上次实际用的**） | 18 | 3 | 3 | 1.01% | 28,649 – 29,238 | 24,891 |
| 7 | 15 | 6 | 6 | 2.01% | 26,656 – 29,238 | 21,919 |
| 8 | 13 | 7 | 7 | 2.35% | 25,963 – 29,238 | 20,179 |

后果有两条，第二条更麻烦：

1. **偏**：已公布的 `val loss 0.39249` 是 295 条上算的，缺的 3 条恰是最长的回合——
   而长回合是这个任务的主体（p90 15,863，max 32,036）。
2. **不可比**：丢哪些样本取决于 `world_size`。6 卡的 val loss 和 7 卡的 val loss 是在
   **两个不同的验证集**上算的。上次是 6 卡（`metadata.json` 的 `world_size: 6`），而
   `train_sft.sh:18` 默认 7 卡——照默认重跑一次，val 曲线就和已公布的那条不同口径，
   而没有任何东西会提示这件事。

在「第 3 个 epoch 只降 0.0032」这种量级的比较上，验证集悄悄换掉 1-2% 的最长样本是不能
接受的。

**修法**（保持集合通信对齐，且每条 val 样本恰好计一次）：给每个 rank 发
`batches[rank::world]`，轮数取全局一致的 `ceil(len(batches)/world)`，轮数不够的 rank
拿自己的第一批**再跑一次但权重 0**——只为凑齐 all-reduce 的对端，不进分子分母。

```python
all_batches = token_budget_batches(val_examples, args.tokens_per_batch)
if not all_batches:
    return None
mine = all_batches[accelerator.process_index :: accelerator.num_processes]
# 轮数由全局批数决定，各 rank 必然一致 → forward 次数一致 → FSDP 不会卡死
rounds = math.ceil(len(all_batches) / accelerator.num_processes)
total = torch.zeros(2, device=accelerator.device)
for i in range(rounds):
    indices, weight = (mine[i], 1.0) if i < len(mine) else (mine[0], 0.0)
    loss, tokens = forward_loss(indices, val_examples)
    total += torch.tensor(
        [loss.item() * weight, tokens * weight], device=accelerator.device
    )
total = accelerator.reduce(total, reduction="sum")
```

`mine[0]` 要求 `len(mine) >= 1`，在 111 批 / ≤8 卡下恒成立，但仍应加断言而不是靠巧合。

**验收**：同一份权重、同一份 val，在 6 卡与 7 卡下算出的 val loss 必须一致（浮点误差内），
且 `RenderStats.kept`（298）与实际参与计算的样本数相等。补一条测试钉住：构造 111 批
假样本，`world_size` 取 6/7/8，断言三者的 (loss 和, token 和) 完全相同。

### S2：val 曲线没落盘，`metadata.final` 被 val 记录顶掉

`grep val_loss outputs/models/sft/train_log.jsonl` 是**空的**。原因在
`scripts/train_sft.py:372-375`：val 记录只 `history.append()`，而写盘发生在
`if step % args.log_every` 那个块里（`:363-365`），val 走不到。三个 val 数只存在于
`outputs/logs/sft_full.log` 的人读日志里（第 49/77/105 行）。

连带第二个问题：`metadata.json` 的 `final` 取 `history[-1]`，而最后一条恰好是 val 记录，
于是 `final` 变成 `{"epoch":3,"step":81,"val_loss":0.39249}`——**最后一步的 train loss、
lr、grad_norm 全丢了**。

要做 2 vs 3 epoch 的对比、要画 loss 曲线，第一步却是 grep 人读日志，这不对。

**修法**：val 记录也写进 `log_path`；`metadata` 里把 `final` 拆成 `final_train`
（最后一条含 loss 的记录）与 `val_curve`（全部 val 点的列表）。顺带在训练开始前跑一次
`evaluate()` 作为 epoch 0 基线——成本约 1-2 min，换来一条完整曲线的起点。

### S3：`steps_per_epoch` 用 ceil，而循环实际走 floor

`scripts/train_sft.py:218`：

```python
steps_per_epoch = max(1, math.ceil(per_rank / args.grad_accum))
```

而 `:318` 的循环是 `range(0, len(rank_batches) - args.grad_accum + 1, args.grad_accum)`，
实际步数 = `per_rank // grad_accum`（**floor**）。两者在 `per_rank % grad_accum != 0`
时不等，后果是 `total_steps` 比实际每 epoch 能走的步数总和大，`while` 循环于是**多跑一个
残 epoch** 去补齐，而余弦调度在残 epoch 中途才走到 0。

上次没出问题纯属巧合：world_size=6 → per_rank=108，108 % 4 == 0，ceil 与 floor 相等
（都是 27，×3 = 81 步，与日志一致）。7 卡也恰好整除（per_rank=92 → 23）。**8 卡会踩**：
per_rank=81 → ceil 21 vs floor 20 → `total_steps` = 63 而 3 个 epoch 只有 60 步 → 第 4 个
epoch 跑 3 步收尾，`--save-each-epoch` 还会多写一个 `epoch4/`。以上 per_rank 均来自
`scripts/analyze_sft_batching.py` 的实测分批（648 批）。

**修法**：让 `steps_per_epoch` 用与循环相同的口径，`total_steps` 就精确落在最后一个 epoch
的最后一步，余弦保证退到 0：

```python
steps_per_epoch = max(1, per_rank // args.grad_accum)
```

**验收**：`--epochs 3` 下，日志最后一条的 `epoch` 恰为 3.0 且 `lr` 为 0.0；val 记录恰好
3 条（加上 epoch 0 基线是 4 条）。6 卡下这个改动不改变任何数字（仍是 81 步），可以用
已有的 run 做回归对照。

### D1：1,219 个零 accepted 任务可以重采

这是**唯一一项能真正提升 SFT 基线本身**的改动，其余都是测量与协议。

`data/sft/verdicts.jsonl`：3,000 个任务、3,000 条轨迹（1 任务 1 轨迹），1,781 条 accepted，
所以**有 1,219 个任务一条可用数据都没有**。按被拒时的 `reward_type` 分类：

| reward_type | 条数 | 换一次采样有希望吗 |
|---|---|---|
| `repeat_loop` | 478 | **有**——绕圈是采样级的偶然，不是任务不可解 |
| `partial_alternative_purchase` | 178 | 一般——教师判错了商品，换样有一定概率改对 |
| `max_steps` | 144 | **有**——步数用完，与绕圈同源 |
| `no_terminal:no_tool_call` | 141 | **有**——格式/服务端偶发 |
| `no_terminal:empty_response` | 92 | **有**——服务端空响应，纯偶发 |
| `wrong_purchase` | 88 | 一般 |
| **`gold_purchase`** | **74** | **已经做对了**，见下 |
| `reward_unverifiable` | 18 | 有 |
| `early_abstain` | 5 | 一般 |
| `no_terminal:rejection_limit` | 1 | 有 |

那 74 条 `gold_purchase` 值得单独看：**任务已经解对了**，被拒是别的原因。按
`rejection_count` 分布 `{0:9, 1:18, 2:13, 3:3, 4:24, 5:6, 7:1}`——其中 31 条是被拒动作
超过上限 3，另 **43 条 `rejection_count ≤ 3`，即完全是 `dropped_tool_calls` / `has_error`
这类格式瑕疵挡下来的**。教师把题做对了，只是转录里有个格式疤。这 43 条重采成功率应当
最高。

**估算收益**：不能直接用 0.5937 的接受率——这 1,219 个任务本身偏难（它们已经失败过
一次）。取「有希望」那几档（478+144+141+92+18+1 = 874 条偶发失败）成功率 40%，
其余 345 条成功率 15%，得 **+400 条**；乐观/悲观区间取 **+370 ~ +550 条**，即
**+21% ~ +31%** 训练数据。**这是估算，不是实测**，真值只有跑完才知道。

**关键性质：不会引入任务重复。** 只重采零 accepted 的任务，所以每个任务最多仍只贡献
一条轨迹。roadmap 阶段 B 末尾担心的「同任务多条轨迹放大该任务权重」在这里不成立。
而且 `select()`（`src/ecom_agent_rl/data/sft.py:274`）本身就按 task_id 去重、保留第一条
accepted，所以合并两份轨迹文件在机制上也是安全的——去重是安全网，不是主要依赖。

**成本：~1.6 h wall-clock，0 GPU。** 教师是外部 API（`collect_teacher.sh`），实测
0.21 ep/s @ 并发 8（`outputs/teacher/sft_train.summary.json`），1,219 ÷ 0.21 ≈ 5,800 s。
API token 开销按实测 ~144k prompt tokens/回合估算约 **176 M prompt tokens**——这是钱，
不是卡时，要先确认预算。

两条操作注意：

- 上次采集被一次 `HTTP 400 Content Exists Risk` 中止过（`llm_error` 属于
  `INFRA_FAILURES`，会中止整批）。用 `outputs/logs/baseline_until_done.sh` 那个
  until-done 循环包一层，别守着终端。
- 并发别照抄「加大就快」：环境是多进程，`--env-workers N --concurrency N` 是实测最优
  工作点（`docs/environment-notes.md`）。若同时在跑 E1，两边**必须用不同端口段**
  （如 E1 用 5700、D1 用 5800），且 worker 总数控制在 32 以内。

**这一步不承诺任何后续动作。** 轨迹落在新文件 `outputs/teacher/sft_train_retry.jsonl`，
不动原文件；要不要拿它建 SFT-v2 是 [D2](#d2-决策点采用新数据就得重跑-grpo) 的决定。
所以可以立刻开跑。

### S5：训练用 7 卡而不是 6 卡

上次实际跑的是 6 卡（GPU 2-7），因为 GPU 0/1 被教师采集的 vLLM 占着
（roadmap 阶段 D）。现在 GPU 1-7 全空，GPU 0 有别人 8 G 常驻。`train_sft.sh:18` 默认
就是 1-7 = 7 卡。

7 卡下每 rank 92 批（实测）→ 23 步/epoch → 69 步。按 72.3 s/step 的中位步时估算
**~1 h 23 min**，比 6 卡的 1 h 41 min 省 ~18%。步时不会完全线性（每步的 token 数不变，
只是批数少了），所以这是估算。

**但 S1 和 S3 必须先落地**：换卡数会换掉 val 集（S1 的表：6 卡丢 3 条、7 卡丢 6 条），
8 卡还会踩 S3 的残 epoch。顺序错了就得到一组和历史不可比的数。

## 执行顺序

三批。批内可并行，批间有依赖。

```
批 1（0 GPU 训练，立刻开）
  ├── S1 S2 S3  改代码 + 补测试          ~1-2 h 人工
  ├── E1  grpo_val × 9 个模型            ~3.5 h，1 卡（端口 5700）
  └── D1  教师重采 1,219 题              ~1.6 h，0 卡（端口 5800）
                    ↓
批 2（依赖 S1 S3 落地）
  ├── S4a 3-epoch run + --save-each-epoch  ~1.4 h，7 卡
  ├── S4b 独立 2-epoch run                 ~0.9 h，7 卡（串行，同一批卡）
  └── E2  两者在 grpo_val 上 k=4 各评一次  ~0.8 h，1 卡
                    ↓
批 3（依赖新预注册；D2 是岔路口）
  ├── R1  seed 43 / 44 各一次 GRPO + 评测  ~14.5 h，7 卡
  └── D2  若采用 D1 的数据 → GRPO 重跑     ~7.3 h，7 卡
```

批 1 合计 ~3.5 h（E1 与 D1 并行，卡上只占 1 张）。批 2 ~3.1 h。批 3 ~14.5 h，D2 视决定
再加 7.3 h。**总计 21 ~ 29 h wall-clock。**

### E1 用 grpo_val 当 dev set，先离线评已存的 8 个快照

评 9 个模型：`outputs/models/sft` + 8 个 `iter*`。每个 200 题 × k=4 = 800 回合。

口径与 `eval_grpo.sh` 逐字一致（temperature 0.7 / top_p 0.95 / max_tokens 1024 /
max_steps 35 / 窗口 24576），**只换池子**。并发可以调（不影响采样语义，只影响速度），
但要遵守 `--env-workers N --concurrency N`。

按实测 0.57 ep/s：7,200 回合 ÷ 0.57 ≈ **3.5 h**，加 9 次 vLLM 起服务（各 ~3 min）≈ 3.9 h。
把并发与 env-workers 提到 32 有望缩到 ~2 h，但这是估算，先按 3.9 h 排期。

**rollout 必须串行**（`eval_grpo.sh` 开头那段：`EnvironmentPool` 的
`BoundedSemaphore` 是每个客户端池各一份，同端口段并行跑两个池会各自以为独占租约 →
`env_error` → 整批中止）。vLLM 可以并行起在不同 GPU 上，省掉中间的加载等待。

**分辨率要先算清楚，否则会把噪声当信号。** 按预注册那套公式外推到 n=200、k=4，配对
半宽约 **±3.2 pp**；两个同血缘检查点相关性更高，实测 n=498 时是 ±1.76 pp，按 √(498/200)
缩放约 **±2.8 pp**。而 iter019 → iter039 的真实差值是 +2.63 pp。结论：

- grpo_val k=4 **能**回答「策略是不是在 iter039 之前就见顶了、且差距大到值得换权重」。
- **不能**给相邻快照排名。若前三名挤在 3 pp 内，就当它们并列，对并列的 2-3 个再补
  k=8（每个 +23 min）——**这条升级规则要现在写下，不是看到结果再定**。

**选中之后怎么报**：500 题池是 test set，只出最终数字。若 grpo_val 选出的不是 iter039，
则在 500 题池上补评它，并**同时报**预注册的 iter039 数字与新选中的数字，明确写出
「后者经过 dev set 选择」。挑好看的那个报，等于把 test set 变成 dev set。

### E2 选择一律在 grpo_val 上做

这是一条协议，不是一次实验，成本为 0，但它决定了前面几项的结论算不算数。

| 池子 | 条数 | 用途 | 谁能碰 |
|---|---|---|---|
| `data/sft_val` | 299 行 | 只算 val loss（诊断，不做选择） | SFT 训练 |
| `grpo_val` | 200 题 | **dev set**：epoch 数、checkpoint、超参，一切选择 | 任何选择动作 |
| `evaluation` | 500 题 | **test set**：只出对外报的数字 | 只在结论确定后 |

五个池两两 task_id 零重叠（本次重新核验，[附录](#附录实测数据来源)），所以 grpo_val 用来
选 SFT 的 epoch 数也是干净的——它与 `sft_train` 不相交。

为什么必须这样：2 vs 3 epoch 和「哪个 iter 最好」都是**模型选择**。在 500 题池上做选择，
再在同一个池上报差值，就是在 test set 上选模型——`docs/eval-preregistration.md` 花了整节
防的「不显著就加采样」是同一类错误的另一个面。grpo_val 一直闲置，正好补上这个位置。

### S4：重训 SFT，两个 run 而不是一个

```bash
# a) 3 epoch，按 epoch 存（epoch1/ epoch2/ 另存，最终权重仍在 out-dir 根）
bash scripts/train_sft.sh \
  --validation data/sft_val/train.jsonl \
  --out-dir outputs/models/sft_e3 \
  --epochs 3 --save-each-epoch

# b) 独立的 2 epoch——余弦在 2 个 epoch 内退到 0，这才是「2-epoch 训练」
bash scripts/train_sft.sh \
  --validation data/sft_val/train.jsonl \
  --out-dir outputs/models/sft_e2 \
  --epochs 2
```

两个 run 串行（同一批卡）。7 卡估 1.4 h + 0.9 h。磁盘：每份权重 15 G，
`sft_e3` 含 epoch1/epoch2/最终 = 45 G，`sft_e2` 15 G，合计 60 G，剩 705 G 够。

比较矩阵（全部在 grpo_val 上，k=4，各 800 回合）：

| 权重 | 意义 |
|---|---|
| `outputs/models/sft`（已有） | 6 卡旧 run，作为 world_size 变化的对照 |
| `sft_e3`（最终） | 7 卡 3-epoch，检查换卡数有没有改变结论 |
| `sft_e2` | **2-epoch 的公平版本**（余弦退完） |
| `sft_e3/epoch2` | 3-epoch 途中的 2-epoch，lr 未退火——用来量「退火值多少」 |

最后一行是白送的：`sft_e3/epoch2` 与 `sft_e2` 的差值就是**退火本身的贡献**，两者数据、
seed、步数全同，只差 lr 轨迹。这个数以前没人量过，而它决定了「从长 run 里切 checkpoint」
这种省钱做法在本项目里到底可不可信。四份权重 × 800 回合 ≈ 1.6 h（其中一份已有权重，
但 grpo_val 上的回合都得新采）。

**验收**：S1 生效后，`sft_e3` 的 val 曲线应有 4 个点（epoch 0/1/2/3）且都在
`train_log.jsonl` 里；`sft_e3` 最终 val loss 与旧 run 的 0.39249 差异应能被
「world_size 6→7 换了 3 条最长样本」解释——若差得远，说明还有别的东西在动。

### R1：补 seed

先写 `docs/multiseed-preregistration.md`（内容见 [建议 1](#建议-1补-2-个-grpo-seed)），
再开跑：

```bash
bash scripts/train_grpo.sh --seed 43 --out-dir outputs/models/grpo_s43
bash scripts/train_grpo.sh --seed 44 --out-dir outputs/models/grpo_s44
```

两次串行，各 6.56 h。然后各自在 500 题池上 k=4 评测，与**同一个** SFT 配对
（`report_metrics.py --baseline`）。

磁盘要留意：每个 run 的 `policy/` + 8 个快照 = 135 G，两个 run = 270 G。剩 705 G 够，
但若同时保留 D2 的产物就紧了。**建议这两个 run 传 `--snapshot-every 0`**（或跑完即删
中间快照）——它们的用途是量 run 间方差，不是做 checkpoint 选择，中间快照留着没用。

### D2 决策点：采用新数据就得重跑 GRPO

D1 跑完会有一份更大的 SFT 数据集。**用它就意味着 SFT 基线变了**，而现在这份
+7.41 pp [+5.40, +9.51] 描述的是「旧 SFT → 从旧 SFT 出发的 GRPO」。换掉 SFT 之后：

- GRPO 的起点变了，那个 +7.41 pp 不再描述新流水线；
- 要拿到 apples-to-apples 的数字，必须从 SFT-v2 重跑一次 GRPO（6.56 h）+ 评测（0.7 h）。

两条路，**明确选一条，不要含糊地两头都占**：

**路 A——冻结 SFT，把已有结论做扎实。** 做 S1/S2/S3 + E1 + E2 + R1。产出：一个带 run 间
方差的 +7.41 pp、一个经过 dev set 选择的最优 checkpoint、一套修好的测量口径。SFT 数据
不动，历史数字全部保持可比。

**路 B——提升 SFT 基线本身。** 在路 A 之上加 D1 → SFT-v2 → GRPO-v2。产出更强的绝对
成功率，代价是旧的对照关系全部作废一轮，且要重讲一遍。

**建议：先完整走路 A，同时把 D1 的数据采下来放着**（1.6 h、0 GPU、不占用任何后续
承诺）。等 E1/E2 的结果出来，再决定要不要开路 B——那时手上会多两个输入：GRPO 的收益
到底还有没有空间（E1 的曲线形状），以及 epoch 数这一维是不是已经调到位（E2）。

## 不做的事

以下四项核算过，**不值得做**，写在这里免得下一轮又被当成待办捡起来。

**padding 浪费——不存在。** `token_budget_batches` 先按长度排序再分桶，实测
padding 浪费 train **0.08%**、val **0.41%**（`scripts/analyze_sft_batching.py`：
train 17,028,306 内容 token / 17,042,097 padded）。这里没有可捡的东西。

**「94% 的 token 不参与 loss」不是可优化项。** 每 epoch 内容 token 17.03 M，可训练
token 948,119（`train_log.jsonl` 前 27 步之和），占 **5.57%**。看着像巨大的浪费，但
那 94.4% 是 observation，**必须前向**才能给 assistant token 提供上下文，省不掉。

真正可省的只有 lm_head 与 CE 那一段——只对可训练位置算。transformers 5.15 的
`Qwen2ForCausalLM.forward` 恰好支持：`logits_to_keep` 非 int 时直接当索引用
（`logits = self.lm_head(hidden_states[:, slice_indices, :])`），传张量即可。但核算下来
不值得：lm_head 前向 32,768 × 3,584 × 152,064 × 2 ≈ 35 TFLOP，含反向约 105 TFLOP；
模型主体含梯度检查点的重算约 1,850 TFLOP。**head + CE 只占约 3-5% 的步时**，而改动要
么绕过 FSDP2 根模块的 unshard 钩子（直接调 `model.model` 与 `model.lm_head`），要么处理
`hidden_states[:, idx, :]` 在 batch 内各行可训练位置不同的问题。**风险与收益不成比例。**

顺带：现在那段分块 fp32 上采（`--loss-chunk-tokens 4096`）是必要的且已经解决了显存问题，
按上面的核算它的**时间**开销本来就不大，不必再优化。

**flash-attention-2——没量过，不要凭直觉装。** `--attn` 默认 `sdpa` 因为本机没装
flash-attn。torch 的 sdpa 在因果掩码下本来就会派发到 flash 后端，而我们传的是带 padding
的显式 `attention_mask`，可能落到 memory-efficient 后端。收益区间从「几乎为 0」到
「20%」都有可能，取决于派发到哪条路径。**要么先用 `--limit 32 --max-train-steps 3` 量
一次 s/step 再决定，要么别碰。** 装它需要拖编译依赖，在这台 driver 570 + cu130 的机器上
是额外的风险面（见 `docs/environment-notes.md` 的 CUDA 一节）。

**选择性梯度检查点——同理，先量。** 现在无条件 `gradient_checkpointing_enable()`，等于
多付一次前向（约 +33%）。7 卡 FSDP2 下每卡权重/梯度/Adam 状态估约 19 G，剩 ~60 G 给激活，
隔层检查点也许放得下。但这是估算，OOM 的代价是重跑，而收益上限只有那 33% 里的一部分。
排在 flash-attn 之后。

## 附录：实测数据来源

本次为写这份文档新跑的核验：

- `scripts/analyze_sft_batching.py`（新增，只用 tokenizer，不碰 GPU）——分批、padding
  浪费、各 world_size 下的分片截断。结果落在
  `outputs/logs/sft_batching_analysis.json`。
- `data/sft/verdicts.jsonl` 的分组统计——1,219 个零 accepted 任务及其
  `reward_type` / `rejection_count` 分布。
- 五个任务池两两求交——全部为 0，与 roadmap 阶段 B 的记录一致。
- `outputs/models/sft/train_log.jsonl`——step 54 的 `lr = 2.616e-06`（S4 的退火混淆
  证据）、可训练 token 之和、72.3 s 的中位步时。
- `grep val_loss outputs/models/sft/train_log.jsonl` 为空——S2。
- `outputs/models/grpo/iter*/` 全部 15 G 且文件齐全——E1 的前提。
- `nvidia-smi`：GPU 1-7 空闲，GPU 0 有 8 G 常驻；`/data` 剩 705 G。

引用的既有实测：教师 0.21 ep/s（`outputs/teacher/sft_train.summary.json`）、评测
0.57 ep/s（`outputs/logs/sft_eval_k4.log`）、GRPO 6.56 h（`outputs/logs/grpo_full.log`）、
评测 wall-clock（`outputs/logs/grpo_eval.nohup`）、k=4 分辨率
（`docs/eval-preregistration.md`）。
