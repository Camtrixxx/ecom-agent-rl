# Ecom Agent RL

面向电商长程 Agent 的后训练与评测项目。Qwen2.5-7B-Instruct、万级数据、8 卡训练，
主线是 **Baseline → SFT → GRPO → Evaluation**。

任务形态：模型在一个 23,421 条商品的模拟商城里通过工具调用搜索、翻页、查看详情、
选规格、下单，最多 35 步，目标是买到满足 instruction 里全部约束的那件商品。

## 结果

留出集 500 题 × 4 次采样 = 2,000 条回合，三个模型同池、同温度（0.7）、同解码参数。
`generation_config.json`、chat template、tokenizer 逐字节一致，所以差异只来自权重。

| 指标 | baseline | SFT | GRPO |
|---|---|---|---|
| **成功率** | 0.0745 [0.0610, 0.0885] | 0.6061 [0.5718, 0.6397] | **0.6838 [0.6475, 0.7182]** |
| gold 率 | 0.0735 | 0.6046 | 0.6818 |
| 错买率 | 0.0060 | 0.0544 | 0.0444 |
| 平均 reward | −0.0276 | 0.4282 | 0.5652 |
| 平均步数 | 4.67 | 14.38 | 8.65 |
| 平均被拒动作 | 1.63 | 0.097 | 0.065 |

按 `task_id` 配对的 bootstrap 差值（配对消掉题目难度方差，比非配对区间更紧）：

| 比较 | 差值 | 95% 区间 |
|---|---|---|
| SFT vs baseline | **+53.13 pp** | [+49.67, +56.17] |
| GRPO vs SFT | **+7.41 pp** | [+5.40, +9.51] |
| GRPO vs baseline | +60.86 pp | [+57.22, +64.24] |

三个区间都不跨 0。GRPO 的 +7.41 pp 是这套流程噪声下限（k=4 时 ±2.0 pp，见下）的
3.7 倍。

### 增益分别来自哪里

成功率可以乘性分解成「走到终局的比例 × 走到终局后买对的比例」，这样才能把「学会
输出合法动作」和「决策变好」分开——否则前者会被误读成后者：

| 因子 | baseline | SFT | GRPO |
|---|---|---|---|
| 走到终局率 | 0.3080 | 0.9950 | 0.9975 |
| 条件买对率 | 0.2419 | 0.6065 | **0.6777** |

**SFT 的增益两项都占**（格式项 +11~17 pp、决策项 +36~42 pp，取决于归因顺序）。
**GRPO 的增益 100% 落在决策项**：走到终局率只动了 0.2 pp，因为 SFT 已经把它推到
0.9950 的天花板，格式那条路已经没有红利可吃。

### 行为怎么变的

| 终局类型 | baseline | SFT | GRPO |
|---|---|---|---|
| `gold_purchase` 买到目标商品 | 7.3% | 60.2% | **67.4%** |
| `partial_alternative` 买到部分匹配 | 5.4% | 13.2% | 14.7% |
| `repeat_loop` 绕圈（−0.65） | 4.6% | 16.5% | 11.0% |
| `wrong_purchase` 买错（−0.85） | 0.6% | 5.4% | 4.2% |
| `max_steps` 耗尽步数 | 0.2% | 2.9% | 0.4% |
| `early_abstain` 不敢买 | 11.3% | 0% | 0% |
| **没能发出合法动作** | **69.2%** | 0.5% | 0.3% |

三段各自的瓶颈完全不同，读这张表比读成功率有用：

**baseline 的瓶颈不是选品，是把动作发出去。** 69.2% 的回合压根没走到购买——其中
21.6% 是把动作写成自然语言（「动作: open_product("...")」），换任何 parser 都救不
回来；7.1% 是写了 `<tool_call>` 块但混进垃圾导致整块解析失败。另有 11.3% 是不敢买。

**SFT 换来「敢买但会买错」。** `early_abstain` 11.3% → 0，代价是错买率 0.6% → 5.4%。
这是笔划算的交易——错买率此时才第一次成为有意义的指标。同时 `repeat_loop` 涨到
16.5%，成为最大失败类型（但它占**走到终局的回合**的比例只从 14.9% 动到 16.5%，
绝对数涨是因为走到终局的回合本身多了 3.2 倍，读成「SFT 让模型更爱绕圈」是错的）。

**GRPO 同时压低绕圈、买错和耗尽步数**，平均步数从 14.4 掉到 8.7（吞吐因此从 0.57
涨到 0.87 ep/s）。但它推动的**不是**约束核验能力：错买率随约束数单调上升的形状没有
被改变，8+ 约束桶从 0.100 只动到 0.103。诊断见 [roadmap](docs/roadmap.md#两个半程在做相反的事)——
该桶 `repeat_loop` 占比只有 0.043、是四桶最低，说明约束一多，模型既没更会核验，也
没退回「没把握就继续找」，而是自信地买错。

## 评测口径

**每题多次采样 + 置信区间，不是每题一次。** 这条不是形式主义，代价被量化过：拿 SFT
自己的 2,000 条按 `attempt` 分半做配对比较（两半是同一个模型，**真实差值为 0**），
量出来的区间就是这套流程的噪声下限。

| k | 半宽（真差值 = 0） | 每模型回合数 |
|---|---|---|
| 1 | ±4.1 pp | 500 |
| 2 | ±2.9 pp（实测） | 1,000 |
| **4** | **±2.0 pp** | **2,000** |
| 8 | ±1.5 pp | 4,000 |
| 16 | ±1.0 pp | 8,000 |

即**每题只采样一次的评测分辨不出 4 个点以内的差异**。RL 阶段的典型增益正是这个量级，
所以 k=1 足以让一次成功的训练和一次失败的训练看起来一样。

配套的三个决定：主指标是成功率而非平均 reward（后者会把「买错」−0.85 和「没买」
−0.15 混成一个数）；置信区间用 bootstrap 而非正态近似（成功率低到 0.1 时 Wald 下界
会越界）；每题 k 次先按题内取平均再对题 bootstrap（否则 k×N 条被当独立样本，区间
算窄）。终局标签取环境的权威判定 `reward_detail.reward_type`，不从 messages 猜。

**判定规则写在看到数字之前**，落盘于 [docs/eval-preregistration.md](docs/eval-preregistration.md)，
含「区间跨 0 就把 k 补到 16」的升级条件和四条可否证的次要预测（其中一条被数据否掉了）。
理由：先看结果再决定采样量，会把假阳性率从 5% 推到 20% 以上，而过程中每一步都显得合理。

**评测隔离做在 rollout 层，不是评测层。** 环境的 `reset` 返回里 `goal_options` 直接给
出目标商品规格，`progress.credited_evidence_added` **每一步**都带
`constraint:<asin>:budget:fail` 这类逐条约束判定——等于把评分器递给模型。两者在**写盘
之前**就被白名单过滤剥离到 `audit`，所以任何读轨迹文件的下游都拿不到答案。白名单遇到
陌生字段直接报错而不是放过：环境升级新增字段时我们需要被吵醒。

## 流水线

前置（每次开新 shell 都要）：

```bash
bash scripts/setup_environment.sh          # 一次性：装依赖 + 校验商品数据 + 建索引
bash scripts/start_environment.sh          # 环境池：8 worker × 4 slot，端口 5700+
bash scripts/serve_model.sh <权重路径>      # 被测模型的 vLLM 服务，默认端口 8180
python scripts/check_environment.py        # 可选：量环境池还剩多少 slot，有泄漏则退出码 1
```

| 阶段 | 目标 | 入口 |
|---|---|---|
| 数据 | 抽三个 `task_id` 零重叠的任务池 | `python scripts/build_task_pools.py` |
| Baseline | 测量原始权重的工具使用能力 | `python scripts/run_rollout.py --pool data/task_pools/evaluation.jsonl --out outputs/rollouts/baseline.jsonl` |
| SFT | 从教师轨迹学习合法、完整的购物行为 | `bash scripts/collect_teacher.sh` → `python scripts/build_sft_dataset.py` → `bash scripts/train_sft.sh` |
| GRPO | 在真实环境 rollout 中优化 Reward v3 | `bash scripts/train_grpo.sh` |
| Evaluation | 在同一批留出任务上公平比较 | `bash scripts/eval_grpo.sh`（三方一键）或 `run_rollout.py --attempts k` → `python scripts/report_metrics.py --trajectories <轨迹> --baseline <对照> --pool <池>` |

`run_rollout.py` 是 baseline 评测、教师采集、smoke 三者共用的入口，区别只在
`--pool` / `--out` / `--base-url`。被中断后重跑同一条命令即按 `(task_id, attempt)`
续跑；基础设施失败（`env_error` 等）写进同名的 `.failures.jsonl`，留证据但不占用
attempt，所以「修好之后重跑即可续跑」是真的。

数据规模：SFT 任务池 3,000 / 500（train 与 val 任务零重叠，接受率 0.5937 → 数据集
1,781 / 299 行），GRPO 3,000 / 200，评测 500。三池 `task_id` 零重叠，sha256 +
provenance 血缘记在各自的 `metadata.json`。

## 环境

[ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator) Environment v2.1 +
Reward v3，位于 `third_party/ShopSimulator/`，不提交进 git。

商品池 23,421 条任务，横跨 9 个品类。每条任务自带 `attributes`（约束列表，均值
4.53 条，是免费且可靠的难度轴，用于分层报告）、`pricing`、`customization_options`
（规格变体及各自价格）。任务与商品一对一，这是「三池 task_id 零重叠」能防住商品级
泄漏的前提。

授权说明：上游 base commit `51bb26012cee31aea7ac26177c5ffe807026ac07` 只提供 WebShop
式基础壳子；Environment v2.1 与 Reward v3 的 engine 层（约 3,100 行，含 `reward.py`、
`variant_price.py`、`comparators.py`、`termination.py` 等）是上游之外的下游实现，
随环境一并引入。

**环境服务单进程被 GIL 锁死在 1 核**（~4-5 episodes/s，并发 1 即饱和），加 slot 不
提升吞吐，扩展只能靠多进程且到 32 进程仍是线性（133 ep/s）。最优工作点是
并发 = worker 数。完整实测见 [docs/environment-notes.md](docs/environment-notes.md)。

## 几个关键的设计选择

**模型选 Qwen2.5-7B-Instruct。** Qwen3.5 全系列是多模态
（`Qwen3_5ForConditionalGeneration`，带 `vision_config`），而本任务是纯文本，vision
tower 只会占显存和参数量。Qwen2.5 纯文本、中文原生，且同系列有 14B/32B 可作后续
scaling 对照，tokenizer 与 chat template 一致，适配层无需改动。工具调用格式用
`hermes`（Qwen2.5 的原生格式），vLLM 的 `--tool-call-parser` 同步。

**上下文压缩是跑满 35 步的前提，不是优化。** 实测搜索结果页均值 2,406 tokens，
35 步需要约 39k，而窗口是 24576 —— **第 18 步就撞 HTTP 400**，前面的步数全部白跑。
压缩时删掉旧组的正文但留一行有界的动作摘要：`repeat_loop` 是 −0.65，专罚重复动作，
一个看不见自己历史的模型必然重复，整组丢弃等于让压缩本身制造它要避免的失败。

**GRPO：μ=1、无 KL、无 reference model。** μ=1 时重要性比恒等于 1、clip 是恒等操作，
所以不实现一个永远不生效的分支；loss 直接写成 `Σ A·CE / Σ 1`，等价于带基线的策略
梯度，还能原样复用 SFT 那段分块 fp32 上采的 CE。优势用 `A_i = r_i − mean(group)`，
不除标准差。KL 系数取 0 省下每卡 2.3 GB 和一整次前向，换成 lr 1e-6 + 梯度裁剪 1.0 +
每轮记回报分布——漂移会先在后者上看见，而不是等 KL 报警。

**权重同步用重启 vLLM，不用 RPC 热更新。** 热更新失败的方式是静默的：服务照常应答，
只是用的还是旧权重，训练曲线上看不出来。重启幂等，代价每轮约 2 分钟。写权重前必须
先停服务——从第 2 轮起被覆盖的正是 vLLM 当前 mmap 着的目录。

## 文档

- [docs/roadmap.md](docs/roadmap.md) — 各阶段的实测结果、踩过的坑与未解决的问题
- [docs/eval-preregistration.md](docs/eval-preregistration.md) — GRPO 评测的判定规则（开跑前落盘，此后不改）
- [docs/environment-notes.md](docs/environment-notes.md) — 环境侧实测：reward 结构、容量、slot 泄漏、CUDA 兼容
