# Roadmap

## 已锁定的决策

| 项 | 决定 | 说明 |
|---|---|---|
| 主线 | Baseline → SFT → GRPO → Evaluation | 沿用参考实现的四阶段，不加新阶段 |
| 模型 | Qwen2.5-7B-Instruct | 纯文本；Qwen3.5 全系列是多模态，vision tower 于本任务无用 |
| 数据 | SFT 3,000/500，GRPO 3,000/200，评测 500 | 比参考实现大一档，后续按需再加 |
| 训练 | 8 卡原生，不 offload | 参考实现的单卡妥协在 80G 卡上纯属速度损失 |
| 环境 | 复制到 `third_party/`，不入 git | 上游没有 v2.1，必须带下游实现 |
| 参考实现 | `/data/heyuhang/shopping-grpo-longhorizon` 只读 | 唯一对照基准，不在其上改动 |

硬件：8 × A800-SXM4-80GB（640G 显存），driver 570.172.08，`/data` 可用 980G。

模型权重：`/data/heyuhang/models/Qwen2.5-7B-Instruct`（7.6G）。

后续 scaling 对照可用同系列 Qwen2.5-14B/32B-Instruct，tokenizer 与 chat
template 一致，适配层无需改动。

## 实施顺序

原则：先有可信评测和足量数据，再训练。否则训练结果无法解读。

### 阶段 A — 环境服务与容量基线

参考实现是单 Flask `app.run()` 进程、`SHOPSIM_ENV_SLOTS` 默认 8。rollout 并发、
评测并发、教师采集并发吃同一个 slot 池，这是全项目的吞吐上限。

- [x] setup 脚本：装环境依赖（独立 py3.10 venv）+ 校验商品数据 SHA-256 + 建索引
- [x] 容量压测：slot 开到 8/32/64/128，测 QPS 与 p99，找单进程饱和点
- [x] 若单进程不够：多进程环境服务 + 负载均衡
- [x] 记录容量 SLO，作为后续所有并发配置的输入

结论与原计划的假设相反：**加 slot 不提升吞吐**。单进程被 GIL 锁死在 1 核、
~4-5 episodes/s，并发 1 即饱和，8 slot 与 32 slot 在吞吐上没有区别。扩展只能靠
多进程，且到 32 进程仍是线性（133 ep/s）。最优工作点是并发 = worker 数。
详见 [environment-notes.md](environment-notes.md)。

所有 env 共享同一个 server 实例，商品库只加载一次，因此加 slot 主要吃会话状态
而非 20M 商品库副本；单进程的限制来自 GIL。

商品数据 SHA-256：`57b10950a0064d16c81535a1d764a75879a508d250dde8a2a1787c5e6045559f`

### 阶段 B — 数据层

- [x] 任务池抽样扩到 SFT 3,000/500 + GRPO 3,000/200 + 评测 500
- [x] 三者 task_id 零重叠，sha256 + provenance 血缘记录
- [x] 按 `attributes` 数量分层记录，用于分层报告
- [ ] 教师轨迹采集：断点续跑 + 并发（并发上限受阶段 A 的 SLO 约束）

切分共 7,200 条（占全池 30.7%），分层轴是 `domain_en_short` × `attributes` 桶。
各池难度分布与全池的偏差在 1 个百分点内，9 个品类全覆盖；grpo_val 只有 200 条，
抽样噪声最大（5-7 桶 39.0% vs 全池 35.9%），分层报告解读时要留意。
`python scripts/build_task_pools.py` 复现，血缘见 `data/task_pools/metadata.json`。

参考实现的接受率是 0.41（2,498 条原始轨迹 → 1,026 条通过），按此反推 3,000 条
accepted 需采集约 7,500 条原始轨迹。

### 阶段 C — 评测

- [x] 并行 rollout（参考实现是 `for task in tasks` 串行）
- [x] 每题 k 次采样 + 置信区间
- [x] 按 task_id 配对比较
- [x] 按 `attributes` 数量与品类分层报告
- [x] 评测隔离：移除 reward、gold、raw observation
- [ ] Baseline 跑通，作为对照下界

隔离做在 rollout 层而不是评测层：`observation.split_env_payload` 用白名单过滤，答案
字段在**写盘之前**就被剥离到 `audit`，所以任何读轨迹文件的下游都拿不到 gold。参考
实现把 `goal`/`reward_detail` 原样存进 `terminal_result`，只在喂 Judge 时才过滤。

实测发现两处参考实现没提的泄漏：`goal_options` 在 reset 就给出目标商品规格；
`progress.credited_evidence_added` **每一步**都带 `constraint:<asin>:budget:fail`
这类逐条约束判定，等于把评分器交给模型。两者都已拦掉并有测试钉住。

指标层的三个决定（`evaluation/metrics.py`）：主指标是成功率而非平均 reward——后者会把
「买错了」（-0.85）和「没买」（-0.15）混成一个数，`wrong_purchase` 率单独报；置信区间
用 bootstrap 而非正态近似，成功率低到 0.1 时 Wald 下界会越界；每题 k 次采样先按题内
取平均再对题 bootstrap，否则 k×N 条被当独立样本、区间算窄。终局标签取
`audit.terminal.reward_detail.reward_type`（环境的权威判定），不从 messages 猜。
配对比较实测比非配对区间紧（0.0530 vs 0.0580），因为消掉了题目难度的方差。
`python scripts/report_metrics.py --trajectories <轨迹> --pool <池> --baseline <对照>`。

拿真轨迹验指标层时发现一个自己的 bug：`run_batch` 把基础设施失败（env_error 等）也写进
主轨迹文件，于是续跑按 `(task_id, attempt)` 去重时永远跳过它们——「修好之后重跑即可续跑」
是空话——而指标层会把 `no_terminal:env_error` 当成模型的失败算进分布。现在这类失败改写
同名的 `.failures.jsonl`，留证据但不占用 attempt。`batch.py` 当时没有任何测试，这是它能
溜过去的原因；补了 23 个（`tests/test_batch.py`），其中 8 个会在退回旧行为时失败。

跑 baseline 前撞到一个必须先解决的问题：**35 步回合放不进上下文窗口**。实测 448 个
真实 observation，搜索结果页均值 2406 tokens（p90 2676），占全部 observation 开销的
91%；固定前缀 3494。按均值外推 35 步要 ~39k tokens，而窗口是 24576 —— **第 18 步就
撞 HTTP 400**，回合到此结束，前面的步数白跑。窗口开到模型上限 32768 也只推到第 25 步。

分两层解决。分类层：超上下文单独作 `ContextOverflowError` / `Status.CONTEXT_OVERFLOW`，
**不算** infra 失败——它和 `max_steps` 同类，是这道题的一个结局，算成 infra 会让一条长
回合掐掉整批采集。机制层：`rollout/context.py` 在发请求前压缩历史。

压缩策略与参考实现的关键差别：参考实现整组丢弃旧历史，而这会删掉「已经搜过哪些词」的
全部记录——`repeat_loop` 是 **−0.65**，专罚重复动作，于是压缩本身就在制造它要避免的
失败。这里删正文但留一条有界的动作摘要（超上限时丢最老的，防重复看的是最近做过什么）。
实测 35 步回合：未压缩峰值 39084，压缩后稳定在 ~21.6k，35 个查询全部仍可见。
7 个测试会在退回整组丢弃时失败。

配套的 `rollout/tokens.py` 必须和服务端算得一样——数少了照样撞 400，数多了白删历史。
两处反直觉的开销：chat template 的 `| tojson` 把中文工具 schema 转义成 `\uXXXX`，957 →
2818 tokens（**每请求多付 1861**）；`arguments` 是 JSON 字符串又被 tojson 二次序列化。
按真模板校准后偏差 1.55% 且偏高（安全方向：早压一点而不是撞墙）。
`python scripts/check_token_counter.py --trajectories <轨迹>` 可随时复核，换模型必跑。

### 阶段 D — 训练

- [ ] 8 卡分布式底座，SFT 与 GRPO 共用
- [x] 工具调用格式改为 `hermes`（参考实现用 `qwen3_coder`，非 Qwen2.5 格式）
- [ ] 7B 全参 SFT
- [ ] GRPO：不 offload，`gpu_memory_utilization` 提到 0.8+，group size 按实测放大
- [ ] ablation 并行（每个并行实验需独立环境端口，否则互抢 slot）

参考实现 GRPO 的 group size 只有 4，组内优势信号方差过大，是其 RL 增益仅
1.5 个点的一个合理怀疑方向。

`hermes` 在 `serve_model.sh` 已配好，并离线验证过：vLLM 0.25.1 里解析器路径变了
（不再是 `vllm.entrypoints.openai.tool_parsers`），改用 `register_lazy_module`
注册表，`ToolParserManager.get_tool_parser("hermes")` 能取到
`Hermes2ProToolParser`。12 个工具 schema 全部合法，中文 query、asin、空参数三种
情况往返都正确。

装训练依赖时注意：PyPI 的 torch==2.11.0 是 cu130 轮子，会把 cu128 覆盖回去
（本机 driver 只到 CUDA 12.8）。`serve_model.sh` 有前置检查，训练入口还没有，
加训练脚本时要补上同样的检查。

## 从参考实现继承什么

值得读的：

- `src/shopping_grpo/environment/` — 动作校验、观测投影、product_id 规范化
- `src/shopping_grpo/evaluation/blind_guard.py`、`task_facts.py` — 评测隔离
- `data/*/metadata.json` — sha256 + provenance 血缘格式
- 34 个测试文件 — 改环境适配层时的安全网

不沿用的：单卡配置、2,250 条数据切片、串行评测。
