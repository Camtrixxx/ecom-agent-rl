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
- [ ] 每题 k 次采样 + 置信区间
- [ ] 按 task_id 配对比较
- [ ] 按 `attributes` 数量与品类分层报告
- [x] 评测隔离：移除 reward、gold、raw observation
- [ ] Baseline 跑通，作为对照下界

隔离做在 rollout 层而不是评测层：`observation.split_env_payload` 用白名单过滤，答案
字段在**写盘之前**就被剥离到 `audit`，所以任何读轨迹文件的下游都拿不到 gold。参考
实现把 `goal`/`reward_detail` 原样存进 `terminal_result`，只在喂 Judge 时才过滤。

实测发现两处参考实现没提的泄漏：`goal_options` 在 reset 就给出目标商品规格；
`progress.credited_evidence_added` **每一步**都带 `constraint:<asin>:budget:fail`
这类逐条约束判定，等于把评分器交给模型。两者都已拦掉并有测试钉住。

### 阶段 D — 训练

- [ ] 8 卡分布式底座，SFT 与 GRPO 共用
- [ ] 工具调用格式改为 `hermes`（参考实现用 `qwen3_coder`，非 Qwen2.5 格式）
- [ ] 7B 全参 SFT
- [ ] GRPO：不 offload，`gpu_memory_utilization` 提到 0.8+，group size 按实测放大
- [ ] ablation 并行（每个并行实验需独立环境端口，否则互抢 slot）

参考实现 GRPO 的 group size 只有 4，组内优势信号方差过大，是其 RL 增益仅
1.5 个点的一个合理怀疑方向。

## 从参考实现继承什么

值得读的：

- `src/shopping_grpo/environment/` — 动作校验、观测投影、product_id 规范化
- `src/shopping_grpo/evaluation/blind_guard.py`、`task_facts.py` — 评测隔离
- `data/*/metadata.json` — sha256 + provenance 血缘格式
- 34 个测试文件 — 改环境适配层时的安全网

不沿用的：单卡配置、2,250 条数据切片、串行评测。
