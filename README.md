# Ecom Agent RL

面向电商长程 Agent 的后训练与评测项目。7B 模型、万级数据、8 卡训练。

## 流水线

沿用四阶段主线，每阶段可独立运行与复现。

前置（每次开新 shell 都要）：

```bash
bash scripts/setup_environment.sh          # 一次性：装依赖 + 校验商品数据 + 建索引
bash scripts/start_environment.sh          # 环境池：8 worker × 4 slot，端口 5700+
bash scripts/serve_model.sh <权重路径>      # 被测模型的 vLLM 服务，默认端口 8180
python scripts/check_environment.py        # 可选：量环境池还剩多少 slot，有泄漏则退出码 1
```

| 阶段 | 目标 | 入口 |
|---|---|---|
| 数据 | 抽三个 task_id 零重叠的任务池 | `python scripts/build_task_pools.py` |
| Baseline | 测量原始 Qwen2.5-7B-Instruct 的工具使用能力 | `python scripts/run_rollout.py --pool data/task_pools/evaluation.jsonl --out outputs/rollouts/baseline.jsonl` |
| SFT | 从教师轨迹学习合法、完整的购物行为 | `bash scripts/collect_teacher.sh` → `python scripts/build_sft_dataset.py` → `bash scripts/train_sft.sh` |
| GRPO | 在真实环境 Rollout 中优化 Reward v3 | `bash scripts/train_grpo.sh` |
| Evaluation | 在同一批留出任务上公平比较模型 | `run_rollout.py --attempts k` → `python scripts/report_metrics.py --trajectories <轨迹> --baseline <对照> --pool <池>` |

`run_rollout.py` 是 baseline 评测、教师采集、smoke 三者共用的入口，区别只在
`--pool` / `--out` / `--base-url`；被中断后重跑同一条命令即按 `(task_id, attempt)`
续跑。并发有实测上限，见 [docs/environment-notes.md](docs/environment-notes.md)。

## 环境

[ShopSimulator](https://github.com/ShopAgent-Team/ShopSimulator) Environment
v2.1 + Reward v3，位于 `third_party/ShopSimulator/`，不提交进 git。

商品池 23,421 条任务，横跨 9 个品类。每条任务自带 `attributes`（约束列表，
均值 4.53 条）、`pricing`、`customization_options`（规格变体及各自价格）。

授权说明：上游 base commit `51bb26012cee31aea7ac26177c5ffe807026ac07` 只提供
WebShop 式基础壳子；Environment v2.1 与 Reward v3 的 engine 层（约 3,100 行，
含 `reward.py`、`variant_price.py`、`comparators.py`、`termination.py` 等）是
上游之外的下游实现，随环境一并引入。

## 相对参考实现的变化

参考实现 `/data/heyuhang/shopping-grpo-longhorizon`（2B、单卡）的主要约束在于
硬件假设与数据规模，本项目针对这两点调整：

| 项 | 参考实现 | 本项目 |
|---|---|---|
| 模型 | Qwen3.5-2B（多模态） | Qwen2.5-7B-Instruct（纯文本） |
| 训练卡数 | 1（参数+优化器双 offload） | 8（不 offload） |
| SFT 任务池 | — （val 从 accepted 里切） | 3,000 / 500（train 与 val 任务零重叠） |
| SFT 数据集行数 | 800 / 200 | 1,781 / 299（接受率 0.5937） |
| GRPO 任务 | 1,000 / 50 | 3,000 / 200 |
| 评测集 | 200 题 × 1 次 rollout | 500 题 × k 次采样 |
| 评测执行 | 串行 | 并行 |
| GRPO group size | 4 | 按显存实测放大 |

评测改动的原因：参考实现每题只跑一次 rollout，其
`experiments/comparison.md` 自述不估计采样方差，导致 SFT 60.5% 与 GRPO 62.0%
之间 3 道题的差距无法与噪声区分。本项目要求多次采样与置信区间。

换用 Qwen2.5 系列的原因：Qwen3.5 全系列均为多模态
（`Qwen3_5ForConditionalGeneration`，带 `vision_config`），而本任务是纯文本，
vision tower 只会占用显存与参数量。Qwen2.5-7B-Instruct 是纯文本模型，中文原生，
且同系列有 14B/32B 可作后续 scaling 对照，tokenizer 与 chat template 一致。

适配层需相应调整：工具调用格式从 `qwen3_coder` 改为 `hermes`
（Qwen2.5 系列的原生格式），vLLM 启动参数中的 `--tool-call-parser` 同步修改。

## 状态

搭建中。详见 [docs/roadmap.md](docs/roadmap.md)，环境侧观察记录见
[docs/environment-notes.md](docs/environment-notes.md)。
