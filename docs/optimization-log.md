# 优化计划执行日志

[optimization-plan.md](optimization-plan.md) 的执行记录。**按时间顺序追加，不回改**——
和预注册一样，回改过的执行日志没有价值。

走的是计划里的**路 A**：冻结 SFT 数据，把已有结论做扎实（S1/S2/S3 + E1 + E2 + R1），
同时并行采 D1 的数据（不占 GPU、不承诺任何后续动作）。D2（采用新数据 → SFT-v2 →
GRPO-v2）留作后面的决策点。

## 2026-08-13 凌晨

### 卡的分配（与计划的偏差，必须记进结论）

计划里 S4 假定 7 张卡。实际 E1 评测占着 GPU 1-2（它在两张卡上交替预热下一个模型），
所以 **S4 用 GPU 3-7 五张**。串行等 E1 跑完再用 7 卡要 6.1 h，并行只要
max(3.8 h, ~3.3 h)。代价是步数变了：

| 卡数 | per_rank = 648/卡 | steps/epoch = per_rank//4 | 3 epoch 共 |
|---|---|---|---|
| 6（已发布的 SFT） | 108 | 27 | 81 |
| 7 | 92 | 23 | 69 |
| **5（本次 S4）** | **129** | **32** | **96** |

S4 内部的三处比较（e3 / e2 / e3-epoch2）共用同一个 world_size，自身自洽；且 S1 修好
之后 val loss 已经与卡数无关。但 **S4 的 val loss 与已发布 SFT 的 0.39249 不可直接比**
——后者是在丢掉 3 条最长验证样本的口径下量的（见下）。

**更要紧的一条：`sft` vs `sft_e3` 这一对不能当作「重训的 run 间方差」来读。** 卡数变了
就等于 optimizer 的 global batch 变了——`grad_accum 4 × 5 卡 = 20 批`，而已发布的 SFT
是 `4 × 6 = 24 批`。所以那一对同时混了「重训」和「批大小」两件事，只能当参考锚。
S4 真正要回答的「第 3 个 epoch 值不值」完全落在 5 卡这一族内部，不受影响。

**推论：R1 的两个 seed 必须用和已发布 seed 42 相同的 6 张训练卡**
（`GPU_VLLM=1, GPUS=2,3,4,5,6,7`，即 `train_grpo.sh` 的默认值）。同样的道理：4 卡的
global batch 是 8 批而 6 卡是 12 批，若 seed 43 用 4 卡、seed 44 用 6 卡，卡数就和 seed
混在一起，恰好毁掉这个实验要量的 run 间方差。因此 R1 只能等 E2 让出 GPU 1-2，不能为了
填满空闲卡而改卡数。

### 排期（受上面的卡数约束）

| 时间 | GPU 1-2 | GPU 3-7 |
|---|---|---|
| 02:35 → ~07:00 | E1（9 个检查点） | 03:09 → ~05:15 S4a `sft_e3`；→ ~06:40 S4b `sft_e2` |
| ~07:00 → ~08:50 | E2（4 份 SFT 权重） | 空闲（不能拿去跑 R1，见上） |
| ~08:50 → ~22:00 | R1：seed 43 然后 seed 44，各 6.56 h，占 GPU 1-7 | |
| → ~23:30 | R1 两个 seed 各一次 500 题 k=4 评测 | |

D1（不占 GPU）02:31 → ~05:35。

GPU 0 上有别人的常驻服务，全程不碰。

### S1/S2/S3：train_sft.py 三处修复

**S1（错）** `evaluate()` 之前用 `shard()` 切验证集。`shard()` 为了对齐 FSDP 的集合
通信把各 rank 截到等长，丢掉尾部 `len(batches) % world_size` 批。训练侧丢一点无所谓
（每个 epoch 重新 shuffle，长期每条都会被采到），验证侧是致命的：
`token_budget_batches` 先按长度排序且 val 从不 shuffle，所以每个 epoch 丢的**永远是
同样那几条最长的样本**，而丢几条**取决于卡数**——6 卡丢 3 条、7 卡丢 6 条、8 卡丢 7 条。
已发布的 0.39249 实际是 295 条样本上的数，且换卡数就变。

修法：新增模块级 `val_rounds()`，按 `batches[rank::world_size]` 分，短的 rank 用
**零权重**重复自己的第一批补齐轮数。轮数对所有 rank 相同（不然 FSDP 卡死在 all-gather），
每条样本恰好计一次权重 1.0。

**S2（缺）** val loss 只进内存 `history`，中途崩了就全丢；且没有 epoch 0 的基线点，
「第 3 个 epoch 只降 0.0032」这类判断没有起点可依。修法：`emit()` 每条记录即时追加到
`train_log.jsonl`；训练前先量一次记为 epoch 0；`metadata.json` 里把含糊的 `final`
拆成 `final_train` + `val_curve`。

**S3（错）** `steps_per_epoch` 用 `math.ceil`，而训练循环是
`range(0, per_rank - grad_accum + 1, grad_accum)`，只走完整的 grad_accum 组，实际步数
恒为 `per_rank // grad_accum`。ceil 会让 `total_steps` 超过所有 epoch 加起来能走的步数，
于是 while 多跑一个残 epoch 去补，余弦在残 epoch 中途才退到 0，`--save-each-epoch`
还会多写一个目录。之前没暴露纯属整除的巧合：6 卡 per_rank=108、7 卡 92 都被 4 整除；
8 卡 per_rank=81 就会踩（ceil 21 vs floor 20 → total 63 而 3 个 epoch 只有 60 步）。
顺带补了 `steps_per_epoch == 0` 的守卫——那种情况下 while 只看 `step < total_steps`，
会永远转下去。

三处各配了回归测试，`tests/test_train_sft.py` 26 条，全套 389 passed / 3 skipped。为了让
逻辑可测，`val_rounds` / `steps_in_epoch` 从 `main()` 的闭包里提到模块级。

**冒烟**（5 卡，`--limit 48 --max-train-steps 2`）：epoch 0 量到 val 1.024，1 步后 0.798，
末步 lr 退到 0，`metadata.json` 里 `val_curve` 两点齐全、`final` 已消失。

先在 2 卡上试过一次，rank0 在 epoch-0 验证之后退出 1，当时 GPU 4 已到 79.9/80 GB。
最可能是 2 卡下 FSDP 只把优化器状态切成两份（约 40 GB/卡，6 卡时约 13 GB）导致的 OOM，
**但当时用了 `| tail -30`，原始异常被截掉了，这个归因是推断而非实证。** 换到 5 卡即通过，
而 5 卡正是 S4 要用的配置，所以没有继续追这条。

顺带核了两件与改动相邻的事：`evaluate()` 结束时会 `model.train()` 回去
（`train_sft.py:376`），且 Qwen2.5 的 `attention_dropout=0.0`，所以我新加的「训练前先量
一次」不会改变 epoch 1 的训练行为；`save()` 只 gather 一份副本转 bf16 写盘、两端各一道
barrier，中途存不动活模型。`--save-each-epoch` 的 `if ... and not stop` 保证最后一个
epoch 不重复存——而这一条恰好依赖 S3 的 floor 口径，ceil 时会多进一个残 epoch、多写一个
`epoch3/`。

### 新增脚本

| 脚本 | 用途 |
|---|---|
| `build_retry_pool.py` | 挑出 1,219 个零 accepted 任务，带 sha256 血缘 |
| `collect_retry.sh` | D1 的「跑到满为止」循环，端口段 5800（避开 E1 的 5700） |
| `eval_checkpoints.sh` | E1：9 个模型 × 200 题 × k=4，两卡交替预热 |
| `run_s4.sh` | S4：串行跑 sft_e3 / sft_e2 |
| `eval_sft_variants.sh` | E2：4 份 SFT 权重互比 |
| `eval_seed.sh` | R1：单个 seed 的 500 题评测 |
| `summarize_reports.py` | 把一堆 report.json 汇成表，并按预注册的并列规则给结论 |

`collect_teacher.sh` 改了一处：`set -a; source .env` 会无条件覆盖 `SHOPSIM_BASE_PORT`
和 `SHOPSIM_WORKERS`，导致调用方没法错开端口段。现在先存下调用方的值，source 之后再
覆盖回去。同端口段并行跑两个客户端池会互抢租约 → `env_error` → 整批中止。

### 预注册

R1 的判定规则写进了新文件 [multiseed-preregistration.md](multiseed-preregistration.md)。
**没有**改 `eval-preregistration.md`——那份末尾写着「此后不再改动」。核心是提前写死：
3 个 seed 的 t 区间跨 0 就降级报「测不出 run 间稳定性」，**不补第 4 个 seed**。

### 成本的实测修正

| 项 | 计划估计 | 实测 | 说明 |
|---|---|---|---|
| D1 | 0.21 ep/s → 1.6 h | 0.11 ep/s → **~3 h** | 重采池全是已失败的难题，`repeat_loop`(478) 和 `max_steps`(144) 会走满 35 步 |
| E1 | 0.57 ep/s | 0.49 ep/s → ~27 min/模型，9 个约 **4 h** | |

D1 只烧 wall-clock 和 API 额度，0 GPU，所以放着跑。

## 2026-08-13 03:30 前后

### E1 的头两个模型（早读，非最终结论）

| 模型 | 成功率 | 95% CI | vs sft |
|---|---|---|---|
| sft | 58.93% | [53.09, 64.59] | （基线） |
| iter004 | 61.99% | [56.51, 67.73] | +2.76 pp [-0.94, +6.21] |

各 800 回合 0 基础设施失败。iter004 的区间跨 0——训到第 5 轮还没走出噪声，和已发布的
iter039 +7.41 pp 放一起是条讲得通的爬升曲线。**并列规则要等 9 个模型齐了再按
`summarize_reports.py` 判，不许看着这两行提前挑。**

E1 实测约 20 min/模型（估的 27），换模型的预热把间隔压到 23 秒，9 个约 3.3 h 而非 4 h。

### D1 的产出率与一个会静默吞任务的机制

415 回合（一题一次）：

| reward_type | 条数 | 占比 |
|---|---|---|
| **gold_purchase**（可入 SFT） | 126 | 30.4% |
| repeat_loop | 100 | 24.1% |
| partial_alternative_purchase | 55 | 13.3% |
| （None） | 53 | 12.8% |
| max_steps | 51 | 12.3% |
| wrong_purchase | 20 | 4.8% |
| reward_unverifiable | 9 | 2.2% |
| early_abstain | 1 | 0.2% |

比首批 168 条时的 25% 上修到 30.4%，外推 1219 题约 **370 条**新样本。两处值得记：

1. `valid_alternative_purchase` 在 415 回合里是 **0**，而不被接受的
   `partial_alternative_purchase` 占 13.3%。即这批难题上「买替代品」几乎从不完整成功。
   **没有**因此去动 `SUCCESS_TYPES`——改接受集就是改 SFT 的数据分布，那是 D2 的决策点，
   不是无人值守时能顺手做的事。
2. reward_type 为 None 的 53 条里，26 条是 `empty_response`（DeepSeek API 重试 4 次仍空），
   28 条是 `no_tool_call`。

**空响应会白占一个任务名额。** `completed_attempts()` 按 `(task_id, attempt)` 跳过，
与 status 无关；而 `empty_response` 不在 `INFRA_FAILURES`（那里只有
llm_error/env_error/observation_error），所以既不触发整批中止、也永远不会被 resume 重试。
扣掉它们，D1 的真实产出率是 126/389 = **32.4%**。已挂任务 #7：等主循环退出后滤掉这 26 行
再跑一次 collect_retry，让 resume 正好补回。`no_tool_call` 不补——模型真的没做成，是合法结果。

**同一机制在评测侧核过了，干净**：`ckpt_eval/sft`、`ckpt_eval/iter004`、已发布的
`sft`/`grpo` 四份里都没有 `empty_response`，非 done 的只有 `rejection_limit`（800 中 2 条）
和 `no_tool_call`（800 中 2 条），这两个本来就该算失败。所以成功率没有被这个机制压低。

### S4 进度

`sft_e3` 03:21 开跑，第一条日志就是 S2 新加的训练前基线：
`{"step": 0, "epoch": 0, "val_loss": 0.99489}`（修复后第一次在真实 run 上用到）。
`epoch: 0.031` = 1/32，与 5 卡的 `129 // 4 = 32 步/epoch` 对上。约 65 s/步。
