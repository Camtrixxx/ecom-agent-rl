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

三处各配了回归测试，`tests/test_train_sft.py` 26 条，全套 392 条通过。为了让逻辑可测，
`val_rounds` / `steps_in_epoch` 从 `main()` 的闭包里提到模块级。

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
