# 环境侧观察记录

这些是搭建期对 ShopSimulator v2.1 / Reward v3 的实测记录。不属于流水线主线，
但在解读奖励曲线和设计分层报告时需要知道。

## Reward v3 的结构

`web_agent_site/engine/reward.py`：两个硬门（price、brand）+ 四维软匹配。

```
DIMENSION_WEIGHTS = {brand: 0.35, model: 0.25, core_functions: 0.25, key_options: 0.15}
```

终局类型与取值：

| reward_type | 取值 |
|---|---:|
| gold_purchase | 1.0 |
| valid_alternative_purchase | 0.55 |
| partial_alternative_purchase | `min(0.25, -0.30 + 0.55 × match_score)` |
| graceful_stop | -0.15 |
| early_abstain | -0.35 |
| max_steps | -0.50 |
| repeat_loop | -0.65 |
| wrong_purchase | -0.85 |
| reward_unverifiable | 0.0 |

硬门任一 FAIL 直接判 `wrong_purchase`；全部 PASS 且四维软匹配全满则按是否命中
目标 asin 区分 gold / valid_alternative；否则走 partial 的连续取值。

## 维度激活率实测

对**全部** 23,421 条任务调用 `compile_reward_features`，统计各维度是否非空
（0 条 compile 失败）。全量跑一次不到 1 分钟，没必要抽样：

| 维度 | 设计权重 | 实际激活率 |
|---|---:|---:|
| brand | 0.35 | 4.9% |
| model | 0.25 | 12.8% |
| core_functions | 0.25 | 99.1% |
| key_options | 0.15 | 98.8% |

`active_weight` 分布（`reward.py` 按激活维度归一化）：

| active_weight | 任务数 | 占比 |
|---|---:|---:|
| 0.40（仅 core_functions + key_options） | 19,052 | 81.3% |
| 0.65 | 2,792 | 11.9% |
| 0.75 | 995 | 4.2% |
| 其他 | 582 | 2.5% |

即 81.3% 的任务里，软匹配实际只有两维参与，归一化后
`core_functions` 占 62.5%、`key_options` 占 37.5%。权重最高的 brand 在 95.1% 的
任务中不参与打分。

成因在 `reward_features.py`：`_explicit_brand` 要求品牌别名同时出现在
instruction 与目标商品的 title/shop_name 中；`_explicit_models` 要求型号 token
在 instruction 与 title/description 中同时出现。这两个条件在真实任务里都不常
满足。而 `core_functions` 直接取 `instruction.attributes`，几乎总是非空。

另有 64.1% 的任务能从 instruction 解析出显式预算，其余走
`deterministic_price_upper` 的哈希采样兜底。

## 任务与商品一对一

23,421 个商品每个恰好带 1 条 instruction，因此 `task_id` 与商品一一对应。阶段 B
「三池 task_id 零重叠」的正确性依赖这一点：若一个商品挂多条任务，仅按 task_id
切分会让同一商品的不同任务落进不同池子，形成商品级泄漏。切分脚本应校验该前提。

## 环境每步都回答案，必须白名单过滤

实测 `reset` 和 `interact` 的返回里含这些不能进 prompt 的字段：

| 字段 | 什么时候给 | 内容 |
|---|---|---|
| `goal_options` | **reset 就给** | 目标商品的规格，如 `["香草奶昔运动衣 绿", "XS：胸围31cm（建议1-3斤）"]` |
| `progress` | **每一步都给** | `credited_evidence_added` 里有 `constraint:<asin>:budget:fail` 这类逐条约束判定 |
| `goal` / `purchase` / `reward_detail` | 终局 | gold asin 与逐维打分 |
| `instruction_simple` / `user_persona` / `reason_key` | reset | 任务的另一种表述与人设 |

前两个尤其要注意：`goal_options` 在回合还没开始时就把答案给了，`progress` 则相当于
每一步都把评分器的中间结果递给模型。参考实现只在喂 Judge 时过滤终局字段，这两处
没有处理。

过滤放在 `observation.split_env_payload`，用白名单：陌生字段直接报错而不是放过，
因为环境升级新增字段时我们需要被吵醒。答案统一收进轨迹的 `audit`，供离线过滤使用。

## 非法动作是静默 no-op，所以守卫必须权威

实测 `click[not-a-button]`：返回 reward 0、done False、页面不变、**不报错**
（`web_agent_text_env.py` 的 `else: status = dict(reward=0, done=False)` 兜底分支）。
也就是说守卫放过一个非法动作，模型会白吃一步且完全收不到反馈。

因此守卫直接读 `observation_state["actions"]` 判合法——这是环境给的权威列表，实测
全为小写且与 `click[...]` 的匹配目标逐字对应（`next >`、`< prev`、`back to search`、
`buy now`、`description` …）。参考实现改成用正则从渲染出的中文文本里抠
`可点击的按钮: [...]`，渲染措辞一改守卫就静默失效。

`select_option` 我们要求模型同时给出 `axis` 与 `value`。环境的 `click[value]` 只按值
匹配，跨轴误选它察觉不到；`available_options` 里有轴→值映射，多要一个参数就能在守卫
层挡住这类错误，也能拒掉同名值出现在两轴的歧义情况。

## 解读上的影响

- 奖励的软匹配部分实质是「属性覆盖率 + 选项匹配」两维，而 `attributes` 是自由
  文本、用模糊匹配判定，精度有限。分析 reward 曲线时不要按四维权重的字面含义解读。
- `constraints.py` 中 `weighted_preferences` 恒为空列表，`brand`/`model`/
  `key_specs` 三个 hard_constraints 字段同样恒空——源码注释明确说明当前任务数据
  没有 hard/soft 标注，不从关键词猜测。
- `attributes` 数量（均值 4.53，中位数 4，最多 22）是免费且可靠的难度轴，用于
  分层报告。分布：1 条 1,409 个、2 条 2,743、3 条 4,207、4 条 4,349、5 条 3,785、
  6 条 2,798、7 条 1,824、8 条以上 2,306。

## 容量：单进程被 GIL 锁死在 1 核

实测（32 slot，每回合 reset + 5 × interact + release_one）：

| 并发 | episodes/s | req/s | p50 ms | p99 ms |
|---|---|---|---|---|
| 1 | 5.27 | 21.1 | 63 | 66 |
| 4 | 4.28 | 17.1 | 226 | 654 |
| 8 | 3.87 | 15.5 | 493 | 1081 |
| 16 | 4.03 | 16.1 | 966 | 1823 |
| 32 | 3.96 | 15.9 | 2091 | 3439 |

吞吐从并发 1 起就不再增长，p50 随并发线性膨胀（63ms → 2091ms，正好 32×）：
请求被完全串行化，并发只是在排队。**加 slot 不会提升吞吐**，参考实现默认的 8
个 slot 和这里的 32 个在吞吐上没有区别。

原因是 GIL，不是 I/O 也不是锁竞争。`pidstat` 显示进程稳定占用 100% CPU
（单核跑满，机器有 128 核）；`py-spy` 在负载下抓到 9 个 `process_request_thread`
里恰好 1 个 `active+gil`，其余全部停在纯 CPU 代码上等 GIL：

- `jinja2/environment.py:_compile` — 渲染 HTML 页面
- `bs4/element.py:_find_all` — 把刚渲染的 HTML 解析回结构化观测
- `web_agent_site/engine/search.py:261` — BM25 检索
- `web_agent_site/engine/engine.py:110 read_html_template` — 每请求读模板文件

观测走了「渲染成 HTML 再解析回来」的往返，这是 WebShop 的历史包袱，也是单请求
63ms 的主要来源。

单进程 SLO：**~4-5 episodes/s、~20 req/s、p50 63ms**，并发 1 即饱和。

## 多进程扩展是线性的

会话状态在进程内——`reset` 返回的 `env_idx` 只对分配它的那个进程有意义——所以不能
用无状态负载均衡。做法是起 N 个各占一个端口的独立进程，由客户端
`EnvironmentPool` 在回合内粘连到同一端口（`scripts/start_environment.sh`
的 `SHOPSIM_WORKERS`）。

8 worker 实测（每 worker 4 slot）：

| 并发 | episodes/s | req/s | p50 ms | p99 ms |
|---|---|---|---|---|
| 1 | 4.33 | 17.3 | 65 | 127 |
| 8 | **38.34** | 153.4 | 64 | 68 |
| 16 | 31.92 | 127.7 | 112 | 577 |
| 32 | 31.75 | 127.0 | 211 | 785 |
| 64 | 30.44 | 121.8 | 221 | 7260 |

32 worker 实测（每 worker 2 slot，256 回合/档）：

| 并发 | episodes/s | req/s | p50 ms | p99 ms |
|---|---|---|---|---|
| 32 | **133.40** | 533.6 | 65 | 155 |
| 64 | 109.44 | 437.7 | 117 | 729 |

扩展到 32 进程仍然线性：

| workers | 峰值 ep/s | 相对单进程 | 每 worker |
|---|---|---|---|
| 1 | 4.3 | 1.0× | 4.3 |
| 8 | 38.3 | 8.9× | 4.8 |
| 32 | 133.4 | 30.8× | 4.2 |

最优工作点是**并发 = worker 数**：此时每个 worker 恰好 1 个回合在飞，p50 稳定在
单进程空载水平（64-65ms，两种规模下都一样）。再往上加并发只会排队——8 worker 时
吞吐从 38.3 掉到 ~31，p99 从 68ms 恶化到 7.3s。slot 只是回合交替的缓冲、不是吞吐
来源，每 worker 给 2-4 个就够。

32 worker 池子占 15.4GB RSS（机器有 1TB），内存不是约束。

## 并发配置口径

- 容量按 **worker 数 × ~4.2 episodes/s** 估算（保守取 32 worker 实测值），不要按
  slot 数估算。
- 客户端并发设成 worker 数；调大只会让尾延迟爆掉而吞吐不变。
- rollout、评测、教师采集三处共用同一个池子。并行跑 ablation 时每个实验必须用
  独立的 `SHOPSIM_BASE_PORT` 段，否则互抢。
- 机器 128 核。32 worker 是当前验证过的配置；留出的核给训练进程和 vLLM。

参考点：按 133 ep/s 算，7,500 条教师轨迹的采集时间约 1 分钟量级的环境开销——
真正的瓶颈会是教师模型的 API 延迟，不是环境。500 题 × 8 次采样的评测同理。

就绪探测只做 TCP connect，不发任何请求。`pack_api.py` 先跑完
`initialize_environments()` 才 `app.run()`，所以「端口在监听」已等价于「env 就绪」。
不能用 `release_all` 探活——服务端会 `slot_pool.reset()` 清掉该 worker 上全部租约，
并行 ablation 撞同一端口段时会静默掐掉别人在飞的回合。

## 这台机器上的端口、显卡与 CUDA 构建

共用机器，三处默认值不能照抄通用配置：

- **端口 8000 已被占用**（非本项目进程）。vLLM 默认端口改为 8180，`serve_model.sh`
  启动前先探测，占用则直接退出——否则要等权重加载完才报错。
- **GPU 0 上有常驻服务**（他人的 embedding-api 约 4.6G + 本人的 aef_inference 约
  3.4G），GPU 1-7 空闲。`serve_model.sh` 默认从 1 号卡起按 `TP_SIZE` 连续取，因此
  可用卡是 7 张；要占满 8 张须显式设 `CUDA_VISIBLE_DEVICES`。
- **torch 必须装 cu128 构建**。driver 570.172.08 只支持到 CUDA 12.8，而 PyPI 上
  `torch==2.11.0` 是 cu130 轮子，需要 driver ≥ 580。装错的表现有欺骗性：
  `torch.cuda.is_available()` 返回 True（它不真正建 CUDA context），`nvidia-smi`
  一切正常，直到 vLLM 的 `gpu_worker.init_device()` 才炸
  `NVIDIA driver is too old (found version 12080)`。
  版本号本身不用改，只换构建变体，vLLM 0.25.1 的依赖约束照样满足：

  ```bash
  uv pip install --python .venv/bin/python \
    "torch @ https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl"
  ```

  `pyproject.toml` 的 `serve` 组只能写 `torch==2.11.0`（PEP 508 不允许在
  版本号里钉 local version 又要求可解析），所以**重建 venv 或装训练依赖时会被
  PyPI 的 cu130 覆盖回去**，之后必须重跑上面这条。检查方法：
  `python -c "import torch; print(torch.version.cuda)"` 要是 `12.8`。

## 复现

```bash
# 奖励维度激活率
python scripts/measure_reward_dimensions.py --sample 3000 --seed 42

# 容量压测
SHOPSIM_WORKERS=8 SHOPSIM_ENV_SLOTS=2 bash scripts/start_environment.sh
python scripts/benchmark_environment.py --workers 8 --slots-per-worker 2 \
  --concurrency 1 8 16 32 64 --steps 5
```
