# 环境侧观察记录

这些是搭建期对 ShopSimulator v2.1 / Reward v3 的实测记录。不属于流水线主线，
但在解读奖励曲线和设计分层报告时需要知道。

## 环境代码是评分器，所以它需要一个内容锚

`third_party/` 不入 git（`.gitignore:21`）、没有 patch 机制，而它里面的 engine 层
**就是评分器本身**：`reward.py` 的终局判定、`termination.py` 的 `repeat_loop` 阈值、
`configs/environment.json` 里 `"wrong_purchase": -0.85` 这些权重。商品数据一直有
SHA-256 把关，计算 reward 的代码此前**没有任何把关**——改一行，本仓库全部数字静默
失去可复现性，而且没有任何机制会报警。

`EMBEDDED_SOURCE.json` 里确实记了 `upstream_base_commit` 与 `source_commit`，但它躺在
被 gitignore 掉的树里（我们仓库没有副本），而且只是一句**声明**：没有任何东西核对文件
是否真的等于那个 commit。

现在锚记在 `data/environment/manifest.json`（在 git 里），46 个文件逐个 SHA-256 加一个
根哈希：

```
734d7100472682f49956f4d1b21ed097cf2a2335dff77d538ef3d0f324bae613
```

**本仓库所有已发布的数字都出自这个环境版本。**

```bash
python scripts/hash_environment.py           # 校验，漂移返回 1
python scripts/hash_environment.py --write   # 首次落锚 / 有意升级环境后重锚
```

三处设计上的判断：

**记逐文件哈希，不只记根哈希。** 漂移时能直接说出是哪个文件变了。"根哈希不一致"等于
让人从头 diff 一棵 6 千行的树，而真实改动通常只有一两个文件——多半还是 reward 权重。

**排除派生产物。** `products.sqlite3` 与 `products.manifest.json` 是 `build_index.py`
的输出，重建后字节可能不同（`index_sha256`、`sqlite_version`、`python_version` 都在
manifest 里），纳入会让"重跑一次 setup"表现成漂移——**一个每次都喊狼来了的闸门比没有
闸门更糟**，人会养成习惯性 `--write` 抹掉告警。它们的上游输入（商品数据 SHA-256 +
`build_index.py` + `search.py` + `configs/`）都已被清单覆盖，排除不留缺口。同理排除
`.venv-shopsim/`（由 `requirements.txt` 锁定）、`__pycache__/`、商品数据、`static/`、
`*.log`。`tests/test_environment_hash.py` 里那组"排除项变动不得触发漂移"的测试钉的就是
这一条。

**堵掉唯一能绕过锚的口子。** `SHOP_ENV_CONFIG` 可以把 reward 权重指到锚范围之外的文件，
而锚照样显示"一致"。`start_environment.sh` 现在会比对解析后的路径并拒绝启动，确属有意
（例如 reward ablation）要显式设 `SHOPSIM_ALLOW_UNANCHORED_CONFIG=1`。

闸门接在四处：`setup_environment.sh`（首次落锚，之后校验）、`start_environment.sh`（起
服务前，服务一旦加载进内存就只有启动时检查有意义）、`train_grpo.sh` 与 `eval_grpo.sh`
（产出数字的入口，硬闸门）。`run_rollout.py` 另把**实际扫出的**根哈希盖进
`.summary.json` 的 `environment` 字段——一份不说明自己出自哪个环境的轨迹文件，日后无从
判断能不能和别的文件比。盖章只记不拦：smoke 和一次性探查不该因为锚没落就跑不起来。

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
- 「互抢」的机制值得说清，因为它不是简单的并发数相加超限：`EnvironmentPool` 的
  `BoundedSemaphore(slots_per_worker)` 是**每个客户端池各自一份**的。两个进程各自
  以为自己在某个 worker 上独占 4 个租约，实际服务端那个 worker 只有 4 个，于是
  8 个请求抢 4 个 slot。同端口段上永远只跑一个客户端池——串行，或者分端口段。

## slot 会泄漏，而且默认看不见

服务端的 slot 租约**不会自动回收**。`pack_api.py` 只在收到 `release_one` 时释放；
客户端 `pool.py` 的 `_release_env` 又只在 reset 已经拿到 `env_idx` 之后的内层
`finally` 里调用。任何在「服务端已 `slot_pool.acquire()`、客户端尚未持有 env_idx」
之间断掉的回合，都会把那个 slot 永久占住，而且**不会打印 `failed to release`**
（那条告警只覆盖归还请求本身失败的情况）。

实测代价：连着跑完 SFT 补采样（1,475 回合）和两轮中止的 baseline 之后，全池
**32 个 slot 泄漏 25 个**，8 个 worker 里 7 个只剩 1 个可用、1 个是 0。

诊断时有两个陷阱：

- **每个 worker 只试租 1 个是测不出来的**——剩 1 个空位也会返回成功。必须一次性
  试租满 `slots_per_worker` 个再看能拿到几个。
- **满了的 worker 不快速失败**：`MAX_RETRIES=5` × `RETRY_DELAY_SECONDS=5`，服务端
  会把请求挂住 **25 秒**才返回 `Unable to get available environment resource`。
  探测脚本按「每端口一次失败 = 25s」估时，否则会以为是自己卡死。

现象上，泄漏表现为「并发明明没超也报 `env_error`」：有效容量已经从 32 掉到 7，
而 `env_error` ∈ `INFRA_FAILURES` → `run_batch` 中止整批。所以先量容量再调并发，
不要一看到 `env_error` 就往下调并发——那是在给一个错误的解释配一个无效的药。

**已修复（客户端侧）。** `third_party/` 不进版本管理也没有打补丁机制，所以服务端那句
「在 reset 的处理里就 acquire」动不了，只能在客户端把窗口关掉：`pool.py` 记一份本池
持有的 `env_idx` 账本（`_owned`），reset 撞上租约耗尽时先把「服务端认为租出去了、本池
并不持有」的编号逐个 `release_one` 收回来，再重试一次。有了账本才能做**定向**回收，
这是它和 `release_all` 的本质区别：后者无法区分孤儿和别人在飞的回合。

配套的两件事同样重要，少一件就会引入比泄漏更隐蔽的问题：

- **每个 worker 一把 reset 锁。**「服务端分配编号」和「客户端把编号记进账本」之间还有
  一个窗口，回收线程在那一刻看到的是一个未登记的合法编号，会判成孤儿放掉——于是两个
  回合共用同一个 env，观测互相串台。锁的代价接近零：服务端被 GIL 锁死在单核，同一个
  worker 的请求本来就是串行的。
- **归还失败也要销账。** 归还失败时服务端可能仍持有租约，销账正是要让它落进下一次
  回收的范围；继续记着等于给它永久豁免。

回收**只在 reset 失败时就地触发**，不在启动时自动跑：若同端口段上真有第二个客户端池，
自动回收会把对方在飞的回合静默掐掉，而现在的表现是响亮的 `env_error` 中止——一个吵闹的
错误比一次静默的数据串台好得多。

实测（在活着的池子上注入泄漏）：占满 5707 的 4 个 slot 并清掉账本模拟客户端进程猝死，
体检量到 0/4；随后一个正常回合先挂 25 秒撞墙，回收 `[0,1,2,3]`，重试 reset 成功并跑完，
结束后回到 4/4。这条同时验证了回收逻辑赖以判断的那句 `release_one` 响应措辞与真服务一致。

体检与手工回收：

```bash
python scripts/check_environment.py            # 只量不改，有泄漏则退出码 1
python scripts/check_environment.py --reclaim   # 顺手回收（要求没有 rollout 在跑）
```

它跨 worker 并行探测，所以全池耗时约等于**单个**满 worker 的 25 秒而非 25 × 8。注意
`--reclaim` 从独立进程跑时，收益只是「能报出哪几个编号是租着的」，安全性并不比
`release_all` 好：一个新池子什么都不持有，别人的租约在它看来全是孤儿。

长跑前后各量一次容量仍是便宜的保险。中止本身无损——`env_error` 进 `.failures.jsonl`
不占 attempt，重跑同一条命令按 `(task_id, attempt)` 续跑。
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
- **CUDA 版本要靠 compat 库补差**，见下一节。内核驱动 570.172.08 只支持到 CUDA
  12.8，而 vLLM 0.25.1 依赖的 `torch==2.11.0` 是 cu130 轮子。

## CUDA forward compatibility

driver 570.172.08 支持到 CUDA 12.8；vLLM 0.25.1 依赖 `torch==2.11.0`，PyPI 上这个
版本是 cu130 轮子，需要 driver ≥ 580。不处理的话 vLLM 要加载完权重、走到
`gpu_worker.init_device()` 才炸 `NVIDIA driver is too old (found version 12080)`。

三条路试了两条，只有第三条通：

**① 换 cu128 的 torch —— 死路，且失败是静默的。** 版本号不用改、只换构建变体，
依赖约束照样满足，看起来是最干净的办法。但 vLLM 0.25.1 的预编译算子链接
`libcudart.so.13`，cu128 环境里这个库不存在，`import vllm._C_stable_libtorch` 直接
`ImportError`。把残留的 `nvidia/cu13/lib` 塞进 `LD_LIBRARY_PATH` 后 import 能过、
不报任何错，但算子**返回全零**：

```
torch.ops._C.rms_norm  ->  out[0][:4] = [0.0, 0.0, 0.0, 0.0]
参考实现（torch 原生）    ->  ref[0][:4] = [0.1808, 1.2051, 0.7803, 0.9014]
```

`torch.cuda.synchronize()` 之后仍是全零，算子 schema 也确认调用方式没错。也就是说
服务能起、请求能回，但模型输出是垃圾——baseline 的数会全废且极难查。**不要走这条。**

**② 升级内核驱动 —— 需要 root**，且这是共用机器，GPU 0 上有别人的常驻服务。

**③ `cuda-compat`（采用）。** NVIDIA 官方支持的 forward compatibility：数据中心卡
（A800 属于）可以用新版 user-mode driver 配旧内核模块。纯用户态，不动内核模块，
对机器上其他人零影响。验证结果：

```
cuDriverGetVersion: 13000 -> CUDA 13.0     # 系统 nvidia-smi 仍报 12.8
torch 分配+运算: 12.0 | 设备: NVIDIA A800-SXM4-80GB
```

包从 NVIDIA 的 el8 仓库取（注意 `developer.download.nvidia.com` 会 301 到 `.cn`）：

```bash
base=https://developer.download.nvidia.cn/compute/cuda/repos/rhel8/x86_64
curl -sSLO "$base/cuda-compat-13-0-580.178.04-1.el8.x86_64.rpm"
```

本机**没有 rpm / cpio / bsdtar / 7z**，解包只能自己来：RPM 是 lead(96B) + signature
header + header，两个 header 都是 `8e ad e8` magic + nindex + hsize 结构，signature
后要对齐到 8 字节；payload 是 xz 压的 newc cpio。Python 标准库的 `lzma` 够用，
cpio 手工解（`070701` magic + 13 个 8 位十六进制字段）。解包脚本见 git 历史。

`scripts/cuda_env.sh` 负责把 compat 目录前置到 `LD_LIBRARY_PATH`，`serve_model.sh`
已 source 它。**加训练入口时要 source 同一个文件**，否则同样会撞 driver too old。

配套的守卫比较的是 `cuDriverGetVersion`（compat 生效后的实际能力）而不是
`nvidia-smi` 的内核模块版本——后者恒为 12.8，用它比会把正确配置误判成错误。

## 从训练进程里起 vLLM 要先洗环境变量

GRPO 每轮换权重都要重起 vLLM，而 vLLM 是 `accelerate launch` 的孙进程，默认继承全部
环境变量。其中 **`TORCHELASTIC_USE_AGENT_STORE=True` 会让 vLLM 起不来**：torch 的
`_create_c10d_store` 见到它就认定 TCPStore 由 elastic agent 提供，于是不启 daemon，
让那个 `world_size=1` 的进程组只作为 client 去连自己随机选的端口——没人监听，死等到
600 s 超时。

这个故障极难对上原因：报错是 `client socket has timed out`，字面上和 vLLM、和 GRPO
都无关；手工跑 `serve_model.sh` 因为没有这些变量，永远是好的。`RANK` / `WORLD_SIZE` /
`MASTER_*` 同理会让 vLLM 误判自己的拓扑。

`train_grpo.py` 的 `clean_torchrun_env()` 在 fork 前剥掉 `TORCHELASTIC_*`、
`ACCELERATE_*`、`FSDP_*` 及 rank 一族，业务变量（`no_proxy`、`LD_LIBRARY_PATH`、key）
原样传下去——剥太狠会让 vLLM 连不上环境池，同样是静默失败，所以两个方向都有测试钉住。

## GRPO 采样的并发工作点

`benchmark_environment.py` 量的是环境池自己的上限（纯 step，不含推理）。真正决定
GRPO 每轮多久的是**带模型推理的端到端吞吐**，两者差一个数量级，必须单独量。在
`grpo_train` 池上用 SFT 权重实测（单卡 vLLM，TP=1）：

| 并发 | 回合 | 耗时 | 吞吐 | LLM 调用 |
|---|---|---|---|---|
| 8 | 32 | 121.9 s | 0.26 ep/s | 597 |
| 16 | 48 | 145.2 s | **0.33 ep/s** | 904 |
| 32 | 64 | 196.5 s | 0.33 ep/s | 1202 |

16 → 32 吞吐一动不动，多出来的并发全部堵在 vLLM 队列里，只是把每回合的延迟拉长。
**工作点取 16。**

瓶颈在推理侧而不是环境侧，且是 **prefill-bound**：一轮 smoke 的 276 次调用里 prompt
264.6 万 token、completion 只有 9,782 token，比例 270:1。多轮 agent 每步都要把整段
历史重新喂进去，观测又长。所以加并发不会有收益——prefix caching 已经开着（vLLM
`enable_prefix_caching=True`），没有更多余量可挤；要提吞吐只能给 vLLM 加卡。

## 复现

```bash
# 奖励维度激活率
python scripts/measure_reward_dimensions.py --sample 3000 --seed 42

# 容量压测
SHOPSIM_WORKERS=8 SHOPSIM_ENV_SLOTS=2 bash scripts/start_environment.sh
python scripts/benchmark_environment.py --workers 8 --slots-per-worker 2 \
  --concurrency 1 8 16 32 64 --steps 5
```
