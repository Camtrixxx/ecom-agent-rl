#!/usr/bin/env python3
"""跑一批 rollout：baseline 评测、教师采集、以及 smoke 都走这个入口。

前置：
    bash scripts/start_environment.sh          # 环境池
    bash scripts/serve_model.sh                # 被测模型（或指 --base-url 到教师 API）

用法：
    # baseline：500 题评测集各跑 1 次
    python scripts/run_rollout.py --pool data/task_pools/evaluation.jsonl \\
        --out outputs/rollouts/baseline.jsonl

    # smoke：先拿 3 题验证链路
    python scripts/run_rollout.py --pool data/task_pools/evaluation.jsonl \\
        --limit 3 --out outputs/rollouts/smoke.jsonl

被中断后重跑同样的命令会自动续跑（按 task_id + attempt 去重）。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import hash_environment  # noqa: E402

from ecom_agent_rl.environment.pool import EnvironmentPool  # noqa: E402
from ecom_agent_rl.rollout.agent import DEFAULT_MAX_STEPS  # noqa: E402
from ecom_agent_rl.rollout.batch import load_task_ids, run_batch  # noqa: E402
from ecom_agent_rl.rollout.llm import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_WINDOW,
    ChatClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pool", type=Path, required=True, help="任务池 jsonl")
    parser.add_argument("--out", type=Path, required=True, help="轨迹输出 jsonl（追加）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（smoke 用）")
    parser.add_argument("--attempts", type=int, default=1, help="每题采样次数")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="默认 = worker 数，这是实测的最优工作点")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--no-resume", action="store_true", help="忽略已有输出，从头跑")

    # 三项都可用环境变量给，教师采集时把 key 从命令行拿掉：这是共用机器，
    # 命令行参数在 `ps` 里对所有用户可见，也会留在 shell history 里。
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "ecom-agent"))
    parser.add_argument("--api-key", default=os.environ.get("LLM_API_KEY"),
                        help="默认读 LLM_API_KEY；优先用环境变量而非命令行")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=1024)
    # 必须和服务端 --max-model-len 一致。不给就不压缩，长回合会在 ~第 18 步撞
    # HTTP 400（实测 35 步需要 ~39k tokens，窗口只有 24576）。
    parser.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW,
                        help="模型上下文窗口，须与服务端 --max-model-len 一致；0 表示不压缩")
    parser.add_argument("--context-margin", type=int, default=512,
                        help="计数器与服务端的残差留白（实测偏差 1.55%%）")

    parser.add_argument("--env-host", default="127.0.0.1")
    parser.add_argument("--env-base-port", type=int, default=5700)
    parser.add_argument("--env-workers", type=int, default=8)
    parser.add_argument("--env-slots", type=int, default=4)
    return parser.parse_args()


def _environment_stamp() -> dict:
    """把环境代码的根哈希盖进 summary。

    为什么：一份不说明自己出自哪个环境的轨迹文件，日后无从判断能不能和别的文件比。
    `third_party/` 不入 git，`reward.py` 改一行就会让两批轨迹不可比，而文件里没有
    任何痕迹。盖的是**实际扫出来的**哈希（这批数据真正出自什么），另附一个是否与锚
    一致的布尔量。

    这里只记不拦：拦的位置在 `train_grpo.sh` / `eval_grpo.sh` 那些产出数字的入口，
    而 smoke 和一次性探查不该因为锚没落就跑不起来。
    """
    try:
        actual = hash_environment.scan()
        recorded = {}
        if hash_environment.MANIFEST_PATH.is_file():
            recorded = json.loads(
                hash_environment.MANIFEST_PATH.read_text(encoding="utf-8")
            )
        return {
            "code_root_sha256": actual["root"],
            "file_count": actual["file_count"],
            "matches_anchor": bool(recorded) and recorded.get("root") == actual["root"],
            "anchor_present": bool(recorded),
        }
    except (OSError, KeyError, ValueError) as exc:
        # 盖不上章不构成丢数据的理由，但要留下"为什么盖不上"。
        return {"code_root_sha256": None, "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    if not args.pool.exists():
        raise SystemExit(f"任务池不存在: {args.pool}\n先跑 python scripts/build_task_pools.py")

    # 解析口径必须在日志第一屏就说清楚，因为它会改成功率：宽容口径把一部分
    # no_tool_call 变成正常步骤或 truncated，两侧的数不能直接比。轨迹记录里也有
    # `tolerant_parse` 标记（见 agent.Trajectory），这里再打一遍是为了让**看日志的人**
    # 不必去 grep jsonl 才知道这批数据是哪个口径。
    from ecom_agent_rl.rollout.agent import TOLERANT_PARSE

    if TOLERANT_PARSE:
        logging.warning(
            "宽容重解析已开启（ROLLOUT_TOLERANT_PARSE）：这批轨迹与 08-15 之前"
            "已发布的严格口径数字**不可直接比较**"
        )
    else:
        logging.info("解析口径：严格（与已发布数字一致）")

    task_ids = load_task_ids(args.pool)
    if args.limit is not None:
        task_ids = task_ids[: args.limit]

    pool = EnvironmentPool(
        host=args.env_host,
        base_port=args.env_base_port,
        workers=args.env_workers,
        slots_per_worker=args.env_slots,
    )
    pool.wait_until_ready(timeout=600.0)

    client = ChatClient(
        base_url=args.base_url,
        model=args.model,
        api_key=args.api_key,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        context_window=args.context_window or None,
        context_margin=args.context_margin,
    )

    summary = run_batch(
        pool=pool,
        client=client,
        task_ids=task_ids,
        output=args.out,
        attempts=args.attempts,
        concurrency=args.concurrency,
        resume=not args.no_resume,
        max_steps=args.max_steps,
    )

    summary["environment"] = _environment_stamp()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n汇总写入 {summary_path}")
    if summary["aborted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
