#!/usr/bin/env python3
"""给 D1 的重采轨迹补一份血缘 metadata。

**为什么需要单独一份。** `outputs/teacher/sft_train_retry.summary.json` 是采集器写的，
它只描述**最后一次**调用（`planned: 67, skipped: 1152`）——因为 D1 的收集是「跑到满为止」
的循环加一次中途手术，`collect_retry.sh` 被反复调用，每次覆盖 summary。于是盘上没有任何
一份文件记录这 1,219 条轨迹**整体**的来源和内容哈希。

没有血缘的数据等于不可用的数据：D2（用这批数据训 SFT-v2）真要做的时候，第一件事就是问
「这个文件是什么、从哪来、动过没有」。narrative 记在执行日志里，但那不是机器可读的。

**这份 metadata 不对 D2 承诺任何事。** 它只描述已经躺在盘上的东西。采不采用、要不要
改 `SUCCESS_TYPES`，都是另外的决策。

`accepted` 的判据从 `ecom_agent_rl.data.sft` import，不在这里抄一份——两处口径不一致
就会出现「metadata 说有 408 条可用，SFT 构建器只收 403 条」这种查起来很费劲的偏差。

    .venv/bin/python scripts/write_retry_lineage.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ecom_agent_rl.data.sft import SUCCESS_TYPES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRAJ = ROOT / "outputs" / "teacher" / "sft_train_retry.jsonl"
POOL_META = ROOT / "data" / "task_pools" / "sft_train_retry.metadata.json"
SUMMARY = ROOT / "outputs" / "teacher" / "sft_train_retry.summary.json"
OUT = ROOT / "outputs" / "teacher" / "sft_train_retry.metadata.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not TRAJ.exists():
        print(f"!! 缺 {TRAJ.relative_to(ROOT)}")
        return 1

    reward_types: collections.Counter[str] = collections.Counter()
    statuses: collections.Counter[str] = collections.Counter()
    task_ids: set = set()
    accepted = 0
    dupes = 0
    for line in TRAJ.open():
        rec = json.loads(line)
        rt = rec.get("reward_type")
        reward_types[str(rt)] += 1
        statuses[str(rec.get("status"))] += 1
        tid = rec.get("task_id")
        if tid in task_ids:
            dupes += 1
        task_ids.add(tid)
        if rt in SUCCESS_TYPES:
            accepted += 1

    pool = json.loads(POOL_META.read_text()) if POOL_META.exists() else {}
    summary = json.loads(SUMMARY.read_text()) if SUMMARY.exists() else {}
    n = sum(reward_types.values())

    meta = {
        "schema_version": "ecom-retry-trajectories-v1",
        "outputs": {
            "trajectories": {
                "path": str(TRAJ.relative_to(ROOT)),
                "sha256": sha256(TRAJ),
                "records": n,
                "distinct_task_ids": len(task_ids),
                "duplicate_task_id_records": dupes,
            }
        },
        "provenance": {
            "retry_pool": pool.get("outputs", {}).get("retry_pool", {}),
            # 重采池自己的血缘链（源池 + 裁决），照抄不改写，便于一路回溯。
            "retry_pool_provenance": pool.get("provenance", {}),
            "teacher": "deepseek（key 只从 .env 读，不进命令行也不记在这里）",
            "environment": summary.get("environment", {}),
        },
        "report": {
            "accept_types": sorted(SUCCESS_TYPES),
            "accepted": accepted,
            "accepted_rate": round(accepted / n, 4) if n else None,
            "reward_types": dict(sorted(reward_types.items(), key=lambda kv: -kv[1])),
            "statuses": dict(sorted(statuses.items(), key=lambda kv: -kv[1])),
        },
        "history": [
            # 这个文件不是一次采集的产物，动过一次刀，写清楚免得日后看哈希对不上。
            {
                "step": "首轮采集",
                "detail": "1,219 个零 accepted 任务各采 1 次，得 403 条 accepted"
                          "（gold_purchase 401 + valid_alternative_purchase 2）",
            },
            {
                "step": "滤掉空响应后补采",
                "detail": "67 条 status=empty_response 的记录白占任务名额——"
                          "completed_attempts() 按 (task_id, attempt) 跳过而与 status 无关，"
                          "且 empty_response 不在 INFRA_FAILURES 里，所以既不中止整批也永不重试。"
                          "删掉这 67 行（1219 → 1152）后重跑 collect_retry.sh，让 resume 补回。"
                          "备份见 sft_train_retry.jsonl.before_topup.bak。",
            },
            {
                "step": "补采结果",
                "detail": "67 条里 24 条仍是空响应，真跑成的 43 条只成 5 条（11.6%，"
                          "远低于 35% 的基准率）。accepted 403 → 408。空响应多半发生在"
                          "上下文特别长的任务上，而那些任务本身也更难。停在这里。",
            },
        ],
        "notes": [
            "本文件只描述盘上已有的数据，不表示采用它。是否用来训 SFT-v2 是另一个决策。",
            "同批里 valid_alternative_purchase 只有 2 条，而不被接受的 "
            "partial_alternative_purchase 有 166 条——这批难题上「买替代品」几乎从不完整成功。"
            "改 SUCCESS_TYPES 就是改 SFT 的数据分布，不在本轮范围内。",
            "134 条记录没有 reward_type（no_tool_call 110 + empty_response 24），"
            "所以 reward_types 各项之和里的 None 就是这一类，不是漏统计。",
            "summary.json 只反映最后一次 collect_retry.sh 调用，不能当整体审计用。",
        ],
    }

    OUT.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    r = meta["report"]
    o = meta["outputs"]["trajectories"]
    print(f"写入 {OUT.relative_to(ROOT)}")
    print(f"  记录 {o['records']}  去重 task_id {o['distinct_task_ids']}  "
          f"重复 {o['duplicate_task_id_records']}")
    print(f"  sha256 {o['sha256']}")
    print(f"  accepted {r['accepted']} / {o['records']} = {r['accepted_rate']:.1%}"
          f"  （判据 {r['accept_types']}）")
    env = meta["provenance"]["environment"]
    print(f"  环境锚 matches_anchor={env.get('matches_anchor')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
