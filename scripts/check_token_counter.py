#!/usr/bin/env python
"""校准 `rollout/tokens.py` 的计数器：拿真 chat template 当基准。

计数器决定何时压缩上下文。数少了照样撞 400（压缩白做），数多了白删历史。
两者都不会报错，只会让长回合悄悄变差——所以这个偏差必须能随时量。

模型或 chat template 一换就重跑：

    python scripts/check_token_counter.py --trajectories <轨迹.jsonl>

无轨迹时用合成消息，也能覆盖各角色与 tool_call 的开销。
偏差超过 --max-error 即非零退出，可挂进 CI。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ecom_agent_rl.environment.tools import TOOL_SCHEMAS  # noqa: E402
from ecom_agent_rl.rollout.tokens import DEFAULT_TOKENIZER, TokenCounter  # noqa: E402

MODEL_DIR = Path(DEFAULT_TOKENIZER).parent


def truth_counter(model_dir: Path):
    """用真模板 + 真 tokenizer 渲染，得到服务端实际会收的 token 数。"""
    import jinja2
    from tokenizers import Tokenizer

    config = json.loads((model_dir / "tokenizer_config.json").read_text())
    template = jinja2.Environment().from_string(config["chat_template"])
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))

    def count(messages: list[dict[str, Any]]) -> int:
        text = template.render(
            messages=messages, tools=TOOL_SCHEMAS, add_generation_prompt=True
        )
        return len(tokenizer.encode(text, add_special_tokens=False).ids)

    return count


def synthetic_cases() -> list[list[dict[str, Any]]]:
    call = {
        "id": "c1",
        "type": "function",
        "function": {
            "name": "search_products",
            "arguments": json.dumps({"query": "铝合金雕花板 白色"}, ensure_ascii=False),
        },
    }
    base = [
        {"role": "system", "content": "你是购物助手。" * 20},
        {"role": "user", "content": "帮我找一件狗狗衣服，预算 35。" * 5},
    ]
    turn = [
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "c1", "content": "【搜索结果】" + "商品行\n" * 60},
    ]
    return [base, base + turn, base + turn * 4, base + turn * 12]


def trajectory_cases(path: Path) -> list[list[dict[str, Any]]]:
    cases: list[list[dict[str, Any]]] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        messages = json.loads(line).get("messages") or []
        for cut in (2, 6, 12, len(messages)):
            if 0 < cut <= len(messages):
                cases.append(messages[:cut])
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, help="真轨迹 jsonl，用它的 messages 做样本")
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--max-error", type=float, default=0.02, help="容许的最大相对偏差")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    cases = synthetic_cases()
    if args.trajectories:
        cases += trajectory_cases(args.trajectories)
    cases = cases[: args.limit]

    truth = truth_counter(args.model_dir)
    counter = TokenCounter(str(args.model_dir / "tokenizer.json"))

    worst = 0.0
    signed_max = 0
    print(f"{'消息数':>6}{'真实':>9}{'计数器':>9}{'差':>8}{'相对':>9}")
    for messages in cases:
        actual = truth(messages)
        mine = counter.count_messages(messages, TOOL_SCHEMAS)
        error = abs(mine - actual) / actual if actual else 0.0
        worst = max(worst, error)
        if abs(mine - actual) > abs(signed_max):
            signed_max = mine - actual
        if error > args.max_error * 0.5 or len(cases) <= 12:
            print(f"{len(messages):>6}{actual:>9}{mine:>9}{mine - actual:>8}{error:>8.2%}")

    print(f"\n{len(cases)} 个样本，最大相对偏差 {worst:.2%}，最大绝对差 {signed_max:+d}")
    if worst > args.max_error:
        print(f"❌ 超出容许的 {args.max_error:.1%}。请重新标定 tokens.py 里的开销常数。")
        return 1
    print(f"✅ 在容许的 {args.max_error:.1%} 以内。")
    if signed_max < 0:
        print("注意：计数器偏低。压缩预算里的安全边际要能盖住这个差。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
