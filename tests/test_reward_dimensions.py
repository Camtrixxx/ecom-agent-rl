"""锁定 Reward v3 维度激活率的实测结论。

docs/environment-notes.md 记录了这些数字，分层报告与奖励解读都依赖它们。
调整奖励权重时本测试应当失败——那是提醒去更新文档，不是 bug。
"""

from __future__ import annotations

import gzip
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SHOP_ENV = ROOT / "third_party/ShopSimulator/shop_env"
PRODUCTS = SHOP_ENV / "data/fine_items_eval_train_all.json.gz"

pytestmark = pytest.mark.skipif(
    not PRODUCTS.is_file(),
    reason="ShopSimulator product archive is absent; run scripts/setup_environment.sh",
)

SAMPLE_SIZE = 3000
SEED = 42


@pytest.fixture(scope="module")
def features():
    sys.path.insert(0, str(SHOP_ENV))
    from web_agent_site.engine.reward_features import compile_reward_features

    with gzip.open(PRODUCTS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    sample = random.Random(SEED).sample(products, SAMPLE_SIZE)

    rows = []
    for item in sample:
        for record in item.get("instructions") or []:
            rows.append(compile_reward_features(record, item))
    return rows


def _rate(rows, key) -> float:
    return sum(1 for row in rows if row[key]) / len(rows)


def test_sample_yields_one_instruction_per_product(features):
    assert len(features) == SAMPLE_SIZE


def test_brand_and_model_rarely_activate(features):
    """两个高权重维度（brand 0.35、model 0.25）在多数任务上不参与打分。"""
    assert _rate(features, "expected_brand") < 0.10
    assert _rate(features, "expected_model") < 0.20


def test_core_functions_and_options_almost_always_activate(features):
    assert _rate(features, "expected_core_functions") > 0.98
    assert _rate(features, "required_options_by_key") > 0.95


def test_effective_scoring_collapses_to_two_dimensions(features):
    """多数任务的软匹配只有 core_functions + key_options 生效。"""
    weights = {"brand": 0.35, "model": 0.25, "core_functions": 0.25, "key_options": 0.15}
    keys = {
        "brand": "expected_brand",
        "model": "expected_model",
        "core_functions": "expected_core_functions",
        "key_options": "required_options_by_key",
    }
    two_dimension_only = sum(
        1
        for row in features
        if abs(
            sum(weights[name] for name, key in keys.items() if row[key])
            - (weights["core_functions"] + weights["key_options"])
        )
        < 1e-9
    )
    assert two_dimension_only / len(features) > 0.75


def test_explicit_budget_covers_most_but_not_all_tasks(features):
    """其余任务走 deterministic_price_upper 的哈希兜底。"""
    from web_agent_site.engine.constraints import explicit_budget_from_instruction

    with gzip.open(PRODUCTS, "rt", encoding="utf-8") as handle:
        products = json.load(handle)
    sample = random.Random(SEED).sample(products, SAMPLE_SIZE)

    instructions = [
        record.get("instruction") or ""
        for item in sample
        for record in item.get("instructions") or []
    ]
    explicit = sum(
        1 for text in instructions if explicit_budget_from_instruction(text) is not None
    )
    assert 0.55 < explicit / len(instructions) < 0.72
