"""测试共用的 observation 样本。

字段和取值都照抄真实环境（`task_id=7`，狗狗衣服那条任务）的实测输出，不是手编的：
守卫和渲染都直接吃这些结构，样本失真就等于测了个假接口。
"""

from __future__ import annotations

from typing import Any

from ecom_agent_rl.environment.observation import OBSERVATION_VERSION

# reset 之后的初始页面。实测 actions 只有 ["search"]。
SEARCH_HOME: dict[str, Any] = {
    "observation_version": OBSERVATION_VERSION,
    "page_type": "search_home",
    "search_available": True,
    "actions": ["search"],
}


def search_state(count: int = 2) -> dict[str, Any]:
    """搜索结果页。注意 `search_available` 是 False——结果页上不能再搜，得先返回。"""
    asins = [f"90000000000{i}" for i in range(count)]
    return {
        "observation_version": OBSERVATION_VERSION,
        "page_type": "search_results",
        "search_available": False,
        "actions": ["back to search", "next >", *asins],
        "query": "狗狗衣服",
        "normalized_query": "狗狗衣服",
        "page": 1,
        "total_pages": 8,
        "total_results": 150,
        "rank_start": 1,
        "rank_end": count,
        "products": [
            {
                "asin": asin,
                "title": f"商品 {i}",
                "brand": "某品牌",
                "category": "宠物›服饰",
                "price": "108.0",
                "key_attributes": ["透气"],
                "rank": i + 1,
            }
            for i, asin in enumerate(asins)
        ],
    }


def detail_state() -> dict[str, Any]:
    """商品详情页。实测这件商品只有 description/features/reviews，没有 attributes 按钮。"""
    return {
        "observation_version": OBSERVATION_VERSION,
        "page_type": "product_detail",
        "search_available": False,
        "actions": ["back to search", "< prev", "description", "buy now", "l", "xs"],
        "product": {
            "asin": "900000000000",
            "title": "狗狗雨衣",
            "brand": "某品牌",
            "category": "宠物›服饰",
            "price": "108.0",
            "key_attributes": ["透气"],
        },
        "available_options": {"尺码": ["l", "xs"]},
        "selected_options": {},
        "selected_price": 108.0,
    }
