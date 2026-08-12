"""环境内容锚的不变量。

这个闸门有两种失效方式，都要钉住：**漏报**（环境改了却说一致，锚形同虚设）和**误报**
（重建索引、跑一次服务就喊漂移，于是所有人开始习惯性 `--write` 抹掉告警——一个每次都
喊狼来了的闸门比没有闸门更糟）。下面的排除项测试全部属于第二类。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import hash_environment as he  # noqa: E402


# 一棵最小但覆盖所有范围规则的假环境树。取值不重要，路径才重要。
TREE = {
    "EMBEDDED_SOURCE.json": '{"source_commit": "deadbeef"}',
    "shop_env/configs/environment.json": '{"rewards": {"wrong_purchase": -0.85}}',
    "shop_env/requirements.txt": "flask==3.0.0\n",
    "shop_env/start.sh": "#!/bin/sh\necho hi\n",
    "shop_env/web_agent_site/engine/reward.py": "REWARD = 1.0\n",
    "shop_env/web_agent_site/engine/termination.py": "LOOP_LIMIT = 3\n",
    "shop_env/web_agent_site/templates/item_page.html": "<html></html>\n",
    # --- 以下都应被排除 ---
    ".venv-shopsim/lib/python3.10/site-packages/flask/__init__.py": "x = 1\n",
    "shop_env/data/items_eval_train.json": '{"big": "blob"}',
    "shop_env/search_engine/products.manifest.json": '{"index_sha256": "aaa"}',
    "shop_env/search_engine/products.sqlite3": "SQLite format 3\x00",
    "shop_env/web_agent_site/static/style.css": "body{}\n",
    "shop_env/shop_env/shop_agent.log": "2026-08-12 started\n",
    "shop_env/web_agent_site/engine/__pycache__/reward.cpython-310.pyc": "\x00\x01",
}


def build(root: Path, tree: dict[str, str] | None = None) -> Path:
    shopsim = root / "ShopSimulator"
    for relpath, content in (tree if tree is not None else TREE).items():
        path = shopsim / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return shopsim


@pytest.fixture()
def env(tmp_path: Path) -> Path:
    return build(tmp_path)


def write_anchor(shopsim: Path, manifest: Path) -> dict:
    assert he.main(["--write", "--shopsim-root", str(shopsim),
                    "--manifest", str(manifest)]) == 0
    return json.loads(manifest.read_text(encoding="utf-8"))


# --- 范围 -----------------------------------------------------------------

def test_scope_includes_engine_and_config_excludes_derived(env: Path) -> None:
    files = he.scan(env)["files"]
    assert "shop_env/web_agent_site/engine/reward.py" in files
    assert "shop_env/configs/environment.json" in files  # reward 权重，最要紧的一个
    assert "shop_env/requirements.txt" in files
    assert "shop_env/web_agent_site/templates/item_page.html" in files
    # 派生产物、依赖、商品数据、静态资源、日志、字节码一律不进
    for relpath in (
        ".venv-shopsim/lib/python3.10/site-packages/flask/__init__.py",
        "shop_env/data/items_eval_train.json",
        "shop_env/search_engine/products.manifest.json",
        "shop_env/search_engine/products.sqlite3",
        "shop_env/web_agent_site/static/style.css",
        "shop_env/shop_env/shop_agent.log",
        "shop_env/web_agent_site/engine/__pycache__/reward.cpython-310.pyc",
    ):
        assert relpath not in files, f"{relpath} 不该进锚"


def test_scan_is_deterministic(env: Path) -> None:
    assert he.scan(env)["root"] == he.scan(env)["root"]


def test_root_hash_covers_paths_not_only_contents(tmp_path: Path) -> None:
    """两个文件互换内容 → 根哈希必须变。

    只把文件内容摘进根哈希的话，内容的多重集没变，根哈希就不变——而"reward.py 与
    termination.py 内容互换"显然是个必须报出来的改动。
    """
    a = build(tmp_path / "a")
    b = build(tmp_path / "b")
    engine = "shop_env/web_agent_site/engine"
    (b / engine / "reward.py").write_text(TREE[f"{engine}/termination.py"])
    (b / engine / "termination.py").write_text(TREE[f"{engine}/reward.py"])
    assert he.scan(a)["root"] != he.scan(b)["root"]


# --- 漏报 -----------------------------------------------------------------

def test_content_change_is_reported(env: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_anchor(env, manifest)
    target = env / "shop_env/configs/environment.json"
    target.write_text('{"rewards": {"wrong_purchase": -0.80}}', encoding="utf-8")

    report = tmp_path / "report.json"
    rc = he.main(["--shopsim-root", str(env), "--manifest", str(manifest),
                  "--json", str(report)])
    assert rc == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["drifted"] is True
    assert payload["changed"] == ["shop_env/configs/environment.json"]
    assert payload["root_matches"] is False


def test_added_and_removed_are_reported(env: Path, tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    write_anchor(env, manifest)
    (env / "shop_env/web_agent_site/engine/patch.py").write_text("x = 1\n")
    (env / "shop_env/web_agent_site/engine/termination.py").unlink()

    report = tmp_path / "report.json"
    assert he.main(["--shopsim-root", str(env), "--manifest", str(manifest),
                    "--json", str(report)]) == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["added"] == ["shop_env/web_agent_site/engine/patch.py"]
    assert payload["removed"] == ["shop_env/web_agent_site/engine/termination.py"]


def test_algorithm_mismatch_fails_instead_of_passing(env: Path, tmp_path: Path) -> None:
    """口径变了要当失败。宁可要一次人工确认，也不要一个假的"一致"。"""
    manifest = tmp_path / "manifest.json"
    recorded = write_anchor(env, manifest)
    recorded["algorithm"] = "some-older-scheme-v0"
    manifest.write_text(json.dumps(recorded), encoding="utf-8")
    assert he.main(["--shopsim-root", str(env), "--manifest", str(manifest)]) == 1


# --- 误报 -----------------------------------------------------------------

@pytest.mark.parametrize(
    "relpath",
    [
        "shop_env/search_engine/products.manifest.json",  # 重建索引会改它
        "shop_env/search_engine/products.sqlite3",
        "shop_env/shop_env/shop_agent.log",               # 跑一次服务就会改
        "shop_env/data/items_eval_train.json",
        "shop_env/web_agent_site/static/style.css",
        ".venv-shopsim/lib/python3.10/site-packages/flask/__init__.py",
    ],
)
def test_excluded_paths_do_not_trigger_drift(
    env: Path, tmp_path: Path, relpath: str
) -> None:
    manifest = tmp_path / "manifest.json"
    write_anchor(env, manifest)
    (env / relpath).write_text("changed\n", encoding="utf-8")
    assert he.main(["--shopsim-root", str(env), "--manifest", str(manifest)]) == 0


def test_new_pycache_does_not_trigger_drift(env: Path, tmp_path: Path) -> None:
    """跑一次 import 就会生成 .pyc，这不能算漂移。"""
    manifest = tmp_path / "manifest.json"
    write_anchor(env, manifest)
    pycache = env / "shop_env/web_agent_site/__pycache__"
    pycache.mkdir(parents=True, exist_ok=True)
    (pycache / "app.cpython-310.pyc").write_text("\x00", encoding="utf-8")
    assert he.main(["--shopsim-root", str(env), "--manifest", str(manifest)]) == 0


# --- 缺失与落锚 -----------------------------------------------------------

def test_missing_manifest_is_distinct_exit_code(env: Path, tmp_path: Path) -> None:
    """缺锚（2）要和漂移（1）分开：前者是"还没锚"，后者是"锚不上"。"""
    assert he.main(["--shopsim-root", str(env),
                    "--manifest", str(tmp_path / "nope.json")]) == 2


def test_missing_environment_is_distinct_exit_code(tmp_path: Path) -> None:
    assert he.main(["--shopsim-root", str(tmp_path / "nope"),
                    "--manifest", str(tmp_path / "m.json")]) == 2


def test_write_copies_upstream_provenance_into_manifest(
    env: Path, tmp_path: Path
) -> None:
    """原件在被 gitignore 掉的树里，我们仓库必须自己留一份。"""
    recorded = write_anchor(env, tmp_path / "manifest.json")
    assert recorded["embedded_source"] == {"source_commit": "deadbeef"}
    assert recorded["file_count"] == 7


def test_write_outside_repo_does_not_crash(env: Path, tmp_path: Path) -> None:
    """回归：显示路径用过裸 `relative_to`，仓库外的 --manifest 会在**写盘之后**抛
    ValueError——表现成"落锚失败"而其实已经落了。"""
    manifest = tmp_path / "outside" / "manifest.json"
    assert he.main(["--write", "--shopsim-root", str(env),
                    "--manifest", str(manifest)]) == 0
    assert manifest.is_file()
