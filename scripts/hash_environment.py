#!/usr/bin/env python3
"""`third_party/ShopSimulator` 的内容锚：记录并校验环境代码的 SHA-256。

为什么需要它：`third_party/` 不入 git（`.gitignore:21`）、没有 patch 机制，而它里面
那 3,100 行 engine 层**就是评分器本身**——`reward.py` 的终局判定、`termination.py` 的
`repeat_loop` 阈值、`configs/environment.json` 里 `"wrong_purchase": -0.85` 这些权重。
商品数据有 SHA-256 把关（`setup_environment.sh`），计算 reward 的代码此前没有任何把
关：改一行，本仓库全部数字就静默失去可复现性，而且没有任何机制会报警。

`EMBEDDED_SOURCE.json` 里确实记了 `upstream_base_commit` 与 `source_commit`，但它躺在
被 gitignore 掉的树里——我们仓库没有副本——而且它只是一句**声明**，没有任何东西去核对
文件是否真的等于那个 commit。所以锚必须记在我们这边，并且可校验。落锚时会把
`EMBEDDED_SOURCE.json` 的内容一并抄进 manifest，补上"我们仓库没有副本"这半边。

## 范围

纳入人工编写、确定性的文件（`.py` / `.json` 配置 / `.sh` / `.html` 模板 /
`requirements.txt`）。

**排除派生产物**，这是本脚本唯一需要判断力的地方：`search_engine/products.sqlite3`
与 `products.manifest.json` 是 `build_index.py` 的输出，重建后字节可能不同
（`index_sha256`、`sqlite_version`、`python_version` 都在 manifest 里），纳入会让"重跑
一次 setup"表现成漂移告警——一个每次都喊狼来了的闸门等于没有闸门。它们的上游输入
（商品数据 SHA-256 + `build_index.py` + `search.py` + `configs/`）都已被这份清单覆盖，
所以排除不留缺口。

同理排除：`.venv-shopsim/`（依赖由 `requirements.txt` 锁定）、`__pycache__/`、
`shop_env/data/`（商品数据另有 SHA-256）、`static/`（css 与图片，不影响行为）、
`*.log`（运行时日志，每跑一次就变）。

## 为什么记逐文件哈希而不只记一个根哈希

漂移时能直接说出**是哪个文件变了**。"根哈希不一致"这种报错等于让人从头 diff 一棵
6 千行的树，而真实场景里改动通常只有一两个文件——多半还是 reward 权重。

用法：

    python scripts/hash_environment.py                  # 校验，漂移返回 1
    python scripts/hash_environment.py --write          # 首次落锚 / 有意升级环境后重锚
    python scripts/hash_environment.py --json out.json  # 顺手落一份报告

退出码：一致 0，漂移 1，环境或锚文件缺失 2。便于在 shell 里当闸门用。
只用标准库，因此在项目 venv 建起来之前也能跑（`setup_environment.sh` 需要这一点）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOPSIM_ROOT = ROOT / "third_party" / "ShopSimulator"
MANIFEST_PATH = ROOT / "data" / "environment" / "manifest.json"

# 根哈希的算法标识。改了下面任何一条范围规则或摘要方式都要顺手改这个字符串，
# 否则新旧 manifest 会被拿来直接比对，而它们根本不是同一个口径下的数。
ALGORITHM = "sha256-over-sorted-path-and-file-digests-v1"

INCLUDED_SUFFIXES = (".py", ".json", ".sh", ".html", ".txt")
EXCLUDED_PREFIXES = (
    ".venv-shopsim/",                    # 依赖，由 requirements.txt 锁定
    "shop_env/data/",                    # 商品数据，另有 SHA-256 把关
    "shop_env/search_engine/",           # build_index.py 的派生产物，重建后字节可能不同
    "shop_env/web_agent_site/static/",   # css 与图片，不影响行为
)
EXCLUDED_SUFFIXES = (".log",)


def _is_included(relpath: str) -> bool:
    """relpath 是 POSIX 相对路径（相对 SHOPSIM_ROOT）。"""
    if not relpath.endswith(INCLUDED_SUFFIXES):
        return False
    if relpath.endswith(EXCLUDED_SUFFIXES):
        return False
    if relpath.startswith(EXCLUDED_PREFIXES):
        return False
    return "__pycache__" not in relpath.split("/")


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def scan(shopsim_root: Path = SHOPSIM_ROOT) -> dict:
    """扫出 {relpath: sha256} 以及根哈希。路径排序后再摘要，因此与遍历顺序无关。"""
    files: dict[str, str] = {}
    total_bytes = 0
    for path in shopsim_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        relpath = path.relative_to(shopsim_root).as_posix()
        if not _is_included(relpath):
            continue
        files[relpath] = _digest(path)
        total_bytes += path.stat().st_size

    root = hashlib.sha256()
    for relpath in sorted(files):
        # 路径也进摘要：只摘内容的话，两个文件互换名字会得到同一个根哈希。
        root.update(f"{relpath}\n{files[relpath]}\n".encode())

    return {
        "algorithm": ALGORITHM,
        "root": root.hexdigest(),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "scope": {
            "included_suffixes": list(INCLUDED_SUFFIXES),
            "excluded_prefixes": list(EXCLUDED_PREFIXES),
            "excluded_suffixes": list(EXCLUDED_SUFFIXES),
            "excluded_dir_names": ["__pycache__"],
        },
        "files": {relpath: files[relpath] for relpath in sorted(files)},
    }


def _embedded_source(shopsim_root: Path) -> dict | None:
    """把上游 provenance 抄一份进我们的仓库——原件在 gitignore 掉的树里。"""
    path = shopsim_root / "EMBEDDED_SOURCE.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def compare(recorded: dict, actual: dict) -> dict:
    """按逐文件哈希算差异。根哈希只用来快速判等，报告靠这里。"""
    old, new = recorded.get("files", {}), actual["files"]
    changed = sorted(
        path for path in set(old) & set(new) if old[path] != new[path]
    )
    return {
        "added": sorted(set(new) - set(old)),
        "removed": sorted(set(old) - set(new)),
        "changed": changed,
        "algorithm_changed": recorded.get("algorithm") != actual["algorithm"],
        "root_matches": recorded.get("root") == actual["root"],
    }


def _display(path: Path) -> str:
    """仓库内的路径显示成相对的，仓库外的原样显示。

    别用裸的 `relative_to`：它对仓库外的路径抛 ValueError，而调用点在**写盘之后**，
    于是表现成"落锚失败"而其实已经落了——测试用 tmp 目录时正好会踩到。
    """
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _write(actual: dict, shopsim_root: Path, manifest_path: Path) -> None:
    payload = dict(actual)
    payload["embedded_source"] = _embedded_source(shopsim_root)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"已落锚 {_display(manifest_path)}")
    print(f"  根哈希:   {actual['root']}")
    print(f"  文件数:   {actual['file_count']}（{actual['total_bytes']:,} 字节）")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--write", action="store_true",
        help="用当前环境重写锚文件。**只在有意升级环境之后用**——它会把漂移抹掉。",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--shopsim-root", type=Path, default=SHOPSIM_ROOT)
    parser.add_argument("--json", type=Path, help="把校验结果另存一份")
    parser.add_argument(
        "--quiet", action="store_true", help="一致时不打印（漂移照样打印）"
    )
    args = parser.parse_args(argv)

    if not args.shopsim_root.is_dir():
        print(f"环境目录不存在：{args.shopsim_root}", file=sys.stderr)
        return 2

    actual = scan(args.shopsim_root)

    if args.write:
        _write(actual, args.shopsim_root, args.manifest)
        return 0

    if not args.manifest.is_file():
        print(f"锚文件不存在：{args.manifest}", file=sys.stderr)
        print("首次落锚：python scripts/hash_environment.py --write", file=sys.stderr)
        return 2

    recorded = json.loads(args.manifest.read_text(encoding="utf-8"))
    diff = compare(recorded, actual)
    drifted = bool(diff["added"] or diff["removed"] or diff["changed"])

    report = {
        "manifest": str(args.manifest),
        "recorded_root": recorded.get("root"),
        "actual_root": actual["root"],
        "drifted": drifted,
        **diff,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    if diff["algorithm_changed"]:
        # 口径变了就没法比对，直接当失败：宁可要一次人工确认，也不要一个假的"一致"。
        print(
            "锚文件的 algorithm 与本脚本不一致，无法比对：\n"
            f"  锚文件: {recorded.get('algorithm')}\n"
            f"  本脚本: {actual['algorithm']}\n"
            "范围规则改过了。确认无误后用 --write 重锚。",
            file=sys.stderr,
        )
        return 1

    if not drifted:
        if not args.quiet:
            print(
                f"环境代码一致（{actual['file_count']} 个文件，"
                f"根哈希 {actual['root'][:16]}…）"
            )
        return 0

    print("!! 环境代码与锚文件不一致——本仓库已有的数字不再可复现", file=sys.stderr)
    print(f"   锚文件根哈希: {recorded.get('root')}", file=sys.stderr)
    print(f"   实际根哈希:   {actual['root']}", file=sys.stderr)
    for label, paths in (
        ("内容改变", diff["changed"]),
        ("新增", diff["added"]),
        ("删除", diff["removed"]),
    ):
        for path in paths:
            print(f"   [{label}] {path}", file=sys.stderr)
    print(
        "\n   若改动是有意的：先想清楚已有轨迹还能不能和新环境比，再 --write 重锚。\n"
        "   若不是：从 EMBEDDED_SOURCE.json 记的 source_commit 恢复。",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
