#!/usr/bin/env python3
"""下载并解包 NVIDIA 的 cuda-compat 包，让 CUDA 13 的 torch 跑在 driver 570 上。

为什么需要：本机内核驱动 570.172.08 只支持到 CUDA 12.8，而 vLLM 0.25.1 依赖的
torch==2.11.0 是 cu130 轮子。cuda-compat 提供新版 user-mode driver，是 NVIDIA
官方支持的 forward compatibility 用法（数据中心卡专有），纯用户态、不动内核模块。
换 cu128 的 torch 那条路是死的——vLLM 的预编译算子会静默返回全零，详见
docs/environment-notes.md。

为什么自己解包：本机没有 rpm / cpio / bsdtar / 7z，一个都没有。所以这里用标准库
解 RPM 头 + xz + newc cpio。

用法：
    python scripts/install_cuda_compat.py                 # 装到默认位置
    python scripts/install_cuda_compat.py --dest /path    # 换位置

装完 source scripts/cuda_env.sh 即可生效。
"""

from __future__ import annotations

import argparse
import lzma
import os
import struct
import sys
import urllib.request
from pathlib import Path

# 301 到 .cn 的那个域名直接写成最终地址，省一次跳转。
REPO = "https://developer.download.nvidia.cn/compute/cuda/repos/rhel8/x86_64"
# cuda-compat-13-0 对应 CUDA 13.0 需要的 580 系列 user-mode driver。
# 钉死版本而不取「最新」：换版本要重新验证数值正确性。
RPM = "cuda-compat-13-0-580.178.04-1.el8.x86_64.rpm"
DEFAULT_DEST = Path("/data/heyuhang/cudacompat")

RPM_LEAD_SIZE = 96
HEADER_MAGIC = b"\x8e\xad\xe8"
CPIO_MAGIC = b"070701"


def read_rpm_header(data: bytes, offset: int) -> int:
    """返回该 header 之后的偏移。header = magic(8) + nindex(4) + hsize(4) + index + store。"""
    if data[offset : offset + 3] != HEADER_MAGIC:
        raise SystemExit(f"RPM header magic 不对 @ {offset}: {data[offset:offset + 8]!r}")
    nindex, hsize = struct.unpack(">II", data[offset + 8 : offset + 16])
    return offset + 16 + nindex * 16 + hsize


def extract_payload(rpm_path: Path) -> bytes:
    data = rpm_path.read_bytes()
    if data[:4] != b"\xed\xab\xee\xdb":
        raise SystemExit(f"不是 RPM 文件: {rpm_path}")
    # signature header 之后要对齐到 8 字节，header 本身不用。
    after_sig = (read_rpm_header(data, RPM_LEAD_SIZE) + 7) & ~7
    payload_offset = read_rpm_header(data, after_sig)
    return lzma.decompress(data[payload_offset:])


def unpack_cpio(raw: bytes, root: Path) -> list[str]:
    """解 newc cpio。返回解出的文件名列表。"""
    root = root.resolve()
    pos, names = 0, []
    while pos < len(raw):
        if raw[pos : pos + 6] != CPIO_MAGIC:
            break
        fields = [int(raw[pos + 6 + i * 8 : pos + 14 + i * 8], 16) for i in range(13)]
        mode, filesize, namesize = fields[1], fields[6], fields[11]
        name = raw[pos + 110 : pos + 110 + namesize - 1].decode()
        pos = (pos + 110 + namesize + 3) & ~3
        body = raw[pos : pos + filesize]
        pos = (pos + filesize + 3) & ~3

        if name == "TRAILER!!!":
            break

        # 路径消毒：归档里是 ./usr/... 相对路径，但仍要挡住 ../ 逃逸和符号链接
        # 指向 root 外——解包的是外部下载的归档，不能假设内容可信。
        dest = (root / name.lstrip("./")).resolve()
        if not str(dest).startswith(str(root) + os.sep):
            print(f"  跳过越界路径: {name}", file=sys.stderr)
            continue

        file_type = mode & 0o170000
        if file_type == 0o040000:
            dest.mkdir(parents=True, exist_ok=True)
        elif file_type == 0o120000:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.is_symlink() or dest.exists():
                dest.unlink()
            dest.symlink_to(body.decode())
            names.append(name)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(body)
            dest.chmod(mode & 0o777)
            names.append(name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help=f"解包目标目录（默认 {DEFAULT_DEST}）")
    parser.add_argument("--keep-rpm", action="store_true", help="保留下载的 rpm")
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    rpm_path = args.dest / RPM

    if rpm_path.exists():
        print(f"已存在，跳过下载: {rpm_path}")
    else:
        url = f"{REPO}/{RPM}"
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, rpm_path)  # noqa: S310 — 常量 https 地址
        print(f"  {rpm_path.stat().st_size / 1048576:.1f} MiB")

    print("解 RPM payload（xz + newc cpio）")
    names = unpack_cpio(extract_payload(rpm_path), args.dest / "root")

    compat = args.dest / "root/usr/local/cuda-13.0/compat"
    libcuda = compat / "libcuda.so.1"
    if not libcuda.exists():
        raise SystemExit(f"解包后找不到 {libcuda}，包内容可能变了")

    print(f"\n解出 {len(names)} 项，compat 目录：{compat}")
    for entry in sorted(compat.iterdir()):
        if entry.is_symlink():
            print(f"  {entry.name} -> {os.readlink(entry)}")

    if not args.keep_rpm:
        rpm_path.unlink()

    print("\n生效方式：. scripts/cuda_env.sh")
    print("（serve_model.sh 已经 source 了它；训练入口也要 source）")


if __name__ == "__main__":
    main()
