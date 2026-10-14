"""Sanitize util (saniutil): 遮蔽心电图 PDF 中的患者隐私信息（姓名/住院号/床号/医生签名）。

用法（安装后直接执行）::

    saniutil data/报告.pdf -o outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import pipeline
from .ocr import detect_device
from .gpu_setup import ensure_gpu_paddle


def run(argv: list[str] | None = None) -> None:
    """命令行入口：输入心电图 PDF（每页一张图片），输出脱敏后的 PDF。"""
    parser = argparse.ArgumentParser(
        prog="saniutil",
        description="遮蔽心电图 PDF 中患者姓名、住院号、床号及报告医生签名",
    )
    parser.add_argument("src", type=Path,
                        help="心电图 PDF 路径（每页一张嵌入图片）")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("outputs"),
                        help="输出目录（默认 outputs）")
    args = parser.parse_args(argv)

    if not args.src.is_file():
        sys.exit(f"文件不存在: {args.src}")

    # 有 NVIDIA GPU 但装的是 CPU 版 paddle 时，引导用户自动切换（无 GPU 机器无感）
    ensure_gpu_paddle()
    print(f"[info] 推理设备: {detect_device()}")

    out = pipeline.run(args.src, args.output_dir)
    print(f"脱敏 PDF 已保存: {out}")


if __name__ == "__main__":
    sys.exit(run())
