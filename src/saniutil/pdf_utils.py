"""PDF 与图像之间的提取 / 回写工具。"""

from __future__ import annotations

import io

import pymupdf
from PIL import Image

# 原始照片较大的 jpeg 重压缩体积收益明显，质量 85 肉眼无差异
JPEG_QUALITY = 85


def extract_page_image(doc: pymupdf.Document, page: pymupdf.Page, xref: int) -> Image.Image:
    """提取页面中某张嵌入图片，并按页面旋转角转正。"""
    info = doc.extract_image(xref)
    img = Image.open(io.BytesIO(info["image"]))
    img.load()
    rotation = page.rotation % 360
    if rotation == 90:
        # 页面内容需顺时针转 90 度显示，说明图片本身逆时针偏了，转回来
        img = img.transpose(Image.ROTATE_270)
    elif rotation == 180:
        img = img.transpose(Image.ROTATE_180)
    elif rotation == 270:
        img = img.transpose(Image.ROTATE_90)
    return img


def to_stored_orientation(img: Image.Image, rotation: int) -> Image.Image:
    """把转正后的图片旋转回页面中的存储方向。

    提取时按页面旋转角转正（见 :func:`extract_page_image`），涂黑后回写前
    必须转回原方向，否则带 rotation 的页面显示会横竖颠倒。
    """
    rotation %= 360
    if rotation == 90:
        return img.transpose(Image.ROTATE_90)
    if rotation == 180:
        return img.transpose(Image.ROTATE_180)
    if rotation == 270:
        return img.transpose(Image.ROTATE_270)
    return img


def replace_page_image(page: pymupdf.Page, xref: int, img: Image.Image) -> None:
    """用处理后的图片原位替换页面中的嵌入图片，保持摆放不变。"""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    page.replace_image(xref, stream=buf.getvalue())
