"""把隐私 bbox 涂黑。"""

from __future__ import annotations

from PIL import Image, ImageDraw


def redact_boxes(img: Image.Image,
                 boxes: list[dict]) -> Image.Image:
    """在图片上用黑色矩形覆盖所有 bbox（[x0, y0, x1, y1]，像素坐标）。"""
    draw = ImageDraw.Draw(img)
    for box in boxes:
        x0, y0, x1, y1 = box["bbox"]
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    return img
