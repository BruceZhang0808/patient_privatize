"""完整流程：纯净 PDF -> OCR 定位隐私 -> 涂黑 -> 回写并输出 PDF（逐页串行）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image
from tqdm import tqdm

from .ocr import draw_ocr_result, find_privacy_boxes, ocr_image, page_record
from .pdf_utils import extract_page_image, replace_page_image, to_stored_orientation
from .redact import redact_boxes


def run(src_pdf: Path, out_dir: Path) -> Path:
    """处理心电图 PDF（每页一张嵌入图片），返回脱敏后 PDF 的路径。

    - 每页 OCR 结果保存到 ``out_dir/ocr/page_XXX.json``
    - OCR 效果图（识别行绿框、隐私项红框）保存到 ``out_dir/ocr_vis/page_XXX.jpg``
    - 带涂黑的中间图片保存到 ``out_dir/images/page_XXX.jpg``
    - 最终 PDF 保存为 ``out_dir/<原文件名>_privatized.pdf``
    """
    clean_pdf = Path(src_pdf)
    out_dir = Path(out_dir)
    ocr_dir = out_dir / "ocr"
    img_dir = out_dir / "images"
    vis_dir = out_dir / "ocr_vis"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(clean_pdf)
    for page_no, page in enumerate(tqdm(doc, desc="处理页面", unit="页")):
        images = page.get_images(full=True)
        if not images:
            continue
        xref = images[0][0]
        img = extract_page_image(doc, page, xref)

        # paddleocr 的 predict 直接接受 numpy 数组（BGR 通道），无需落盘
        arr = np.ascontiguousarray(np.asarray(img)[:, :, ::-1])
        ocr_res = ocr_image(arr)
        privacy = find_privacy_boxes(ocr_res, img.size)

        record = page_record(page_no, ocr_res, privacy)
        (ocr_dir / f"page_{page_no:03d}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        # 中间复核图存 JPEG：12MP 图 PNG 编码需 ~3.3s/张，JPEG 仅 ~0.04s，
        # 每页可省 6s 以上（复核图质量 90 肉眼无差异）
        draw_ocr_result(img, ocr_res, privacy).save(
            vis_dir / f"page_{page_no:03d}.jpg", quality=90
        )

        redacted = redact_boxes(img.copy(), privacy)
        redacted.save(img_dir / f"page_{page_no:03d}.jpg", quality=90)
        # 回写前转回页面中的存储方向，保证带 rotation 的页面显示方向不变
        replace_page_image(page, xref, to_stored_orientation(redacted, page.rotation))

    out_path = out_dir / f"{clean_pdf.stem}_privatized.pdf"
    doc.save(out_path, deflate=True, garbage=4)
    doc.close()
    return out_path
