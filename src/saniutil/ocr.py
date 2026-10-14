"""OCR 识别与隐私信息（姓名 / 住院号 / 床号 / 报告医生签名）定位。"""

import re
from typing import Any

import numpy as np
from paddleocr import PaddleOCR

# 需要遮蔽的字段标签
PRIVACY_KEYWORDS = ("住院号", "姓名", "床号")

# 报告医生签名：手写体，常与标签不同行甚至未被识别，需特殊处理
DOCTOR_KEYWORD = "报告医生"

# OCR 可能把“姓名”识别成“姓 名”（中间带空白），匹配时允许字间空白
def _keyword_pattern(kw: str) -> re.Pattern[str]:
    return re.compile(r"\s*".join(map(re.escape, kw)))

# 值的边界：遇到这些标签词（或空白）就认为值结束，防止整行误涂
_STOP_WORDS = (
    "性别", "年龄", "科室", "诊断", "心率", "心室率", "电轴", "报告",
    "P-R", "QRS", "QT", "SV1", "RV5", "PR", "住院号", "姓名", "床号",
)
_SEPARATORS = "：:=＝　 \t"
_PUNCT = "，,。;；、"

# 引擎实例（进程内惰性加载一份）
_ocr: PaddleOCR | None = None


def detect_device() -> str:
    """探测可用推理设备：装的是 CUDA 版 paddle 且有 GPU 则 "gpu"，否则 "cpu"。

    医院机器硬件各异：装了 GPU 版 paddle 则自动用 GPU；没有 GPU 或安装的
    是 CPU 版 paddle 时自动回退 CPU 推理，代码无需任何改动。
    """
    try:
        import paddle

        if paddle.device.is_compiled_with_cuda() \
                and paddle.device.cuda.device_count() > 0:
            return "gpu"
    except Exception:  # paddle 探测失败一律回退 CPU
        pass
    return "cpu"


def get_ocr() -> PaddleOCR:
    """懒加载 OCR 引擎（首次调用会下载/加载模型），设备自动选择。"""
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="ch",
            device=detect_device(),
            # paddle 3.3 的 oneDNN 对 PP-OCRv6 模型有未实现的算子属性转换
            # bug，CPU 模式必须关闭（对 GPU 无影响）
            enable_mkldnn=False,
        )
    return _ocr


def ocr_image(img: str | np.ndarray) -> dict[str, Any]:
    """对单张图片做 OCR，返回 texts / scores / polys（各自为等长列表）。

    ``img`` 为图片路径或 numpy 数组；数组按 OpenCV 惯例为 **BGR** 通道
    （paddlex 的 ReadImage 对 ndarray 输入不做转换），RGB 的 PIL 图片请先
    用 ``np.ascontiguousarray(np.asarray(img)[:, :, ::-1])`` 转换。
    """
    result = get_ocr().predict(img)
    r = result[0]
    return {
        "texts": list(r["rec_texts"]),
        "scores": [float(s) for s in r["rec_scores"]],
        "polys": [[[float(x), float(y)] for x, y in poly] for poly in r["rec_polys"]],
    }


def _char_width(ch: str) -> float:
    """估算字符渲染宽度：CJK 全角记 1，其余（数字/字母/符号）记 0.55。"""
    return 1.0 if ord(ch) > 0x2E80 else 0.55


def _value_span(text: str, kw: str) -> tuple[int, int]:
    """定位行内关键字之后的“值”的字符区间 [start, end)。"""
    m = _keyword_pattern(kw).search(text)
    i = m.end() if m else -1
    if i < 0:
        return 0, 0
    while i < len(text) and text[i] in _SEPARATORS:
        i += 1
    start = i
    while i < len(text):
        if text[i] in " 　\t":
            break
        if any(text.startswith(w, i) for w in _STOP_WORDS):
            break
        i += 1
    # 去掉值末尾的标点
    while i > start and text[i - 1] in _PUNCT:
        i -= 1
    return start, i


def _quad_bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _pad(bbox: tuple[float, float, float, float], pad: float,
         size: tuple[int, int]) -> tuple[float, float, float, float]:
    return (
        max(0.0, bbox[0] - pad),
        max(0.0, bbox[1] - pad),
        min(float(size[0]), bbox[2] + pad),
        min(float(size[1]), bbox[3] + pad),
    )


def _doctor_box(text: str, poly: list[list[float]],
                img_size: tuple[int, int]) -> dict[str, Any]:
    """报告医生签名框：从标签（或行内值）起一直涂到页面右边距。

    签名是手写体，常位于标签右侧且往往未被识别，涂到右边距最稳妥；
    纵向多留 1.2 倍行高，覆盖笔画上下出界的情况。
    """
    w, _ = img_size
    x0, y0, x1, y1 = _quad_bbox(poly)
    m = _keyword_pattern(DOCTOR_KEYWORD).search(text)
    start, end = _value_span(text, DOCTOR_KEYWORD)
    units = [_char_width(c) for c in text]
    total = sum(units)
    line_w = x1 - x0
    # 有行内值从值起涂；没有则从标签开始涂（避免同行的报告日期等被连带）
    anchor = start if end > start else m.start()
    left = x0 + sum(units[:anchor]) / total * line_w
    pad = (y1 - y0) * 1.2
    bbox = _pad((left, y0, float(w), y1), pad=pad, size=img_size)
    return {"label": DOCTOR_KEYWORD, "value": text[start:end],
            "bbox": list(bbox)}


def _signature_region_boxes(ocr_res: dict[str, Any],
                            img_size: tuple[int, int]) -> list[dict[str, Any]]:
    """兜底：整页没找到“报告医生”标签时，涂掉页面右下角区域内的文本行。

    该区域在本报告版式中只包含医生签名（报告日期在中间偏左，不会落入）。
    """
    w, h = img_size
    boxes = []
    for text, poly in zip(ocr_res["texts"], ocr_res["polys"]):
        x0, y0, x1, y1 = _quad_bbox(poly)
        if x0 > w * 0.6 and y0 > h * 0.75:
            boxes.append({
                "label": DOCTOR_KEYWORD,
                "value": text,
                "bbox": list(_pad((x0, y0, x1, y1),
                                  pad=max(4.0, (y1 - y0) * 0.8), size=img_size)),
            })
    return boxes


def find_privacy_boxes(ocr_res: dict[str, Any],
                       img_size: tuple[int, int]) -> list[dict[str, Any]]:
    """从 OCR 结果中找出隐私信息的 bbox（图像像素坐标）。

    覆盖姓名 / 住院号 / 床号的值，以及右下角的报告医生签名。返回
    [{"label": ..., "value": ..., "bbox": [x0, y0, x1, y1]}]，已按行内比例
    定位值的部分，避免同一行内其他信息（如心率、性别）被连带涂黑。
    """
    boxes: list[dict[str, Any]] = []
    has_doctor_label = False
    for text, poly in zip(ocr_res["texts"], ocr_res["polys"]):
        if _keyword_pattern(DOCTOR_KEYWORD).search(text):
            has_doctor_label = True
            boxes.append(_doctor_box(text, poly, img_size))
            continue
        for kw in PRIVACY_KEYWORDS:
            if not _keyword_pattern(kw).search(text):
                continue
            x0, y0, x1, y1 = _quad_bbox(poly)
            start, end = _value_span(text, kw)
            if end > start:
                # 按字符宽度比例把字符区间映射到行的横向范围
                units = [_char_width(c) for c in text]
                total = sum(units)
                line_w = x1 - x0
                left = sum(units[:start]) / total * line_w
                right = sum(units[:end]) / total * line_w
                pad = _char_width(text[start]) * line_w / total if end > start else 0
                bbox = (x0 + left, y0, x0 + right, y1)
            else:
                bbox = (x0, y0, x1, y1)
            value = text[start:end]
            boxes.append({
                "label": kw,
                "value": value,
                "bbox": list(_pad(bbox, pad=max(2.0, (y1 - y0) * 0.15), size=img_size)),
            })
    if not has_doctor_label:
        boxes.extend(_signature_region_boxes(ocr_res, img_size))
    return boxes


def draw_ocr_result(img, ocr_res: dict[str, Any],
                    privacy_boxes: list[dict[str, Any]]):
    """在原图上绘制 OCR 效果图：绿框为识别出的每行文字，红框为隐私项。"""
    from PIL import ImageDraw
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    for poly in ocr_res["polys"]:
        pts = [(x, y) for x, y in poly]
        draw.polygon(pts, outline=(0, 180, 0), width=3)
    for box in privacy_boxes:
        x0, y0, x1, y1 = box["bbox"]
        draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=5)
        draw.text((x0, max(0, y0 - 30)), f"{box['label']}", fill=(255, 0, 0))
    return vis


def page_record(page_no: int, ocr_res: dict[str, Any],
                privacy_boxes: list[dict[str, Any]]) -> dict[str, Any]:
    """组装单页保存到 JSON 的记录。"""
    return {
        "page": page_no,
        "texts": ocr_res["texts"],
        "scores": ocr_res["scores"],
        "polys": ocr_res["polys"],
        "privacy": privacy_boxes,
    }
