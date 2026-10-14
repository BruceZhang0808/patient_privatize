# Sanitize util (saniutil)

自动化遮蔽 12 导联心电图 PDF 中患者隐私信息（姓名、住院号、床号、报告医生签名）
的命令行工具。

## 快速上手

以下教学假设本包已发布到 PyPI，医院方在自己电脑上从零开始使用。

### 1. 安装

需要 Python ≥ 3.12，然后：

```bash
pip install saniutil
```

安装会自动带上全部依赖（CPU 版 paddlepaddle、paddleocr、pymupdf 等），
普通电脑（无独立显卡）装完即可直接使用。

PaddleOCR 识别模型在**首次运行时自动下载**（约 150MB，需要联网），之后
缓存在用户目录，离线可用。如需在内网机器上离线部署：先在一台有网的机器上
运行一次，再把 `~/.paddlex/official_models/` 目录整个拷贝到内网机器的同一路径。

### 2. （可选）有 NVIDIA 显卡的机器启用 GPU 加速

`paddlepaddle`（CPU）与 `paddlepaddle-gpu` 提供同一个 `paddle` 包、不能共存，
需要先卸载 CPU 版再安装 GPU 版（无需改任何代码，程序会自动检测并使用 GPU；
没有 GPU 的机器跳过本步即可，自动使用 CPU 推理）：

```bash
pip uninstall -y paddlepaddle
pip install paddlepaddle-gpu==3.3.0 \
    --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cu130/
```

### 3. 准备数据

把医院提供的 PDF 放到项目的 `data/` 目录下。数据形式要求：

- **一个 PDF 包含多名患者**的心电图报告；
- **每一页是一张以图片形式嵌入的完整报告照片**（扫描或拍照导出的 PDF 均可）。

例如本仓库自带的示例：`data/shandong.pdf`（23 页，每页一张心电图报告照片）。

### 4. 运行

```bash
saniutil data/shandong.pdf
```

运行时程序会：逐页 OCR 定位隐私信息 → 涂黑 → 原位回写并重新输出 PDF，
全程自动，进度条逐页显示。可选参数 `-o` 指定输出目录（默认 `outputs/`）。

### 5. 查看结果

所有输出都在输出目录（默认 `outputs/`）中：

```
outputs/
├── shandong_privatized.pdf   # ★ 最终脱敏 PDF（交给医院方的文件）
├── ocr/page_XXX.json         # 每页 OCR 原文与定位到的隐私值（便于抽查）
├── ocr_vis/page_XXX.jpg      # OCR 效果图：绿框=识别行，红框=隐私项
└── images/page_XXX.jpg       # 每页涂黑后的整页图片（便于人工复核）
```

建议翻看 `outputs/images/` 逐页确认涂黑效果，`outputs/ocr/` 可核对有无漏检。

## 工作流程

1. **OCR**：使用 PaddleOCR 逐页识别文字，定位"姓名 / 住院号 / 床号"对应的值
   及右下角"报告医生"手写签名，结果保存为 JSON；
2. **涂黑**：按 bbox 用黑色矩形覆盖隐私内容（只覆盖值的区域，同一行的性别、
   年龄、心率等其他信息不受影响）；
3. **回写**：将处理后的图片原位替换回对应页面（保持摆放位置不变）；
4. **输出**：全部页处理完成后保存脱敏 PDF 及中间产物。

## 硬件兼容性说明

医院机器硬件各异，程序**自动适配**，无需任何配置：

- 装了 CUDA 版 paddle 且有 GPU → 使用 GPU 推理；
- 没有 GPU 或装的是 CPU 版 paddle → 自动回退 CPU 推理（同一份代码）。

## 项目结构

```
src/saniutil/
├── __init__.py    # 命令行入口（saniutil 命令）
├── pdf_utils.py   # 图片提取、方向转正/还原、图片原位替换
├── ocr.py         # 设备探测（GPU/CPU 回退）、PaddleOCR 封装、隐私值定位
├── redact.py      # bbox 涂黑
└── pipeline.py    # 串起 OCR → 涂黑 → JSON 保存 → 回写 → 输出
```

## 效果验证

`outputs/ocr_vis/` 下保存了每页 OCR 效果图（绿框为识别行、红框为定位到的隐私
值），`outputs/images/` 保存了每页涂黑后的整页图片，均可直接翻看确认隐私信息
已被覆盖且未误伤其他信息；`outputs/ocr/` 中的 JSON 记录了每页识别到的原文与
隐私值，便于抽查漏检情况。
