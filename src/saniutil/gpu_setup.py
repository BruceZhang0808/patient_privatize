"""首次运行时自动配置 GPU 推理环境。

paddlepaddle（CPU）与 paddlepaddle-gpu（按 CUDA 版本分多个构建、位于官方
私有源）是互相覆盖的两个包，pip 的依赖元数据无法表达"有 GPU 就装 GPU 版"，
因此放在**首次运行时**处理：检测到 NVIDIA GPU 但当前装的是 CPU 版 paddle
时，询问用户并自动替换为与驱动匹配的 GPU 版，之后程序自动使用 GPU 推理；
没有 GPU 的机器不做任何事，直接用 CPU。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from importlib.util import find_spec

# 本包适配的 paddlepaddle-gpu 版本及其官方源提供的 CUDA 构建列表（官方仅有
# cu118 / cu126 / cu129 / cu130，见 https://www.paddlepaddle.org.cn/install/quick）
PADDLE_GPU_VERSION = "3.3.0"
_CUDA_BUILDS = (118, 126, 129, 130)
_CUDA_DEFAULT = 126
_PADDLE_INDEX = "https://www.paddlepaddle.org.cn/packages/stable/cu{build}/"


def _nvidia_smi() -> str | None:
    """返回 nvidia-smi 的输出；无 NVIDIA 驱动/机器上不可用时返回 None。"""
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             timeout=10)
        return out.stdout if out.returncode == 0 else None
    except Exception:
        return None


def detect_nvidia() -> tuple[bool, str | None]:
    """检测 (是否有 NVIDIA GPU, 驱动支持的 CUDA 版本字符串，如 "12.9")。"""
    smi = _nvidia_smi()
    if smi is None:
        return False, None
    m = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi)
    return True, (m.group(1) if m else None)


def pick_cuda_build(driver_cuda: str | None) -> int:
    """根据驱动支持的 CUDA 版本挑选可用的 paddle 构建版本号（如 126）。"""
    if not driver_cuda:
        return _CUDA_DEFAULT
    major_minor = round(float(driver_cuda) * 10)  # "12.9" -> 129
    for build in sorted(_CUDA_BUILDS, reverse=True):
        if build <= major_minor:
            return build
    return _CUDA_BUILDS[0]


def _installer_cmd() -> list[str] | None:
    """找到可用的 pip 安装器命令（uv 管理的 venv 里可能没有 pip 模块）。"""
    if shutil.which("uv"):
        return ["uv", "pip"]
    if find_spec("pip"):
        return [sys.executable, "-m", "pip"]
    return None


def _run_installer(cmd: list[str], *args: str) -> bool:
    if cmd == ["uv", "pip"]:
        # uv 要求 --python 等选项放在子命令之后，显式指定目标解释器环境
        full = [*cmd, args[0], "--python", sys.executable, *args[1:]]
    else:
        full = [*cmd, *args]
    print("$", " ".join(full))
    try:
        r = subprocess.run(full, check=False)
        return r.returncode == 0
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] 安装命令执行失败: {exc}")
        return False


def ensure_gpu_paddle(interactive: bool | None = None) -> None:
    """GPU 环境自动配置入口，在打印推理设备前调用。

    - 已是 GPU 版 paddle → 什么都不做；
    - 无 NVIDIA GPU → 什么都不做（CPU 推理）；
    - 有 NVIDIA GPU 且装的是 CPU 版 → 询问用户，同意则自动替换安装并退出
      （paddle 已加载进当前进程，需重新运行才能生效）。
    """
    from .ocr import detect_device

    if detect_device() == "gpu":
        return
    has_gpu, driver_cuda = detect_nvidia()
    if not has_gpu:
        return

    build = pick_cuda_build(driver_cuda)
    print(f"[info] 检测到 NVIDIA GPU（驱动支持 CUDA {driver_cuda or '未知'}），"
          f"但当前安装的是 CPU 版 paddlepaddle。")
    if interactive is None:
        interactive = sys.stdin.isatty()
    env_flag = os.environ.get("PATIENT_PRIVATIZE_AUTO_GPU", "").lower()
    if env_flag == "0":
        return
    if interactive and env_flag != "1":
        try:
            answer = input(
                f"是否自动下载并安装 GPU 版 paddlepaddle=={PADDLE_GPU_VERSION}"
                f"（CUDA cu{build}，约 2GB）？[y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""
        if answer not in ("y", "yes"):
            print("[info] 跳过，本次使用 CPU 推理。")
            return
    elif env_flag != "1":
        # 非交互且未显式开启：不擅自安装，给出手动命令
        print("[info] 非交互环境，跳过 GPU 版安装（仍用 CPU 推理）。"
              "如需 GPU 加速，请手动执行下述安装命令后重新运行，"
              "或设置环境变量 PATIENT_PRIVATIZE_AUTO_GPU=1 后重新运行。")
        print(f"[info]   pip uninstall -y paddlepaddle && "
              f"pip install paddlepaddle-gpu=={PADDLE_GPU_VERSION} "
              f"--extra-index-url {_PADDLE_INDEX.format(build=build)}")
        return

    cmd = _installer_cmd()
    if cmd is None:
        print("[warn] 未找到 pip/uv 安装器，无法自动安装，请手动执行：")
        print(f"[info]   pip uninstall -y paddlepaddle && "
              f"pip install paddlepaddle-gpu=={PADDLE_GPU_VERSION} "
              f"--extra-index-url {_PADDLE_INDEX.format(build=build)}")
        return

    print("[info] 正在卸载 CPU 版 paddlepaddle ...")
    if not _run_installer(cmd, "uninstall", "-y", "paddlepaddle"):
        print("[warn] 卸载失败，请手动执行上述命令；本次继续用 CPU 推理。")
        return
    print(f"[info] 正在安装 GPU 版 paddlepaddle-gpu=={PADDLE_GPU_VERSION} "
          f"(cu{build})，视网速需要几分钟 ...")
    if _run_installer(cmd, "install",
                      f"paddlepaddle-gpu=={PADDLE_GPU_VERSION}",
                      f"--index-url={_PADDLE_INDEX.format(build=build)}"):
        print("[info] GPU 版安装完成，请重新运行本命令以启用 GPU 推理。")
        sys.exit(0)
    print("[warn] GPU 版安装失败，请按上方提示手动安装；本次继续用 CPU 推理。")
