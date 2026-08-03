#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 环境检测脚本
======================
功能：检查当前 conda 环境中的 Python 版本和依赖包安装情况。
运行方式：
    python check_env.py

该脚本会：
1. 检测 Python 版本是否满足要求 (>=3.9)
2. 检测当前是否在 conda 环境 chess_ai 中
3. 逐一检查项目所需的核心依赖是否已安装
4. 输出详细的检查报告，标明缺失的包及其安装命令
"""

import sys
import subprocess
import importlib.metadata
from typing import Dict, List, Tuple


# ============================================================================
# 依赖列表定义
# ============================================================================

# 每个依赖项：(包名, 导入名, 是否必需, 用途说明)
# 导入名可能与 pip 包名不同（如 PyQt5 导入名是 PyQt5，scikit-learn 导入名是 sklearn）
REQUIRED_PACKAGES: List[Tuple[str, str, bool, str]] = [
    # ---------- 核心框架 ----------
    ("numpy", "numpy", True, "数值计算基础库"),
    ("torch", "torch", True, "PyTorch 深度学习框架（训练必需）"),
    # ---------- 国际象棋规则 ----------
    ("python-chess", "chess", True, "国际象棋规则引擎、FEN/棋盘表示"),
    # ---------- 图形界面 ----------
    ("PyQt5", "PyQt5", False, "本地 GUI 对弈窗口（推荐）"),
    # ---------- 视觉识别 ----------
    ("Pillow", "PIL", False, "图像处理（截图分析、棋子识别预处理）"),
    ("opencv-python", "cv2", False, "OpenCV 计算机视觉（棋盘检测、角点提取）"),
    ("scikit-learn", "sklearn", False, "机器学习工具（可选，用于棋子分类器）"),
    # ---------- 网页交互 ----------
    ("pyautogui", "pyautogui", False, "屏幕截图与鼠标控制（网页对战）"),
    # ---------- 辅助工具 ----------
    ("tqdm", "tqdm", True, "进度条显示（训练循环使用）"),
    ("pyyaml", "yaml", False, "YAML 配置文件解析（可选，默认使用 config.py）"),
]


def get_python_version() -> str:
    """获取当前 Python 版本字符串。"""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_python_version() -> bool:
    """检查 Python 版本是否 >= 3.9。"""
    if sys.version_info >= (3, 9):
        return True
    return False


def get_conda_env() -> str:
    """
    检测当前 conda 环境名称。
    通过读取环境变量 CONDA_DEFAULT_ENV 获取。
    如果未使用 conda，尝试通过 conda info 检测。
    """
    import os
    env_name = os.environ.get("CONDA_DEFAULT_ENV", None)
    if env_name:
        return env_name
    
    # 尝试通过 conda info 获取
    try:
        result = subprocess.run(
            ["conda", "info", "--json"],
            capture_output=True, text=True, timeout=10
        )
        import json
        info = json.loads(result.stdout)
        env_name = info.get("active_prefix_name", None)
        if env_name:
            return env_name
    except Exception:
        pass
    
    return "未知 (未检测到 conda 环境)"


def check_package(import_name: str, pip_name: str) -> Tuple[bool, str]:
    """
    检查单个 Python 包是否已安装。
    
    Args:
        import_name: 导入时使用的名称（如 "torch"）。
        pip_name: pip 安装时使用的名称（如 "torch"）。
    
    Returns:
        (是否已安装, 版本字符串或错误信息)
    """
    try:
        version = importlib.metadata.version(pip_name)
        return True, version
    except importlib.metadata.PackageNotFoundError:
        # 尝试用导入名查找
        try:
            version = importlib.metadata.version(import_name)
            return True, version
        except importlib.metadata.PackageNotFoundError:
            return False, "未安装"


def check_cuda_available() -> Tuple[bool, str]:
    """检查 PyTorch 是否支持 CUDA（GPU 加速）。"""
    try:
        import torch
        if torch.cuda.is_available():
            return True, f"CUDA {torch.version.cuda} (GPU: {torch.cuda.get_device_name(0)})"
        else:
            return False, "CUDA 不可用 (将使用 CPU 训练)"
    except ImportError:
        return False, "PyTorch 未安装"


def print_header(title: str):
    """打印带分隔线的标题。"""
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def run_check() -> int:
    """
    执行完整的环境检查。
    
    Returns:
        0 表示所有必需依赖已满足，1 表示有问题需要解决。
    """
    issues_found = 0
    
    # ---- 1. 基本信息 ----
    print_header("ChessMate 环境检测报告")
    print(f"  Python 版本 : {get_python_version()}")
    print(f"  Conda 环境   : {get_conda_env()}")
    print(f"  工作目录     : {__import__('os').getcwd()}")
    
    # ---- 2. Python 版本检查 ----
    print_header("1) Python 版本检查")
    if check_python_version():
        print("  ✅ Python 版本满足要求 (>=3.9)")
    else:
        print("  ❌ Python 版本过低！需要 >=3.9，请升级 Python。")
        issues_found += 1
    
    # ---- 3. Conda 环境检查 ----
    print_header("2) Conda 环境检查")
    conda_env = get_conda_env()
    if conda_env == "chess_ai":
        print(f"  ✅ 当前在目标 conda 环境：{conda_env}")
    elif conda_env == "未知 (未检测到 conda 环境)":
        print("  ⚠️  未检测到 conda 环境。建议使用 conda 环境 chess_ai 运行。")
    else:
        print(f"  ⚠️  当前 conda 环境为 '{conda_env}'，建议切换到 'chess_ai'。")
        print(f"     运行：conda activate chess_ai")
    
    # ---- 4. 核心依赖检查 ----
    print_header("3) 核心依赖检查 (必需)")
    for pip_name, import_name, required, description in REQUIRED_PACKAGES:
        if not required:
            continue
        installed, version = check_package(import_name, pip_name)
        if installed:
            print(f"  ✅ {pip_name:25s} {version:15s} -- {description}")
        else:
            print(f"  ❌ {pip_name:25s} {'未安装':15s} -- {description} [必需]")
            issues_found += 1
    
    # ---- 5. 可选依赖检查 ----
    print_header("4) 可选依赖检查")
    missing_optional: List[str] = []
    for pip_name, import_name, required, description in REQUIRED_PACKAGES:
        if required:
            continue
        installed, version = check_package(import_name, pip_name)
        if installed:
            print(f"  ✅ {pip_name:25s} {version:15s} -- {description}")
        else:
            print(f"  ⬜ {pip_name:25s} {'未安装':15s} -- {description} [可选]")
            missing_optional.append(pip_name)
    
    # ---- 6. CUDA 检查 ----
    print_header("5) GPU / CUDA 检查")
    cuda_ok, cuda_msg = check_cuda_available()
    if cuda_ok:
        print(f"  ✅ {cuda_msg}")
    else:
        print(f"  ⚠️  {cuda_msg}")
        print("     训练可继续使用 CPU，但速度较慢。")
    
    # ---- 7. 汇总建议 ----
    print_header("6) 汇总与建议")
    if issues_found == 0:
        print("  ✅ 所有必需依赖已安装，环境就绪！")
        if missing_optional:
            print(f"\n  可选安装以下包以启用全部功能：")
            for pkg in missing_optional:
                desc = next((d for n, _, _, d in REQUIRED_PACKAGES if n == pkg), "")
                print(f"    pip install {pkg}    # {desc}")
    else:
        print(f"  ❌ 发现 {issues_found} 个问题需要解决。")
        print("\n  请使用以下命令安装缺失的必需依赖：")
        print("    pip install -r requirements.txt")
        print("\n  或单独安装缺失的包。")
    
    # ---- 8. 可选安装命令 ----
    if missing_optional:
        print("\n  要启用全部功能，可运行以下命令安装可选依赖：")
        print(f"    pip install {' '.join(missing_optional)}")
    
    print("\n" + "=" * 65)
    
    return 0 if issues_found == 0 else 1


def main():
    """脚本入口函数。"""
    exit_code = run_check()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()