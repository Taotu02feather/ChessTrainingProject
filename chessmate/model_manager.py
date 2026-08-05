#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 模型管理器
====================
统一管理所有模型加载逻辑，为 GUI、网页对战、训练恢复提供一致的接口。

功能：
1. 扫描 models/ 目录，列出所有可用模型文件
2. 交互式选择模型（让用户选择加载哪个）
3. 安全加载模型（处理版本不匹配等异常）
4. 训练检查点自动检测与恢复
"""

import sys
import os
import glob
import logging

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import torch

logger = logging.getLogger("ModelManager")


# ============================================================================
# 模型文件扫描
# ============================================================================

def list_available_models(model_dir: str) -> List[Dict[str, str]]:
    """
    扫描模型目录，返回所有 .pth 文件的元信息列表。

    Args:
        model_dir: 模型目录路径。

    Returns:
        模型信息列表，每个元素包含：
        - filename: 文件名
        - path: 完整路径
        - size_mb: 文件大小 (MB)
        - modified: 最后修改时间字符串
    """
    if not os.path.isdir(model_dir):
        return []

    models = []
    pattern = os.path.join(model_dir, "*.pth")
    for filepath in sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True):
        filename = os.path.basename(filepath)
        size_bytes = os.path.getsize(filepath)
        mtime = os.path.getmtime(filepath)

        models.append({
            "filename": filename,
            "path": filepath,
            "size_mb": round(size_bytes / (1024 * 1024), 1),
            "modified": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
        })

    return models


def _sanitize_config_value(value):
    """
    修复被错误序列化的配置值。
    
    旧版 to_dict() 将整数错误地通过 logging.getLevelName() 转换，
    导致如 board_size=8 被保存为 "Level 8"。
    此函数将这类值还原为整数。
    """
    if isinstance(value, str) and value.startswith("Level "):
        try:
            return int(value.split(" ", 1)[1])
        except (ValueError, IndexError):
            pass
    # 处理 "Level True" / "Level False"
    if isinstance(value, str) and value == "Level True":
        return True
    if isinstance(value, str) and value == "Level False":
        return False
    return value


def is_checkpoint(filepath: str) -> bool:
    """
    判断模型文件是否为完整检查点（包含优化器状态）。
    检查点通常比纯模型文件大（包含 optimizer_state_dict）。

    Args:
        filepath: 模型文件路径。

    Returns:
        True 如果是完整检查点。
    """
    try:
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        # 检查点包含 optimizer_state_dict 和 iteration
        has_optimizer = "optimizer_state_dict" in checkpoint
        has_iteration = "iteration" in checkpoint
        return has_optimizer and has_iteration
    except Exception:
        return False


# ============================================================================
# 模型选择（交互式）
# ============================================================================

def select_model(config, auto_select: bool = False) -> Optional[any]:
    """
    交互式选择要加载的模型。

    列出 models/ 目录下的所有模型文件，让用户选择。
    支持自动选择默认模型（非交互模式）。

    Args:
        config: ChessConfig 配置对象。
        auto_select: 如果为 True，自动选择 best_model.pth（不询问用户）。

    Returns:
        加载好的 ChessNet 模型，或 None（用户选择不使用模型）。
    """
    from chessmate.training.neural_net import ChessNet

    model_dir = config.model_dir
    models = list_available_models(model_dir)

    if not models:
        logger.info("models/ 目录为空，将使用随机初始化模型")
        model = ChessNet(config)
        model.eval()
        return model

    # 自动选择模式
    if auto_select:
        # 优先选择 best_model.pth
        for m in models:
            if m["filename"] == config.best_model_name:
                return _load_model_from_file(config, m["path"])
        # 否则选择最新的
        return _load_model_from_file(config, models[0]["path"])

    # 交互式选择
    print("\n" + "=" * 55)
    print("  📁 发现以下模型文件（models/）:")
    print("=" * 55)
    for i, m in enumerate(models):
        checkpoint_tag = " [含训练状态]" if is_checkpoint(m["path"]) else ""
        print(f"  [{i + 1}] {m['filename']:<25s} {m['size_mb']:>5.1f} MB  {m['modified']}{checkpoint_tag}")

    print(f"  [0] 不使用模型（随机初始化）")
    print("=" * 55)

    while True:
        try:
            choice = input("请选择要加载的模型 [1]: ").strip()
            if choice == "":
                choice = "1"  # 默认选第一个

            if choice == "0":
                model = ChessNet(config)
                model.eval()
                logger.info("使用随机初始化模型")
                return model

            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"正在加载: {selected['filename']} ...")
                return _load_model_from_file(config, selected["path"])
            else:
                print(f"无效选项，请输入 0~{len(models)}")
        except (ValueError, KeyboardInterrupt, EOFError):
            # 默认选第一个
            logger.info("输入无效，默认使用第一个模型")
            return _load_model_from_file(config, models[0]["path"])


def _load_model_from_file(config, filepath: str) -> any:
    """
    安全加载模型文件。

    自动检测是纯模型文件还是完整检查点，并正确处理。

    Args:
        config: ChessConfig 配置。
        filepath: 模型文件路径。

    Returns:
        加载好的 ChessNet 模型。
    """
    from chessmate.training.neural_net import ChessNet

    try:
        # 先尝试作为完整检查点加载（含 optimizer 状态）
        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)

        # 如果是完整检查点，提取 model_state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # 从检查点中尝试获取配置
            saved_config = checkpoint.get("config", {})
            iteration = checkpoint.get("iteration", "?")

            # 用保存的配置更新当前 config（同时修复被损毁的值）
            for k, v in saved_config.items():
                if hasattr(config, k) and k not in (
                    "project_root", "model_dir", "log_dir",
                ):
                    setattr(config, k, _sanitize_config_value(v))

            model = ChessNet(config)
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            logger.info(f"模型已加载: {os.path.basename(filepath)} (第 {iteration} 轮训练)")
            print(f"  ✅ 模型已加载（第 {iteration} 轮训练）")
        else:
            # 纯模型权重文件
            model = ChessNet.load(filepath, config=config, device="cpu")
            logger.info(f"模型已加载: {os.path.basename(filepath)}")
            print(f"  ✅ 模型已加载（纯权重文件）")

        return model

    except Exception as e:
        logger.error(f"加载模型失败 ({filepath}): {e}")
        print(f"  ⚠️ 加载失败: {e}")
        print("  将使用随机初始化模型")
        from chessmate.training.neural_net import ChessNet
        model = ChessNet(config)
        model.eval()
        return model


# ============================================================================
# 训练恢复
# ============================================================================

def check_and_restore_training(trainer) -> bool:
    """
    检查是否存在可恢复的训练检查点，询问用户是否恢复。

    Args:
        trainer: Trainer 实例（已初始化但尚未开始训练）。

    Returns:
        True 如果成功恢复了训练状态。
    """
    model_dir = trainer.config.model_dir
    latest_path = os.path.join(model_dir, trainer.config.latest_model_name)

    if not os.path.exists(latest_path):
        return False

    # 检查是否为完整检查点
    if not is_checkpoint(latest_path):
        return False

    try:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        iteration = checkpoint.get("iteration", "?")
        best_loss = checkpoint.get("best_loss", "?")

        print(f"\n📁 发现训练检查点: latest_model.pth")
        print(f"   训练轮次: 第 {iteration} 轮")
        if isinstance(best_loss, float):
            print(f"   最佳损失: {best_loss:.6f}")

        choice = input("\n是否从此检查点恢复训练？[Y/n]: ").strip().lower()
        if choice == "n":
            print("将从头开始训练。")
            return False

        trainer.load_checkpoint(latest_path)
        print(f"✅ 已恢复第 {iteration} 轮训练状态")
        return True

    except Exception as e:
        logger.error(f"恢复检查点失败: {e}")
        return False


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from chessmate.config import ChessConfig

    cfg = ChessConfig()
    print("模型管理模块测试")
    print(f"模型目录: {cfg.model_dir}")

    models = list_available_models(cfg.model_dir)
    if models:
        print(f"\n找到 {len(models)} 个模型文件:")
        for m in models:
            cp = "检查点" if is_checkpoint(m["path"]) else "纯权重"
            print(f"  {m['filename']} ({m['size_mb']} MB) [{cp}] {m['modified']}")
    else:
        print("  没有找到模型文件。")