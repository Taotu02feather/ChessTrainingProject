#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 国际象棋 AI 系统
==========================
一个完整的国际象棋 AI 系统，包含三大核心功能：
1. 自行训练强化学习模型（基于 AlphaZero 类算法）
2. 本地图形界面（GUI）进行人机对弈
3. 通过屏幕截图识别网页棋局并自动走子
"""

__version__ = "0.1.0"
__author__ = "ChessMate Team"
__license__ = "MIT"

# 便捷导入
from chessmate.config import ChessConfig