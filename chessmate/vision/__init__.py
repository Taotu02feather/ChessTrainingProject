#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 视觉识别模块
======================
通过屏幕截图检测棋盘、识别棋子，输出 FEN 字符串。
支持多种识别策略，可从简单场景逐步升级到复杂场景。
"""

from chessmate.vision.board_detector import BoardDetector
from chessmate.vision.piece_classifier import PieceClassifier