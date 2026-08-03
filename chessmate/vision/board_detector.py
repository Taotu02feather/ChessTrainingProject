#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 棋盘检测器
====================
从屏幕截图中检测国际象棋棋盘的位置和边界。

主要功能：
1. 检测图片中的棋盘区域（基于角点检测或轮廓分析）
2. 将棋盘区域分割为 8x8 的方格
3. 提取每个方格的坐标中心点
4. 输出可用于棋子识别的标准化方格图像

检测策略（从简到繁）：
- 方案A（简单）：假设棋盘在截图的固定区域（用户手动配置坐标）
- 方案B（自动）：使用 OpenCV 边缘检测 + 霍夫变换检测棋盘网格
- 方案C（高级）：使用深度学习目标检测模型（如 YOLO）定位棋盘
"""

import sys
import os
import cv2
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class ChessboardRegion:
    """
    检测到的棋盘区域信息。

    属性:
        top_left: 棋盘左上角坐标 (x, y)
        bottom_right: 棋盘右下角坐标 (x, y)
        square_size: 每个方格的大小（像素）
        corners: 四个角点坐标 [(x,y), ...]，按顺序：左上、右上、右下、左下
        confidence: 检测置信度 (0~1)
    """
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    square_size: float
    corners: List[Tuple[int, int]]
    confidence: float = 1.0

    @property
    def width(self) -> int:
        return self.bottom_right[0] - self.top_left[0]

    @property
    def height(self) -> int:
        return self.bottom_right[1] - self.top_left[1]

    def get_square_center(self, row: int, col: int) -> Tuple[int, int]:
        """
        获取棋盘格 (row, col) 的中心屏幕坐标。
        row=0 对应第 8 行（黑方底线），row=7 对应第 1 行（白方底线）。
        col=0 对应 a 列，col=7 对应 h 列。

        Args:
            row: 行索引 (0-7), 0 = 棋盘顶部（黑方视角的底线）。
            col: 列索引 (0-7), 0 = 左边缘（a列）。

        Returns:
            (x, y) 屏幕坐标。
        """
        x = int(self.top_left[0] + (col + 0.5) * self.square_size)
        y = int(self.top_left[1] + (row + 0.5) * self.square_size)
        return (x, y)

    def get_square_rect(self, row: int, col: int, margin: int = 2) -> Tuple[int, int, int, int]:
        """
        获取棋盘格的边界矩形（可用于裁剪子图像进行棋子识别）。

        Args:
            row: 行索引 (0-7)。
            col: 列索引 (0-7)。
            margin: 边界缩进像素（避免边框干扰）。

        Returns:
            (x, y, width, height) 矩形。
        """
        x = int(self.top_left[0] + col * self.square_size) + margin
        y = int(self.top_left[1] + row * self.square_size) + margin
        w = int(self.square_size) - 2 * margin
        h = int(self.square_size) - 2 * margin
        return (x, y, max(w, 1), max(h, 1))


# ============================================================================
# 棋盘检测器
# ============================================================================

class BoardDetector:
    """
    国际象棋棋盘检测器。

    支持手动区域配置和自动检测两种模式。

    使用方式：
        detector = BoardDetector(config)
        region = detector.detect(image)           # 自动检测
        region = detector.detect_manual(image)    # 手动区域
    """

    def __init__(self, config):
        """
        初始化检测器。

        Args:
            config: ChessConfig 配置对象。
        """
        self.config = config
        self.square_size_min = config.vision_square_size_min
        self.square_size_max = config.vision_square_size_max

    def detect(self, image: np.ndarray) -> Optional[ChessboardRegion]:
        """
        从图像中检测棋盘。

        优先使用手动配置区域，若未配置则尝试自动检测。

        Args:
            image: BGR 格式的 numpy 图像数组。

        Returns:
            检测到的棋盘区域，或 None（检测失败）。
        """
        if self.config.vision_screenshot_region is not None:
            return self.detect_manual(image)

        return self.detect_auto(image)

    def detect_manual(self, image: np.ndarray) -> ChessboardRegion:
        """
        使用手动配置的区域检测棋盘。

        根据配置中的棋盘左上角和格子大小计算完整棋盘区域。

        Args:
            image: BGR 格式的图像（用于获取尺寸验证）。

        Returns:
            ChessboardRegion 对象。
        """
        x, y = self.config.web_board_top_left
        square_size = self.config.web_square_size
        board_pixels = square_size * 8

        corners = [
            (x, y),                          # 左上
            (x + board_pixels, y),           # 右上
            (x + board_pixels, y + board_pixels),  # 右下
            (x, y + board_pixels),           # 左下
        ]

        return ChessboardRegion(
            top_left=(x, y),
            bottom_right=(x + board_pixels, y + board_pixels),
            square_size=square_size,
            corners=corners,
            confidence=1.0,
        )

    def detect_auto(self, image: np.ndarray) -> Optional[ChessboardRegion]:
        """
        自动检测棋盘位置。

        使用 OpenCV 的边缘检测 + 轮廓查找来定位棋盘。
        注意：此方法在复杂背景下可能不准确，建议优先使用手动配置。

        Args:
            image: BGR 格式的图像。

        Returns:
            检测到的棋盘区域，或 None。
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 自适应阈值 + Canny 边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # 查找轮廓
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 查找最大的近似四边形轮廓
        best_board = None
        max_area = 0

        for contour in contours:
            # 计算轮廓面积
            area = cv2.contourArea(contour)
            if area < (w * h * 0.05):  # 至少占图像 5%
                continue

            # 多边形近似
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # 需要近似为四边形
            if len(approx) == 4 and area > max_area:
                max_area = area
                best_board = approx

        if best_board is None:
            return None

        # 整理角点顺序（左上、右上、右下、左下）
        corners = self._order_corners(best_board.reshape(4, 2))

        # 计算格子大小
        board_width = corners[1][0] - corners[0][0]
        square_size = board_width / 8.0

        # 验证格子大小是否在合理范围内
        if square_size < self.square_size_min or square_size > self.square_size_max:
            return None

        x1, y1 = corners[0]
        x2, y2 = corners[2]

        return ChessboardRegion(
            top_left=(x1, y1),
            bottom_right=(x2, y2),
            square_size=square_size,
            corners=[tuple(c) for c in corners],
            confidence=min(1.0, max_area / (w * h)),
        )

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """
        将四个角点排序为：左上、右上、右下、左下。

        Args:
            pts: (4, 2) 形状的 numpy 数组。

        Returns:
            排序后的角点。
        """
        rect = np.zeros((4, 2), dtype=np.float32)

        # 按 x+y 排序（最小值为左上，最大值为右下）
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # 左上
        rect[2] = pts[np.argmax(s)]  # 右下

        # 按 y-x 排序
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # 右上
        rect[3] = pts[np.argmax(diff)]  # 左下

        return rect

    def extract_squares(
        self, image: np.ndarray, region: ChessboardRegion
    ) -> List[List[np.ndarray]]:
        """
        从棋盘区域提取 8x8 个方格子图像。

        Args:
            image: 原始图像（BGR）。
            region: 检测到的棋盘区域。

        Returns:
            二维列表 grid[row][col]，每个元素是一个方格的 BGR 子图像。
            row=0 对应棋盘顶部（黑方底线）。
        """
        grid = []
        for row in range(8):
            row_images = []
            for col in range(8):
                x, y, w, h = region.get_square_rect(row, col)
                x, y = max(0, x), max(0, y)
                w, h = min(w, image.shape[1] - x), min(h, image.shape[0] - y)
                square_img = image[y:y+h, x:x+w].copy()
                row_images.append(square_img)
            grid.append(row_images)
        return grid


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from chessmate.config import ChessConfig

    print("测试棋盘检测器...")

    cfg = ChessConfig()
    detector = BoardDetector(cfg)

    # 创建模拟棋盘图像
    board_size = 8 * cfg.web_square_size
    test_image = np.zeros((board_size + 100, board_size + 100, 3), dtype=np.uint8)

    # 使用手动模式检测
    cfg.vision_screenshot_region = None  # 不设置自动区域
    region = detector.detect_manual(test_image)
    print(f"手动检测结果:")
    print(f"  棋盘区域: {region.top_left} -> {region.bottom_right}")
    print(f"  格子大小: {region.square_size:.1f} px")
    print(f"  宽度x高度: {region.width}x{region.height}")

    # 测试方格坐标
    center_a1 = region.get_square_center(7, 0)  # a1
    center_h8 = region.get_square_center(0, 7)  # h8
    print(f"  a1 格中心: {center_a1}")
    print(f"  h8 格中心: {center_h8}")

    print("\n棋盘检测器测试通过！")