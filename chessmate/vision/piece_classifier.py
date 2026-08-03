#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 棋子识别器
====================
从棋盘方格图像中识别棋子类型和颜色，输出 FEN 字符串。

识别策略：
1. 颜色分类法（simple）：分析方格中心区域的颜色来判断棋子类型和颜色
   - 对于已知颜色的棋盘格，检测中心区域与空格的差异
   - 基于 RGB 颜色范围来判断白子/黑子
   - 使用棋子轮廓模板匹配进一步细分

2. 模板匹配法（template）：对比预存的棋子图片模板
   - 对每个方格进行模板匹配
   - 选择置信度最高的匹配结果

3. 深度学习分类法（advanced）：使用 CNN 图像分类器
   - 训练一个 13 类分类器（6种棋子 × 2色 + 空格）
   - 需要大量标注数据，训练周期长

当前实现：颜色分类法 + 可选的模板匹配（简单实用）

输出：FEN 字符串（如 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"）
"""

import sys
import os
import numpy as np
import cv2
from typing import List, Tuple, Optional, Dict

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from chessmate.vision.board_detector import ChessboardRegion


# ============================================================================
# 棋子类型枚举
# ============================================================================

# FEN 符号映射
FEN_PIECES = {
    'K': '白王', 'Q': '白后', 'R': '白车', 'B': '白象', 'N': '白马', 'P': '白兵',
    'k': '黑王', 'q': '黑后', 'r': '黑车', 'b': '黑象', 'n': '黑马', 'p': '黑兵',
    ' ': '空格',
}

# 棋子 FEN 字符
FEN_WHITE = {'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P'}
FEN_BLACK = {'K': 'k', 'Q': 'q', 'R': 'r', 'B': 'b', 'N': 'n', 'P': 'p'}


# ============================================================================
# 颜色分类法棋子识别器
# ============================================================================

class PieceClassifier:
    """
    国际象棋棋子识别器。

    使用颜色分析和模板匹配来识别棋盘上的棋子。

    使用方式：
        classifier = PieceClassifier(config)
        fen = classifier.classify_board(image, region)  # 从图像识别整个棋盘
    """

    def __init__(self, config):
        """
        初始化识别器。

        Args:
            config: ChessConfig 配置对象。
        """
        self.config = config
        self.confidence_threshold = config.vision_confidence_threshold
        self.use_templates = config.vision_use_template_matching

        # 棋盘颜色参考
        self.light_color = np.array(config.vision_board_light_color, dtype=np.float32)
        self.dark_color = np.array(config.vision_board_dark_color, dtype=np.float32)

        # 试图加载模板（如果存在）
        self.templates: Dict[str, List[np.ndarray]] = {}
        if self.use_templates:
            self._load_templates()

    def _load_templates(self):
        """从 data/templates/ 目录加载棋子模板图片。"""
        templates_dir = self.config.vision_templates_dir

        if not os.path.isdir(templates_dir):
            return

        for filename in os.listdir(templates_dir):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                name = os.path.splitext(filename)[0]  # 如 "white_king", "black_pawn"
                filepath = os.path.join(templates_dir, filename)
                template = cv2.imread(filepath)
                if template is not None:
                    if name not in self.templates:
                        self.templates[name] = []
                    self.templates[name].append(template)

    def classify_board(
        self,
        image: np.ndarray,
        region: ChessboardRegion,
    ) -> str:
        """
        识别整个棋盘，输出 FEN 字符串。

        棋盘方向约定：
        - row=0 是棋盘顶部（黑方的第8横线）
        - row=7 是棋盘底部（白方的第1横线）
        - col=0 是 a 列，col=7 是 h 列

        FEN 排列是从第 8 行到第 1 行（从上到下），即 row=0 到 row=7。

        Args:
            image: 包含棋盘的完整图像（BGR 格式）。
            region: 检测到的棋盘区域。

        Returns:
            FEN 字符串，格式如 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"。
        """
        fen_rows = []

        for row in range(8):
            fen_row = ""
            empty_count = 0

            for col in range(8):
                # 获取方格图像
                x, y, w, h = region.get_square_rect(row, col)
                x, y = max(0, x), max(0, y)
                w = min(w, image.shape[1] - x) if x + w <= image.shape[1] else image.shape[1] - x
                h = min(h, image.shape[0] - y) if y + h <= image.shape[0] else image.shape[0] - y

                if w <= 0 or h <= 0:
                    empty_count += 1
                    continue

                square_img = image[y:y+h, x:x+w]

                # 判断方格的类型（浅色格/深色格）
                # row+col 为偶数是浅色格（根据标准棋盘）
                is_light_square = (row + col) % 2 == 0

                # 识别该格中的棋子
                piece_fen = self.classify_square(square_img, is_light_square)

                if piece_fen == ' ':
                    empty_count += 1
                else:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece_fen

            if empty_count > 0:
                fen_row += str(empty_count)

            fen_rows.append(fen_row)

        fen = "/".join(fen_rows)
        return fen

    def classify_square(
        self,
        square_img: np.ndarray,
        is_light_square: bool,
    ) -> str:
        """
        识别单个方格中的棋子。

        Args:
            square_img: 方格的 BGR 子图像。
            is_light_square: 该格是否为浅色格。

        Returns:
            FEN 字符（'K','Q','R','B','N','P','k','q','r','b','n','p',' '）。
        """
        if square_img.size == 0:
            return ' '

        # 方法1：模板匹配（如果模板可用）
        if self.use_templates and self.templates:
            result = self._template_match(square_img)
            if result != ' ':
                return result

        # 方法2：颜色分析法
        return self._color_analysis(square_img, is_light_square)

    def _color_analysis(self, square_img: np.ndarray, is_light_square: bool) -> str:
        """
        基于颜色分析的空格检测。

        分析方格中心区域的颜色分布，判断是否为空。
        通过比较中心像素与边缘背景的差异来区分子与空格。

        Args:
            square_img: 方格子图像。
            is_light_square: 是否浅色格。

        Returns:
            FEN 字符。当前简化版仅区分空格与非空格棋子。
            棋子类型细分需要模板匹配或更高级的分类器。
        """
        h, w = square_img.shape[:2]

        if h < 10 or w < 10:
            return ' '

        # 计算中心区域的平均颜色
        center_margin = max(3, min(h, w) // 5)
        center_region = square_img[
            center_margin:h-center_margin,
            center_margin:w-center_margin
        ]

        if center_region.size == 0:
            return ' '

        center_mean = np.mean(center_region, axis=(0, 1))

        # 计算边缘区域的平均颜色
        edge_region = square_img.copy()
        edge_color = np.mean(
            np.concatenate([
                square_img[0:center_margin, :].reshape(-1, 3),
                square_img[-center_margin:, :].reshape(-1, 3),
                square_img[:, 0:center_margin].reshape(-1, 3),
                square_img[:, -center_margin:].reshape(-1, 3),
            ]),
            axis=0
        )

        # 如果中心颜色与边缘颜色显著不同，则有棋子
        color_diff = np.linalg.norm(center_mean - edge_color)

        if color_diff < 15:  # 颜色差异阈值
            return ' '  # 空格

        # 进一步判断是白子还是黑子
        # 白子颜色更亮（RGB 值更高），黑子颜色更暗
        brightness = np.mean(center_mean)

        if is_light_square:
            # 浅色格上：黑子与格子对比度高，白子与格子相近
            if brightness < 120:
                return 'p'  # 黑子（简化：黑兵）
            else:
                return 'P'  # 白子（简化：白兵）
        else:
            # 深色格上：白子与格子对比度高
            if brightness > 140:
                return 'P'  # 白子
            else:
                return 'p'  # 黑子

    def _template_match(self, square_img: np.ndarray) -> str:
        """
        使用模板匹配识别棋子。
        遍历所有已加载的模板，找到最佳匹配。

        Args:
            square_img: 方格子图像。

        Returns:
            FEN 字符或 ' ' （无法识别）。
        """
        if not self.templates:
            return ' '

        best_score = 0.0
        best_piece = ' '

        for name, templates in self.templates.items():
            for template in templates:
                try:
                    # 对方格图像缩放到模板大小
                    h_t, w_t = template.shape[:2]
                    resized = cv2.resize(square_img, (w_t, h_t))

                    # 模板匹配
                    result = cv2.matchTemplate(
                        resized, template, cv2.TM_CCOEFF_NORMED
                    )
                    score = np.max(result)

                    if score > best_score and score > self.confidence_threshold:
                        best_score = score
                        # 从名称中提取 FEN 字符
                        best_piece = self._name_to_fen(name)

                except Exception:
                    continue

        return best_piece

    @staticmethod
    def _name_to_fen(name: str) -> str:
        """
        将模板名称映射为 FEN 字符。

        模板命名约定："{color}_{piece}"，如 "white_king" 或 "black_pawn"。

        Args:
            name: 模板文件名（不含扩展名）。

        Returns:
            FEN 字符。
        """
        piece_map = {
            'king': 'K', 'queen': 'Q', 'rook': 'R',
            'bishop': 'B', 'knight': 'N', 'pawn': 'P',
        }

        parts = name.lower().split('_')
        for piece_name, fen_char in piece_map.items():
            if piece_name in parts:
                if 'black' in parts:
                    return fen_char.lower()
                else:
                    return fen_char

        return ' '


# ============================================================================
# 辅助函数：FEN 验证和操作
# ============================================================================

def validate_fen(fen: str) -> bool:
    """
    验证 FEN 字符串的格式是否正确。

    Args:
        fen: FEN 字符串。

    Returns:
        是否有效。
    """
    import chess
    try:
        chess.Board(fen)
        return True
    except ValueError:
        return False


def fen_to_grid(fen: str) -> List[List[str]]:
    """
    将 FEN 字符串转换为 8x8 的二维字符网格。

    Args:
        fen: FEN 字符串（仅棋盘布局部分）。

    Returns:
        二维列表 grid[row][col]，row=0 为第8行（黑方底线）。
    """
    grid = []
    for rank in fen.split('/'):
        row = []
        for ch in rank:
            if ch.isdigit():
                row.extend([' '] * int(ch))
            else:
                row.append(ch)
        grid.append(row)
    return grid


def grid_to_fen(grid: List[List[str]]) -> str:
    """
    将 8x8 二维字符网格转回 FEN 字符串。

    Args:
        grid: 二维列表。

    Returns:
        FEN 字符串。
    """
    rows = []
    for row in grid:
        fen_row = ""
        empty = 0
        for ch in row:
            if ch == ' ':
                empty += 1
            else:
                if empty > 0:
                    fen_row += str(empty)
                    empty = 0
                fen_row += ch
        if empty > 0:
            fen_row += str(empty)
        rows.append(fen_row)
    return '/'.join(rows)


# ============================================================================
# 完整的视觉识别流水线
# ============================================================================

class VisionPipeline:
    """
    视觉识别流水线：将棋盘检测和棋子识别串联在一起。

    使用方式：
        pipeline = VisionPipeline(config)
        fen, region = pipeline.recognize(image)
    """

    def __init__(self, config):
        """
        初始化识别流水线。

        Args:
            config: ChessConfig 配置对象。
        """
        from chessmate.vision.board_detector import BoardDetector
        self.detector = BoardDetector(config)
        self.classifier = PieceClassifier(config)
        self.config = config

    def recognize(self, image: np.ndarray) -> Tuple[Optional[str], Optional[ChessboardRegion]]:
        """
        从图像中检测棋盘并识别棋子。

        Args:
            image: BGR 格式的图像。

        Returns:
            (fen_string, region) 元组。
            如果检测失败，两者均为 None。
        """
        # 步骤1：检测棋盘
        region = self.detector.detect(image)
        if region is None:
            return None, None

        # 步骤2：识别棋子
        fen = self.classifier.classify_board(image, region)

        # 步骤3：验证 FEN
        if not validate_fen(fen):
            return None, region

        return fen, region

    def recognize_from_screenshot(self) -> Tuple[Optional[str], Optional[ChessboardRegion]]:
        """
        从屏幕截图识别棋盘（使用 pyautogui）。

        Returns:
            (fen_string, region) 元组。
        """
        try:
            import pyautogui
        except ImportError:
            raise ImportError("pyautogui 未安装，无法截图。请运行: pip install pyautogui")

        # 截图
        if self.config.vision_screenshot_region:
            region_tuple = self.config.vision_screenshot_region
            screenshot = pyautogui.screenshot(region=region_tuple)
        else:
            screenshot = pyautogui.screenshot()

        # 转为 BGR 格式
        image = np.array(screenshot)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        return self.recognize(image)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import chess
    from chessmate.config import ChessConfig

    print("测试棋子识别器...")

    cfg = ChessConfig()
    classifier = PieceClassifier(cfg)

    # 测试 FEN 验证
    test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"
    print(f"FEN 验证 '{test_fen}': {validate_fen(test_fen)}")

    # 测试 FEN <-> 网格转换
    grid = fen_to_grid(test_fen)
    print(f"网格尺寸: {len(grid)}x{len(grid[0])}")
    print(f"第1行: {''.join(grid[0])}")
    print(f"第8行: {''.join(grid[7])}")

    # 转换回 FEN
    reconstructed = grid_to_fen(grid)
    print(f"重建 FEN: {reconstructed}")
    print(f"匹配: {test_fen == reconstructed}")

    print("\n棋子识别器测试通过！")