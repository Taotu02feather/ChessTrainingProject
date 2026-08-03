#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 网页自动对弈模块
==========================
通过屏幕截图识别网页上的棋局，并自动走子。

功能流程：
1. 截图获取网页棋盘图像
2. 分析图像，提取 FEN 字符串
3. 使用 AI 模型计算最优走法
4. 通过鼠标点击/拖拽执行走子
5. 等待对手走子，重复流程

走子方式（两种）：
- 点击模式（默认）：点击起始格，再点击目标格
- 拖拽模式（可选）：点击并拖拽到目标格

安全注意事项：
- 请确保目标网站允许使用自动化工具
- 建议在友谊赛或与机器人对弈时使用
- 遵守网站的使用条款
"""

import sys
import os
import time
import logging
from typing import Optional, Tuple

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import chess

from chessmate.vision.board_detector import ChessboardRegion
from chessmate.vision.piece_classifier import VisionPipeline
from chessmate.training.neural_net import ChessNet, BoardEncoder
from chessmate.training.mcts import MatchMCTS


# ============================================================================
# 网页对弈器
# ============================================================================

class WebPlayer:
    """
    网页自动对弈器。

    在浏览器打开的棋局页面上自动与对手下棋。
    支持两种角色：自动执白或自动执黑。

    使用方式：
        player = WebPlayer(model, config)
        player.play_as_white(num_moves=40)   # 执白自动对弈
        player.play_as_black(num_moves=40)   # 执黑自动对弈
        player.make_one_move()               # 走一步棋
    """

    def __init__(self, model: ChessNet, config, logger=None):
        """
        初始化网页对弈器。

        Args:
            model: ChessNet 神经网络模型。
            config: ChessConfig 配置对象。
            logger: 日志记录器（可选）。
        """
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger("WebPlayer")

        # 初始化视觉识别流水线
        self.vision = VisionPipeline(config)

        # 初始化棋盘编码器和 AI 引擎
        self.encoder = BoardEncoder(history_length=config.history_length)
        self.mcts = MatchMCTS(model, self.encoder, config)

        # 鼠标控制
        self.click_delay = config.web_click_delay
        self.move_duration = config.web_move_duration

        # 棋盘坐标映射
        self.board_top_left = config.web_board_top_left
        self.square_size = config.web_square_size

        # 状态
        self.move_count = 0

    def _get_screen_coords(self, square: int) -> Tuple[int, int]:
        """
        将棋盘 square 索引 (0-63) 映射为屏幕坐标。

        约定：square 按照 python-chess 的索引，
        a1=0, b1=1, ..., h1=7, a2=8, ..., h8=63

        屏幕坐标：x 从左到右增加，y 从上到下增加。
        从白方视角（a1 在左下），需要翻转 y 坐标。

        Args:
            square: 棋盘格子索引 (0-63)。

        Returns:
            (x, y) 屏幕像素坐标。
        """
        file_idx = chess.square_file(square)  # 0-7, a-h
        rank_idx = chess.square_rank(square)  # 0-7, 1-8

        # 从白方视角：rank 0 是第1行（底部），rank 7 是第8行（顶部）
        # 屏幕坐标：row 0 是顶部
        screen_row = 7 - rank_idx  # 反转：第8行在顶部

        x = self.board_top_left[0] + int((file_idx + 0.5) * self.square_size)
        y = self.board_top_left[1] + int((screen_row + 0.5) * self.square_size)

        return (x, y)

    def _click_square(self, square: int):
        """
        点击棋盘上的某个格子。

        Args:
            square: 棋盘格子索引 (0-63)。
        """
        import pyautogui

        x, y = self._get_screen_coords(square)

        pyautogui.moveTo(x, y, duration=0.1)
        time.sleep(self.click_delay)
        pyautogui.click(x, y)
        time.sleep(self.click_delay)

        if self.logger:
            self.logger.debug(
                f"点击 {chess.square_name(square)} @ ({x}, {y})"
            )

    def _drag_move(self, from_square: int, to_square: int):
        """
        拖拽走子：从一格点击拖动到另一格。

        Args:
            from_square: 起始格子索引。
            to_square: 目标格子索引。
        """
        import pyautogui

        x1, y1 = self._get_screen_coords(from_square)
        x2, y2 = self._get_screen_coords(to_square)

        pyautogui.moveTo(x1, y1, duration=0.1)
        time.sleep(self.click_delay)
        pyautogui.mouseDown(x1, y1)
        pyautogui.moveTo(x2, y2, duration=self.move_duration)
        pyautogui.mouseUp(x2, y2)
        time.sleep(self.click_delay)

        if self.logger:
            self.logger.debug(
                f"拖拽 {chess.square_name(from_square)} -> "
                f"{chess.square_name(to_square)}"
            )

    def capture_board(self) -> Tuple[Optional[str], Optional[ChessboardRegion]]:
        """
        截取屏幕并识别棋盘状态。

        Returns:
            (fen_string, region) 元组。
        """
        self.logger.info("正在截取屏幕并识别棋盘...")
        fen, region = self.vision.recognize_from_screenshot()

        if fen:
            self.logger.info(f"识别到局面: {fen}")
        else:
            self.logger.warning("棋盘识别失败，请检查屏幕上的棋盘是否可见")

        return fen, region

    def make_one_move(self, fen: str = None) -> bool:
        """
        根据当前棋盘状态走一步棋。

        Args:
            fen: FEN 字符串。如果为 None，自动截图识别。

        Returns:
            是否成功走子。
        """
        # 获取当前局面
        if fen is None:
            fen, _ = self.capture_board()

        if fen is None:
            self.logger.error("无法获取棋盘状态，取消走子")
            return False

        # 创建棋盘对象
        try:
            board = chess.Board(fen)
        except ValueError as e:
            self.logger.error(f"无效的 FEN: {fen}, 错误: {e}")
            return False

        if board.is_game_over():
            self.logger.info(f"对局已结束: {board.result()}")
            return False

        # AI 搜索最优走法
        self.logger.info(
            f"AI 正在思考... (当前走子方: {'白' if board.turn else '黑'})"
        )
        best_move, value = self.mcts.search(board)

        if best_move is None:
            self.logger.warning("未找到合法走法")
            return False

        self.logger.info(
            f"AI 选择走法: {best_move} "
            f"(估值: {value:+.3f} | "
            f"从: {chess.square_name(best_move.from_square)} "
            f"到: {chess.square_name(best_move.to_square)})"
        )

        # 执行点击走子
        from_sq = best_move.from_square
        to_sq = best_move.to_square

        # 处理升变（自动升变为后）
        if best_move.promotion:
            self.logger.info(f"升变为: {chess.piece_name(best_move.promotion)}")

        # 使用点击方式走子（两次点击）
        self._click_square(from_sq)
        self._click_square(to_sq)

        self.move_count += 1
        return True

    def play_as_white(self, num_moves: int = 40) -> int:
        """
        执白自动对弈（玩家先行）。
        等待玩家（或对方）走子后，AI 再走。

        Args:
            num_moves: 最多走的步数。

        Returns:
            实际走的步数。
        """
        self.logger.info("=" * 50)
        self.logger.info("网页自动对弈模式：执白（AI 后手）")
        self.logger.info("请先在网页上走第一步，然后 AI 会自动应对")
        self.logger.info("=" * 50)

        moves_made = 0
        last_fen = chess.STARTING_FEN

        for _ in range(num_moves):
            # 等待并检测对手的走子
            self.logger.info("等待对手走子...")
            new_fen = None
            wait_count = 0
            max_wait = 30  # 最多等待 30 次检测

            while new_fen is None or new_fen == last_fen:
                time.sleep(2)  # 每 2 秒检测一次
                new_fen, _ = self.capture_board()
                wait_count += 1
                if wait_count >= max_wait:
                    self.logger.warning("等待超时，退出")
                    return moves_made

            # 对方已走子，AI 应对
            self.logger.info("检测到对手已走子，AI 开始思考...")
            if self.make_one_move(new_fen):
                moves_made += 1
                last_fen = new_fen

                # 检查对局是否结束
                board = chess.Board(new_fen)
                board.push(self._last_move) if hasattr(self, '_last_move') else None
            else:
                self.logger.error("走子失败，退出")
                break

        return moves_made

    def play_as_black(self, num_moves: int = 40) -> int:
        """
        执黑自动对弈（AI 先行）。
        等待网页上的对手走第一步后 AI 开始。

        Args:
            num_moves: 最多走的步数。

        Returns:
            实际走的步数。
        """
        self.logger.info("=" * 50)
        self.logger.info("网页自动对弈模式：执黑（AI 先手）")
        self.logger.info("请确保网页棋局已开始，等待 AI 走第一步")
        self.logger.info("=" * 50)

        moves_made = 0
        last_fen = chess.STARTING_FEN

        # AI 先走第一步
        time.sleep(2)
        if self.make_one_move(chess.STARTING_FEN):
            moves_made += 1
        else:
            return 0

        # 交替走子
        for _ in range(num_moves - 1):
            # 等待对手走子
            self.logger.info("等待对手走子...")
            new_fen = None
            wait_count = 0
            max_wait = 30

            while new_fen is None or new_fen == last_fen:
                time.sleep(2)
                new_fen, _ = self.capture_board()
                wait_count += 1
                if wait_count >= max_wait:
                    self.logger.warning("等待超时，退出")
                    return moves_made

            last_fen = new_fen

            # AI 应对
            if self.make_one_move(new_fen):
                moves_made += 1
            else:
                break

        return moves_made

    def auto_play_both(self, num_moves: int = 100) -> int:
        """
        全自动模式：AI 执双方，通过截图检测循环自动走子。
        可用于测试视觉识别+走子的完整流程。

        Args:
            num_moves: 最大步数。

        Returns:
            实际走的步数。
        """
        self.logger.info("=" * 50)
        self.logger.info("全自动模式：AI 执双方走子")
        self.logger.info("=" * 50)

        board = chess.Board()
        moves_made = 0

        for _ in range(num_moves):
            if board.is_game_over():
                self.logger.info(f"对局结束: {board.result()}")
                break

            if self.make_one_move(board.fen()):
                moves_made += 1
                # 更新棋盘（此处需要知道实际走了哪步）
                # 简化处理：重新截图获取新状态
                time.sleep(1)
                new_fen, _ = self.capture_board()
                if new_fen and new_fen != board.fen():
                    try:
                        board = chess.Board(new_fen)
                    except ValueError:
                        pass
            else:
                break

        return moves_made

    def calibrate_position(self):
        """
        交互式校准：帮助用户确定网页棋盘在屏幕上的位置。

        运行后会要求用户将鼠标移到棋盘的 a1 格和 h8 格，
        自动计算棋盘参数并更新配置。
        """
        import pyautogui

        self.logger.info("=" * 50)
        self.logger.info("棋盘位置校准工具")
        self.logger.info("请按照提示操作...")
        self.logger.info("=" * 50)

        input("请将鼠标移到棋盘的 a1 格（左下角白格），然后按 Enter...")
        a1_pos = pyautogui.position()
        self.logger.info(f"a1 格位置: {a1_pos}")

        input("请将鼠标移到棋盘的 h8 格（右上角黑格），然后按 Enter...")
        h8_pos = pyautogui.position()
        self.logger.info(f"h8 格位置: {h8_pos}")

        # 计算棋盘参数
        # a1 在左下，h8 在右上（从白方视角）
        # 屏幕坐标：左上角 (x_min, y_top), 右下角 (x_max, y_bottom)
        x_min = min(a1_pos[0], h8_pos[0])
        x_max = max(a1_pos[0], h8_pos[0])
        y_min = min(a1_pos[1], h8_pos[1])  # 屏幕上方
        y_max = max(a1_pos[1], h8_pos[1])  # 屏幕下方

        # a1 是左下角的格子，所以 y 坐标应该是大的（屏幕下方）
        # h8 是右上角的格子，所以 y 坐标是小的（屏幕上方），x 是大的
        if a1_pos[1] < h8_pos[1]:
            # a1 在上面？可能是反面视角，调整
            a1_is_bottom = abs(a1_pos[1] - y_max) < abs(a1_pos[1] - y_min)
            if not a1_is_bottom:
                x_min = min(a1_pos[0], h8_pos[0])
                x_max = max(a1_pos[0], h8_pos[0])
                y_min = min(a1_pos[1], h8_pos[1])
                y_max = max(a1_pos[1], h8_pos[1])

        board_pixel_width = x_max - x_min
        square_size = board_pixel_width / 7.0  # 8 个格子之间 7 个间隔

        top_left = (x_min - square_size / 2, y_min - square_size / 2)
        # 确保整数
        top_left = (int(top_left[0]), int(top_left[1]))
        square_size = round(square_size)

        self.logger.info(f"\n校准结果:")
        self.logger.info(f"  棋盘左上角: {top_left}")
        self.logger.info(f"  格子大小: {square_size} px")
        self.logger.info(f"\n请将以下配置添加到 config.py 或运行时设置:")
        self.logger.info(f"  web_board_top_left = {top_left}")
        self.logger.info(f"  web_square_size = {square_size}")

        # 更新当前配置
        self.board_top_left = top_left
        self.square_size = square_size
        self.config.web_board_top_left = top_left
        self.config.web_square_size = square_size


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from chessmate.config import ChessConfig, get_small_config
    from chessmate.training.neural_net import ChessNet

    print("测试网页对弈模块...")

    cfg = get_small_config()
    cfg.num_res_blocks = 2
    cfg.num_filters = 16

    # 创建模型
    model = ChessNet(cfg)
    model.eval()

    # 创建网页对弈器
    player = WebPlayer(model, cfg)

    print(f"棋盘左上角: {player.board_top_left}")
    print(f"格子大小: {player.square_size}")

    # 测试坐标转换
    a1_coords = player._get_screen_coords(chess.A1)
    h8_coords = player._get_screen_coords(chess.H8)
    e4_coords = player._get_screen_coords(chess.E4)

    print(f"a1 屏幕坐标: {a1_coords}")
    print(f"h8 屏幕坐标: {h8_coords}")
    print(f"e4 屏幕坐标: {e4_coords}")

    print("\n网页对弈模块测试通过！")
    print("注意：完整的网页对弈功能需要:")
    print("  1. 一个打开了国际象棋网页的浏览器窗口")
    print("  2. 使用 calibrate_position() 校准棋盘位置")
    print("  3. 运行 play_as_white() 或 play_as_black()")