#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 本地 GUI 对弈窗口
===========================
使用 PyQt5 实现的国际象棋对弈界面。

功能特性：
- 绘制标准的 8x8 棋盘，含坐标标签
- 展示棋子（使用 Unicode 国际象棋符号）
- 支持鼠标点击走子（点击起始格 + 目标格）
- 显示走子历史和状态信息
- 玩家可选择执白或执黑
- AI 思考时显示状态提示
- 支持新局、悔棋（未来扩展）、保存棋谱

交互流程：
1. 玩家点击一个格子选中棋子（高亮显示合法走法）
2. 玩家点击目标格子执行走子
3. AI 自动回应（如果轮到 AI）
4. 更新棋盘显示和走子历史
"""

import sys
import os
import time
import threading

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from typing import Optional, List, Tuple

import chess

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QGroupBox, QRadioButton,
    QMessageBox, QStatusBar, QFrame, QGridLayout, QSizePolicy,
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QRect, QSize
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush, QMouseEvent,
    QPixmap, QPalette,
)


# ============================================================================
# AI 思考线程
# ============================================================================

class AIThinkThread(QThread):
    """
    在后台线程中运行 AI 搜索，防止 GUI 冻结。

    信号:
        move_ready: 发出找到的最优走法 (move_uci_str, value)
        error_occurred: 发出错误信息 str
    """
    move_ready = pyqtSignal(str, float)
    error_occurred = pyqtSignal(str)

    def __init__(self, mcts, board: chess.Board):
        super().__init__()
        self.mcts = mcts
        self.board = board.copy()

    def run(self):
        """在后台线程中执行 MCTS 搜索。"""
        try:
            best_move, value = self.mcts.search(self.board)
            if best_move:
                self.move_ready.emit(best_move.uci(), value)
            else:
                self.error_occurred.emit("AI 未找到合法走法")
        except Exception as e:
            self.error_occurred.emit(str(e))


# ============================================================================
# 棋盘绘制组件
# ============================================================================

class ChessBoardWidget(QWidget):
    """
    国际象棋棋盘绘制组件。

    负责绘制棋盘、棋子，处理鼠标点击选择格子。
    """

    # 颜色定义
    COLOR_LIGHT = QColor(240, 217, 181)   # 浅色格（白格）
    COLOR_DARK = QColor(181, 136, 99)     # 深色格（黑格）
    COLOR_SELECTED = QColor(130, 180, 80, 160)   # 选中格高亮（绿）
    COLOR_LEGAL_MOVE = QColor(130, 180, 80, 100)  # 合法走法指示
    COLOR_LAST_MOVE = QColor(255, 255, 100, 120)   # 上一步走法高亮（黄）
    COLOR_BORDER = QColor(80, 50, 30)     # 边框颜色

    # Unicode 国际象棋符号（白方使用实心，黑方使用空心）
    PIECE_SYMBOLS = {
        'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
        'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟',
    }

    square_clicked = pyqtSignal(int)  # 发出被点击的格子索引 (0-63)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.square_size = config.gui_square_size
        self.show_coordinates = config.gui_show_coordinates

        # 棋盘大小（含坐标边距）
        self.margin = 25
        total_size = self.square_size * 8 + self.margin * 2
        self.setMinimumSize(total_size, total_size)
        self.setFixedSize(total_size, total_size)

        # 状态
        self.board = chess.Board()
        self.selected_square: Optional[int] = None
        self.legal_moves_for_selected: List[chess.Move] = []
        self.last_move: Optional[chess.Move] = None
        self.flip_board = False  # 是否翻转棋盘（黑方视角）

        # 启用鼠标追踪
        self.setMouseTracking(True)

    def set_board(self, board: chess.Board):
        """更新棋盘状态并重绘。"""
        self.board = board.copy()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.update()

    def set_last_move(self, move: chess.Move):
        """设置上一步走法，用于高亮显示。"""
        self.last_move = move
        self.update()

    def set_flipped(self, flipped: bool):
        """设置棋盘方向。"""
        self.flip_board = flipped
        self.update()

    def _screen_to_square(self, x: int, y: int) -> Optional[int]:
        """
        将屏幕像素坐标转换为棋盘 square 索引。

        Args:
            x, y: 组件内的像素坐标。

        Returns:
            square 索引 (0-63) 或 None（点在棋盘外）。
        """
        board_x = x - self.margin
        board_y = y - self.margin

        if board_x < 0 or board_y < 0:
            return None
        if board_x >= self.square_size * 8 or board_y >= self.square_size * 8:
            return None

        col = board_x // self.square_size
        row = board_y // self.square_size

        # 考虑翻转
        if self.flip_board:
            col = 7 - col
            row = 7 - row

        # 转为 square 索引 (0-63, a1=0)
        # 标准视图：屏幕 row 0 = 第 8 横线
        square_row = 7 - row
        square = square_row * 8 + col

        return square

    def mousePressEvent(self, event: QMouseEvent):
        """处理鼠标点击事件。"""
        if event.button() == Qt.LeftButton:
            square = self._screen_to_square(event.x(), event.y())
            if square is not None:
                self.square_clicked.emit(square)

    def paintEvent(self, event):
        """绘制棋盘和棋子。"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制棋盘背景
        self._draw_board(painter)
        # 绘制坐标标签
        if self.show_coordinates:
            self._draw_coordinates(painter)
        # 绘制棋子
        self._draw_pieces(painter)
        # 绘制选中高亮和合法走法
        self._draw_highlights(painter)

        painter.end()

    def _draw_board(self, painter: QPainter):
        """绘制 8x8 方格棋盘和边框。"""
        # 外边框
        pen = QPen(self.COLOR_BORDER, 2)
        painter.setPen(pen)
        painter.drawRect(
            self.margin - 1, self.margin - 1,
            self.square_size * 8 + 1, self.square_size * 8 + 1
        )

        # 绘制方格
        for row in range(8):
            for col in range(8):
                x = self.margin + col * self.square_size
                y = self.margin + row * self.square_size

                # 确定颜色
                if (row + col) % 2 == 0:
                    color = self.COLOR_LIGHT
                else:
                    color = self.COLOR_DARK

                painter.fillRect(x, y, self.square_size, self.square_size, color)

    def _draw_coordinates(self, painter: QPainter):
        """绘制行列坐标标签 (a-h, 1-8)。"""
        font = QFont("Arial", max(8, self.square_size // 5))
        painter.setFont(font)
        painter.setPen(QPen(QColor(100, 70, 50)))

        files = "abcdefgh"
        for i in range(8):
            # 列标签 (a-h)
            x = self.margin + i * self.square_size + self.square_size // 2 - 5
            y_top = self.margin - 8
            y_bottom = self.margin + 8 * self.square_size + 15

            if self.flip_board:
                col_idx = 7 - i
            else:
                col_idx = i

            painter.drawText(x, y_bottom, files[col_idx])

            # 行标签 (1-8)
            x_left = self.margin - 18
            y = self.margin + i * self.square_size + self.square_size // 2 + 5

            if self.flip_board:
                row_idx = i + 1  # 翻转时，屏幕顶部 = 第1行
            else:
                row_idx = 8 - i  # 屏幕顶部 = 第8行

            painter.drawText(x_left, y, str(row_idx))

    def _draw_pieces(self, painter: QPainter):
        """绘制棋盘上的所有棋子。"""
        # 棋子字体（根据格子大小自适应）
        font_size = int(self.square_size * 0.75)
        font = QFont("Segoe UI Symbol", font_size)
        painter.setFont(font)

        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece is None:
                continue

            symbol = piece.symbol()  # 获取 FEN 字符
            unicode_char = self.PIECE_SYMBOLS.get(symbol, '?')

            # 计算屏幕位置
            file_idx = chess.square_file(square)
            rank_idx = chess.square_rank(square)

            if self.flip_board:
                screen_col = 7 - file_idx
                screen_row = rank_idx
            else:
                screen_col = file_idx
                screen_row = 7 - rank_idx

            x = self.margin + screen_col * self.square_size
            y = self.margin + screen_row * self.square_size

            # 设置棋子绘制区域
            text_rect = QRect(x, y, self.square_size, self.square_size)

            # 棋子颜色和阴影
            if piece.color == chess.WHITE:
                # 先绘制阴影（偏移 1 像素，让白子在浅色格上也可见）
                painter.setPen(QPen(QColor(60, 60, 60, 80)))
                shadow_rect = QRect(x + 1, y + 1, self.square_size, self.square_size)
                painter.drawText(shadow_rect, Qt.AlignCenter, unicode_char)
                # 再绘制白色棋子本体
                painter.setPen(QPen(QColor(250, 250, 250)))
            else:
                # 黑子同样加阴影
                painter.setPen(QPen(QColor(20, 20, 20, 80)))
                shadow_rect = QRect(x + 1, y + 1, self.square_size, self.square_size)
                painter.drawText(shadow_rect, Qt.AlignCenter, unicode_char)
                # 黑子本体
                painter.setPen(QPen(QColor(30, 30, 30)))

            painter.drawText(text_rect, Qt.AlignCenter, unicode_char)

    def _draw_highlights(self, painter: QPainter):
        """绘制选中格高亮和合法走法指示。"""
        # 上一步走法高亮
        if self.last_move:
            for sq in [self.last_move.from_square, self.last_move.to_square]:
                self._highlight_square(painter, sq, self.COLOR_LAST_MOVE)

        # 选中格高亮
        if self.selected_square is not None:
            self._highlight_square(painter, self.selected_square, self.COLOR_SELECTED)

            # 合法走法指示
            for move in self.legal_moves_for_selected:
                to_sq = move.to_square
                self._draw_legal_move_indicator(painter, to_sq)

    def _highlight_square(self, painter: QPainter, square: int, color: QColor):
        """高亮某个格子。"""
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        if self.flip_board:
            screen_col = 7 - file_idx
            screen_row = rank_idx
        else:
            screen_col = file_idx
            screen_row = 7 - rank_idx

        x = self.margin + screen_col * self.square_size
        y = self.margin + screen_row * self.square_size

        painter.fillRect(x, y, self.square_size, self.square_size, color)

    def _draw_legal_move_indicator(self, painter: QPainter, square: int):
        """在合法走法的目标格上绘制指示圆点。"""
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)

        if self.flip_board:
            screen_col = 7 - file_idx
            screen_row = rank_idx
        else:
            screen_col = file_idx
            screen_row = 7 - rank_idx

        cx = self.margin + screen_col * self.square_size + self.square_size // 2
        cy = self.margin + screen_row * self.square_size + self.square_size // 2
        radius = self.square_size // 6

        painter.setBrush(QBrush(self.COLOR_LEGAL_MOVE))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(cx - radius), int(cy - radius),
                           int(radius * 2), int(radius * 2))


# ============================================================================
# 主窗口
# ============================================================================

class ChessWindow(QMainWindow):
    """
    ChessMate 本地对弈主窗口。

    提供完整的 GUI 对弈体验，包括棋盘显示、走子历史、状态栏等。
    """

    def __init__(self, model, config, logger=None):
        """
        初始化 GUI 窗口。

        Args:
            model: ChessNet 神经网络模型。
            config: ChessConfig 配置对象。
            logger: 日志记录器。
        """
        super().__init__()
        self.model = model
        self.config = config
        self.logger = logger

        # 初始化 AI
        from chessmate.training.neural_net import BoardEncoder
        from chessmate.training.mcts import MatchMCTS

        self.encoder = BoardEncoder(history_length=config.history_length)
        self.mcts = MatchMCTS(model, self.encoder, config)

        # 对局状态
        self.board = chess.Board()
        self.player_is_white = config.gui_player_color
        self.ai_is_thinking = False
        self.move_history: List[str] = []
        self.think_thread: Optional[AIThinkThread] = None

        # 初始化 UI
        self._init_ui()
        self._update_title()

    def _init_ui(self):
        """初始化用户界面。"""
        self.setWindowTitle("ChessMate - 国际象棋 AI")
        self.setMinimumSize(self.config.gui_window_width, self.config.gui_window_height)

        # 主部件
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ---- 左侧：棋盘 ----
        self.board_widget = ChessBoardWidget(self.config)
        self.board_widget.square_clicked.connect(self._on_square_clicked)
        main_layout.addWidget(self.board_widget)

        # ---- 右侧：控制面板 ----
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_panel.setFixedWidth(280)

        # 颜色选择
        color_group = QGroupBox("玩家执子")
        color_layout = QVBoxLayout()
        self.radio_white = QRadioButton("执白 (先手)")
        self.radio_black = QRadioButton("执黑 (后手)")
        self.radio_white.setChecked(self.player_is_white)
        self.radio_black.setChecked(not self.player_is_white)
        self.radio_white.toggled.connect(self._on_color_changed)
        color_layout.addWidget(self.radio_white)
        color_layout.addWidget(self.radio_black)
        color_group.setLayout(color_layout)
        right_layout.addWidget(color_group)

        # 按钮
        btn_layout = QVBoxLayout()
        self.btn_new_game = QPushButton("新对局")
        self.btn_new_game.clicked.connect(self._new_game)
        self.btn_flip = QPushButton("翻转棋盘")
        self.btn_flip.clicked.connect(self._flip_board)
        self.btn_undo = QPushButton("悔棋 (开发中)")
        self.btn_undo.setEnabled(False)
        btn_layout.addWidget(self.btn_new_game)
        btn_layout.addWidget(self.btn_flip)
        btn_layout.addWidget(self.btn_undo)
        right_layout.addLayout(btn_layout)

        # 走子历史
        history_group = QGroupBox("走子历史")
        history_layout = QVBoxLayout()
        self.history_text = QTextEdit()
        self.history_text.setReadOnly(True)
        self.history_text.setMaximumHeight(400)
        history_layout.addWidget(self.history_text)
        history_group.setLayout(history_layout)
        right_layout.addWidget(history_group)

        # AI 状态
        self.ai_status_label = QLabel("AI 状态: 就绪")
        self.ai_status_label.setStyleSheet("color: green; font-weight: bold;")
        right_layout.addWidget(self.ai_status_label)

        right_layout.addStretch()

        main_layout.addWidget(right_panel)

        # ---- 状态栏 ----
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("欢迎使用 ChessMate！请走子或开始新对局。")

    def _on_square_clicked(self, square: int):
        """
        处理棋盘格子点击事件。

        Args:
            square: 被点击的格子索引 (0-63)。
        """
        if self.ai_is_thinking:
            self.status_bar.showMessage("AI 正在思考，请稍候...")
            return

        # 当前走子方
        is_player_turn = (
            (self.player_is_white and self.board.turn == chess.WHITE) or
            (not self.player_is_white and self.board.turn == chess.BLACK)
        )

        if not is_player_turn:
            self.status_bar.showMessage("轮到 AI 走子，请等待...")
            return

        selected = self.board_widget.selected_square

        if selected is None:
            # 第一步：选中一个格子
            piece = self.board.piece_at(square)
            if piece and piece.color == self.board.turn:
                self.board_widget.selected_square = square
                # 计算该格子的所有合法走法
                self.board_widget.legal_moves_for_selected = [
                    move for move in self.board.legal_moves
                    if move.from_square == square
                ]
                self.board_widget.update()
                piece_name = chess.piece_name(piece.piece_type)
                self.status_bar.showMessage(
                    f"选中: {chess.square_name(square)} 上的{piece_name}"
                )
        else:
            # 第二步：尝试走子
            move = chess.Move(selected, square)

            # 检查是否需要升变（兵到达底线）
            piece = self.board.piece_at(selected)
            if piece and piece.piece_type == chess.PAWN:
                rank = chess.square_rank(square)
                if rank in [0, 7]:
                    # 自动升变为后（简化处理）
                    move = chess.Move(selected, square, promotion=chess.QUEEN)

            if move in self.board.legal_moves:
                self._execute_player_move(move)
            else:
                # 点击了非法目标，切换选择
                new_piece = self.board.piece_at(square)
                if new_piece and new_piece.color == self.board.turn:
                    self.board_widget.selected_square = square
                    self.board_widget.legal_moves_for_selected = [
                        m for m in self.board.legal_moves
                        if m.from_square == square
                    ]
                else:
                    self.board_widget.selected_square = None
                    self.board_widget.legal_moves_for_selected = []
                self.board_widget.update()

    def _execute_player_move(self, move: chess.Move):
        """
        执行玩家的走子。

        Args:
            move: 要执行的走法。
        """
        san = self.board.san(move)  # 标准代数记谱法
        self.board.push(move)
        self.move_history.append(san)

        self.board_widget.set_last_move(move)
        self.board_widget.selected_square = None
        self.board_widget.legal_moves_for_selected = []
        self.board_widget.set_board(self.board)

        self._update_history()
        self._update_title()

        # 检查对局是否结束
        if self.board.is_game_over():
            self._handle_game_end()
            return

        # AI 走子
        self._ai_move()

    def _ai_move(self):
        """触发 AI 走子。"""
        if self.board.is_game_over():
            return

        self.ai_is_thinking = True
        self.ai_status_label.setText("AI 状态: 思考中...")
        self.ai_status_label.setStyleSheet("color: orange; font-weight: bold;")
        self.status_bar.showMessage("AI 正在思考...")

        # 在后台线程中运行 AI 搜索
        self.think_thread = AIThinkThread(self.mcts, self.board)
        self.think_thread.move_ready.connect(self._on_ai_move_ready)
        self.think_thread.error_occurred.connect(self._on_ai_error)
        self.think_thread.start()

        # 设置一个最短思考时间（让玩家看到 AI 的"思考"过程）
        self.think_start_time = time.time()
        self.min_think_timer = QTimer()
        self.min_think_timer.setSingleShot(True)
        self.min_think_timer.timeout.connect(self._min_think_elapsed)
        self.min_think_timer.start(int(self.config.gui_ai_thinking_time * 1000))
        self.ai_move_ready = False
        self.ai_best_move = None
        self.ai_best_value = 0.0

    def _on_ai_move_ready(self, uci_move: str, value: float):
        """AI 走法就绪（来自后台线程）。"""
        self.ai_best_move = chess.Move.from_uci(uci_move)
        self.ai_best_value = value
        self.ai_move_ready = True

    def _min_think_elapsed(self):
        """最小思考时间已过，如果 AI 结果已就绪则执行走子。"""
        if self.ai_move_ready and self.ai_best_move:
            self._execute_ai_move()

    def _on_ai_error(self, error_msg: str):
        """AI 搜索出错。"""
        self.ai_is_thinking = False
        self.ai_status_label.setText(f"AI 错误: {error_msg}")
        self.ai_status_label.setStyleSheet("color: red;")
        self.status_bar.showMessage(f"AI 错误: {error_msg}")

    def _execute_ai_move(self):
        """执行 AI 的走子。"""
        move = self.ai_best_move
        value = self.ai_best_value

        if move not in self.board.legal_moves:
            self.ai_is_thinking = False
            self.ai_status_label.setText("AI 状态: 错误 (非法走法)")
            self.ai_status_label.setStyleSheet("color: red;")
            return

        san = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san)

        self.board_widget.set_last_move(move)
        self.board_widget.set_board(self.board)

        self._update_history()
        self._update_title()

        self.ai_is_thinking = False
        self.ai_status_label.setText(f"AI 状态: 就绪 (估值: {value:+.2f})")
        self.ai_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_bar.showMessage(
            f"AI 走了: {san} (估值: {value:+.2f})"
        )

        # 检查对局是否结束
        if self.board.is_game_over():
            self._handle_game_end()

    def _handle_game_end(self):
        """处理对局结束。"""
        result = self.board.result()
        if self.board.is_checkmate():
            winner = "白方" if result == "1-0" else "黑方"
            msg = f"将杀！{winner}获胜！"
        elif self.board.is_stalemate():
            msg = "逼和！和棋。"
        elif self.board.is_insufficient_material():
            msg = "子力不足，和棋。"
        else:
            msg = f"对局结束: {result}"

        self.ai_status_label.setText(f"对局结束: {msg}")
        self.status_bar.showMessage(msg)
        QMessageBox.information(self, "对局结束", msg)

    def _new_game(self):
        """开始新对局。"""
        self.board = chess.Board()
        self.move_history = []
        self.ai_is_thinking = False
        self.ai_move_ready = False

        self.board_widget.set_board(self.board)
        self.board_widget.set_last_move(None)
        self.history_text.clear()

        self._update_title()
        self.ai_status_label.setText("AI 状态: 就绪")
        self.ai_status_label.setStyleSheet("color: green; font-weight: bold;")
        self.status_bar.showMessage("新对局已开始。")

        # 如果 AI 执白（玩家执黑），AI 先行
        if not self.player_is_white:
            self._ai_move()

    def _flip_board(self):
        """翻转棋盘视角。"""
        flipped = not self.board_widget.flip_board
        self.board_widget.set_flipped(flipped)

    def _on_color_changed(self):
        """玩家颜色切换。"""
        self.player_is_white = self.radio_white.isChecked()
        self._new_game()

    def _update_history(self):
        """更新走子历史显示。"""
        # 每行显示 2 步（白黑一对）
        lines = []
        for i in range(0, len(self.move_history), 2):
            move_num = i // 2 + 1
            line = f"{move_num}. {self.move_history[i]}"
            if i + 1 < len(self.move_history):
                line += f"  {self.move_history[i+1]}"
            lines.append(line)

        self.history_text.setPlainText("\n".join(lines))
        # 滚动到底部
        scrollbar = self.history_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _update_title(self):
        """更新窗口标题。"""
        turn = "白方" if self.board.turn == chess.WHITE else "黑方"
        result = ""
        if self.board.is_game_over():
            result = f" - {self.board.result()}"
        self.setWindowTitle(
            f"ChessMate - 轮到{turn}{result} | 步数: {self.board.fullmove_number}"
        )

    def closeEvent(self, event):
        """窗口关闭时，等待 AI 思考线程结束。"""
        if self.think_thread and self.think_thread.isRunning():
            self.think_thread.quit()
            self.think_thread.wait(1000)
        event.accept()


# ============================================================================
# 启动 GUI 的便捷函数
# ============================================================================

def launch_gui(model=None, config=None):
    """
    启动 ChessMate GUI。

    Args:
        model: ChessNet 模型。如果为 None，创建一个未训练的模型。
        config: ChessConfig 配置。如果为 None，使用默认配置。

    在 Windows 上，PyQt 需要注意 DPI 缩放。
    """
    from chessmate.config import ChessConfig
    from chessmate.training.neural_net import ChessNet

    if config is None:
        config = ChessConfig()

    if model is None:
        from chessmate.model_manager import select_model
        model = select_model(config)

    # 高 DPI 支持（Windows）
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 设置应用样式
    app.setStyle("Fusion")

    window = ChessWindow(model, config)
    window.show()

    # 如果 AI 先行
    if not config.gui_player_color:
        window._ai_move()

    sys.exit(app.exec_())


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    from chessmate.config import ChessConfig, get_small_config
    from chessmate.training.neural_net import ChessNet

    print("启动 ChessMate GUI 测试...")
    print("提示：关闭窗口即可退出测试。")

    cfg = get_small_config()
    cfg.num_res_blocks = 2
    cfg.num_filters = 16
    cfg.mcts_simulations = 50
    cfg.gui_player_color = True  # 玩家执白
    cfg.gui_ai_thinking_time = 0.5

    model = ChessNet(cfg)
    model.eval()

    launch_gui(model, cfg)