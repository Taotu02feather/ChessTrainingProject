#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 神经网络定义
======================
实现 AlphaZero 风格的残差卷积神经网络，用于国际象棋局面评估。

网络架构（参考 AlphaZero 论文）：
  输入 (B, 119, 8, 8) ── 国际象棋的 119 个特征平面
    │
  卷积层 (num_filters, 3x3, stride=1)
    │
  N 个残差块 (ResBlock)
    │
    ├──> 策略头 (Policy Head) ── 走子概率分布 (B, 4672)
    │
    └──> 价值头 (Value Head)  ── 局面价值估计 (B, 1)

特性：
- 使用 Batch Normalization 稳定训练
- 残差连接帮助梯度传播
- 策略头输出所有可能走法的概率分布
- 价值头输出 [-1, 1] 范围的对局结果预测（tanh 激活）
"""

import sys
import os
import math
from typing import Tuple, List
from collections import deque

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 残差块 (Residual Block)
# ============================================================================

class ResBlock(nn.Module):
    """
    残差块 (Residual Block)。

    结构：
        Conv2d(3x3, same padding) -> BN -> ReLU ->
        Conv2d(3x3, same padding) -> BN -> 与输入相加 -> ReLU
    """

    def __init__(self, num_filters: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


# ============================================================================
# 棋盘状态编码器
# ============================================================================

class BoardEncoder:
    """
    将 python-chess 棋盘状态编码为神经网络可用的 119 平面特征张量。

    AlphaZero 的输入不是单一局面，而是"局面历史序列"：
    - 19 个基础平面：12 棋子 + 1 当前玩家颜色 + 4 易位权利 + 1 无进展计数器 + 1 常数
    - 时间维度：当前局面 + 过去 6 步历史 = 7 × 19 = 133，截断为 119（取最近 7 步 × 19，重复最旧步填充余数）

    平面布局（119 个平面）：
        [0~18]   当前局面 (T=0)
        [19~37]  1 步前历史 (T=-1)
        [38~56]  2 步前历史 (T=-2)
        [57~75]  3 步前历史 (T=-3)
        [76~94]  4 步前历史 (T=-4)
        [95~113] 5 步前历史 (T=-5)
        [114~118] 6 步前历史的前 5 个棋子平面 (T=-6 的棋子 0~4)
                  = 总数 119
    """

    BASE_PLANES = 19  # 单个时间步的平面数

    # 平面索引映射（在 encode 时动态使用）
    PIECE_OFFSET = 0
    COLOR_OFFSET = 12
    CASTLING_OFFSET = 13
    HALFMOVE_OFFSET = 17
    CONSTANT_OFFSET = 18

    # 棋子类型 + 颜色 → 平面偏移（在 19 平面组内的偏移）
    PIECE_PLANE_MAP = {
        (chess.PAWN, True):   0, (chess.KNIGHT, True):   1, (chess.BISHOP, True):   2,
        (chess.ROOK, True):   3, (chess.QUEEN,  True):   4, (chess.KING,   True):   5,
        (chess.PAWN, False):  6, (chess.KNIGHT, False):  7, (chess.BISHOP, False):  8,
        (chess.ROOK, False):  9, (chess.QUEEN,  False):  10, (chess.KING,  False):  11,
    }

    def __init__(self, history_length: int = 8):
        """
        初始化编码器。

        Args:
            history_length: 保留的历史局面 FEN 数。默认 8（= T=0 + T=-1 ~ T=-7）。
        """
        self.history_length = history_length
        self.num_planes = 119  # AlphaZero 标准
        # 存储最近 N 步的 FEN，用于重建历史局面
        self.history_stack = deque(maxlen=history_length)

    def push_history(self, board):
        """记录一步历史局面。每走一步棋后调用。"""
        self.history_stack.append(board.fen())

    def clear_history(self):
        """清空历史记录（新对局/新搜索开始时调用）。"""
        self.history_stack.clear()

    def _encode_time_step(self, board, planes, base_idx):
        """
        填充一个时间步的 19 个平面到预分配的 planes 数组。

        Args:
            board: python-chess Board 对象。
            planes: (119, 8, 8) numpy 数组。
            base_idx: 此时间步的起始平面索引。
        """
        import numpy as np

        # 棋子位置 (12 平面)
        for square, piece in board.piece_map().items():
            row = 7 - (square // 8)
            col = square % 8
            offset = self.PIECE_PLANE_MAP.get((piece.piece_type, piece.color == chess.WHITE), -1)
            if offset >= 0:
                planes[base_idx + offset, row, col] = 1.0

        # 当前玩家颜色
        planes[base_idx + self.COLOR_OFFSET] = 1.0 if board.turn == chess.WHITE else 0.0

        # 易位权利 (4 平面)
        planes[base_idx + self.CASTLING_OFFSET + 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[base_idx + self.CASTLING_OFFSET + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[base_idx + self.CASTLING_OFFSET + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[base_idx + self.CASTLING_OFFSET + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

        # 无进展计数器
        planes[base_idx + self.HALFMOVE_OFFSET] = board.halfmove_clock / 50.0

        # 常数平面
        planes[base_idx + self.CONSTANT_OFFSET] = 1.0

    def encode(self, board) -> torch.Tensor:
        """
        将 python-chess Board 编码为 119 平面特征张量（含历史）。

        从 history_stack 读取过去局面填充 T=-1 到 T=-6，
        不足的时间步用当前局面填充。

        Args:
            board: python-chess Board 对象。

        Returns:
            形状为 (num_planes, 8, 8) 的 float32 张量。
        """
        import numpy as np

        planes = np.zeros((self.num_planes, 8, 8), dtype=np.float32)

        # T=0：当前局面
        self._encode_time_step(board, planes, 0)

        # 获取历史 FEN 列表（最近在前）
        history = list(self.history_stack)

        # 填充 T=-1 到 T=-5（每个时间步 19 平面，共填充 5 步 = 95 平面，索引 19~113）
        for t in range(1, 6):
            base_idx = t * self.BASE_PLANES
            hist_idx = len(history) - 1 - (t - 1)
            if hist_idx >= 0:
                hist_board = chess.Board(history[hist_idx])
                self._encode_time_step(hist_board, planes, base_idx)
            else:
                # 用当前局面填充缺失的历史
                self._encode_time_step(board, planes, base_idx)

        # 最后 5 个平面 (114~118) 用 T=-6 或当前局面的棋子平面 0~4 填充
        last_step_base = 6 * self.BASE_PLANES
        hist_idx = len(history) - 1 - 5
        if hist_idx >= 0:
            hist_board = chess.Board(history[hist_idx])
            # 只填棋子 0~4
            for square, piece in hist_board.piece_map().items():
                row = 7 - (square // 8)
                col = square % 8
                offset = self.PIECE_PLANE_MAP.get((piece.piece_type, piece.color == chess.WHITE), -1)
                if 0 <= offset < 5:
                    planes[114 + offset, row, col] = 1.0
        else:
            planes[114:119] = planes[0:5]

        return torch.from_numpy(planes).float()

    def encode_batch(self, boards: list) -> torch.Tensor:
        """批量编码多个棋盘状态。"""
        batch = [self.encode(b) for b in boards]
        return torch.stack(batch, dim=0)


# ============================================================================
# 象棋神经网络 (ChessNet)
# ============================================================================

class ChessNet(nn.Module):
    """
    国际象棋神经网络 - AlphaZero 风格的双头架构。

    输入: (B, num_planes, 8, 8)
    输出: (policy_logits, value)
    """

    def __init__(self, config):
        super(ChessNet, self).__init__()
        self.config = config
        num_planes = config.num_planes
        num_filters = config.num_filters
        num_res_blocks = config.num_res_blocks

        self.conv_input = nn.Sequential(
            nn.Conv2d(num_planes, num_filters, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(num_filters) for _ in range(num_res_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, config.action_space_size),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, config.value_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.value_head_hidden, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_input(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def predict(self, board_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(board_encoded)
            policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, value

    def save(self, filepath: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'num_planes': self.config.num_planes,
                'num_filters': self.config.num_filters,
                'num_res_blocks': self.config.num_res_blocks,
                'action_space_size': self.config.action_space_size,
                'value_head_hidden': self.config.value_head_hidden,
            }
        }, filepath)

    @classmethod
    def load(cls, filepath: str, config=None, device: str = 'cpu'):
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        saved_config = checkpoint.get('config', {})
        if config is None:
            from chessmate.config import ChessConfig
            config = ChessConfig()
            for k, v in saved_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model


# ============================================================================
# 走法映射
# ============================================================================

def move_to_index(move) -> int:
    from_sq = move.from_square
    to_sq = move.to_square
    base_index = from_sq * 64 + to_sq
    if move.promotion:
        promotion_offset = {1: 1, 2: 2, 3: 3, 4: 4}
        offset = promotion_offset.get(move.promotion, 0)
        base_index = 4096 + (from_sq * 4 + offset)
    return min(base_index, 4671)


def index_to_move(index: int, board) -> any:
    if index < 4096:
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in [0, 7]:
            move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    else:
        offset = index - 4096
        from_sq = offset // 4
        prom_type = offset % 4
        promotion_map = {0: chess.QUEEN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK}
        prom = promotion_map.get(prom_type, chess.QUEEN)
        to_sq = from_sq + 8
        move = chess.Move(from_sq, to_sq, promotion=prom)
    if move in board.legal_moves:
        return move
    return None


# ============================================================================
# 损失函数
# ============================================================================

class ChessLoss(nn.Module):
    def __init__(self):
        super(ChessLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, policy_pred, value_pred, policy_target, value_target):
        policy_loss = self.cross_entropy(policy_pred, policy_target)
        value_loss = self.mse(value_pred, value_target)
        return policy_loss + value_loss, policy_loss, value_loss


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from chessmate.config import ChessConfig

    print("测试 ChessNet 网络构建和前向传播...")

    cfg = ChessConfig()
    cfg.num_res_blocks = 2
    cfg.num_filters = 32
    cfg.device = "cpu"

    net = ChessNet(cfg)
    print(f"网络参数量: {sum(p.numel() for p in net.parameters()):,}")

    encoder = BoardEncoder(history_length=8)
    board = chess.Board()
    encoded = encoder.encode(board).unsqueeze(0)
    policy, value = net(encoded)
    print(f"策略输出: {policy.shape}, 价值输出: {value.shape}, 价值={value.item():.4f}")

    # 测试走法映射
    for move in list(board.legal_moves)[:1]:
        idx = move_to_index(move)
        rec = index_to_move(idx, board)
        print(f"  走法: {move} -> 索引: {idx} -> 还原: {rec}")

    print("\n网络构建和前向传播测试通过！")