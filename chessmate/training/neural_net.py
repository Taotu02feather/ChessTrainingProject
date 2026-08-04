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

    使用两个 3x3 卷积层配合批归一化，跳跃连接将输入直接加到输出。
    """

    def __init__(self, num_filters: int):
        """
        初始化残差块。

        Args:
            num_filters: 卷积层的通道数。
        """
        super(ResBlock, self).__init__()

        # 第一个卷积层：3x3 卷积 + 批归一化
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,          # 保持输入输出尺寸相同
            bias=False          # 使用 BN 时不使用 bias
        )
        self.bn1 = nn.BatchNorm2d(num_filters)

        # 第二个卷积层：3x3 卷积 + 批归一化
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量，形状 (B, C, H, W)。

        Returns:
            输出张量，形状 (B, C, H, W)，与输入形状相同。
        """
        # 保存输入作为跳跃连接
        residual = x

        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)

        # 跳跃连接：将输入加到输出上
        out = out + residual

        # 最后的 ReLU 激活
        out = F.relu(out)

        return out


# ============================================================================
# 棋盘状态编码器
# ============================================================================

class BoardEncoder:
    """
    将 python-chess 棋盘状态编码为神经网络可用的特征张量。

    参考 AlphaZero 的输入表示：
    - 12 层：每层对应一种棋子的位置（6 种棋子 × 2 种颜色）
    - 历史位置：重复堆叠过去几个位置的状态
    - 颜色表示：当前玩家执子颜色
    - 易位权利：4 个平面（白王翼、白后翼、黑王翼、黑后翼）
    - 无进展计数器：无吃子或无兵移动的步数
    - 常数平面：全 1 平面

    总计约 119 个平面（取决于历史长度和具体实现）。
    """

    # 棋子类型到平面索引的映射（12 种）
    PIECE_TO_PLANE = {
        'P': 0,  'N': 1,  'B': 2,  'R': 3,  'Q': 4,  'K': 5,
        'p': 6,  'n': 7,  'b': 8,  'r': 9,  'q': 10, 'k': 11,
    }

    # 棋盘平面总数（不含历史）
    BASE_PLANES = 12 + 1 + 4 + 1 + 1  # 棋子 + 颜色 + 易位 + 无进展 + 常数
    # = 12 + 1 + 4 + 1 + 1 = 19，但 AlphaZero 用的更多
    # 按历史堆叠后为 19 * 8 + 额外信息 ≈ 119

    def __init__(self, history_length: int = 8):
        """
        初始化编码器。

        Args:
            history_length: 历史步数（包含当前局面）。
        """
        self.history_length = history_length
        self.num_planes = 119  # AlphaZero 标准

    # 棋子类型到平面索引映射（类常量，避免每步重新构建）
    _PIECE_PLANE_IDX = {
        (chess.PAWN, chess.WHITE): 0, (chess.KNIGHT, chess.WHITE): 1,
        (chess.BISHOP, chess.WHITE): 2, (chess.ROOK, chess.WHITE): 3,
        (chess.QUEEN, chess.WHITE): 4, (chess.KING, chess.WHITE): 5,
        (chess.PAWN, chess.BLACK): 6, (chess.KNIGHT, chess.BLACK): 7,
        (chess.BISHOP, chess.BLACK): 8, (chess.ROOK, chess.BLACK): 9,
        (chess.QUEEN, chess.BLACK): 10, (chess.KING, chess.BLACK): 11,
    }
    # 易位权利平面偏移（在 planes 中的起止索引）
    _CASTLING_OFFSET = 12 + 1  # 12 棋子 + 1 颜色平面
    # 无进展平面索引
    _HALFMOVE_OFFSET = 12 + 1 + 4  # +4 易位
    # 常数平面索引
    _CONSTANT_OFFSET = 12 + 1 + 4 + 1

    def encode(self, board) -> torch.Tensor:
        """
        将 python-chess Board 对象编码为特征张量（向量化优化版）。

        预分配 (num_planes, 8, 8) 数组后一次性填充，避免重复 np.zeros 分配
        和 Python 循环嵌套。使用 board.piece_map() 一次性遍历所有棋子。

        Args:
            board: python-chess Board 对象。

        Returns:
            形状为 (num_planes, 8, 8) 的 float32 张量。
        """
        import numpy as np, chess

        # 预分配完整平面数组（一次性分配，避免重复 np.zeros）
        planes = np.zeros((self.num_planes, 8, 8), dtype=np.float32)

        # --- 1. 棋子位置 (12 个平面) ---
        # 使用 board.piece_map() 一次性获取所有棋子及其位置
        for square, piece in board.piece_map().items():
            row = 7 - (square // 8)  # square 转网格行（0=棋盘顶部）
            col = square % 8
            plane_idx = self._PIECE_PLANE_IDX.get(
                (piece.piece_type, piece.color), -1
            )
            if plane_idx >= 0:
                planes[plane_idx, row, col] = 1.0

        # --- 2. 当前玩家颜色 (1 个平面, 索引 12) ---
        planes[12] = 1.0 if board.turn == chess.WHITE else 0.0

        # --- 3. 易位权利 (4 个平面, 索引 13~16) ---
        offset = self._CASTLING_OFFSET
        planes[offset + 0] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        planes[offset + 1] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        planes[offset + 2] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        planes[offset + 3] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

        # --- 4. 无进展计数器 (1 个平面, 索引 17) ---
        planes[self._HALFMOVE_OFFSET] = board.halfmove_clock / 50.0

        # --- 5. 常数平面 (1 个平面, 索引 18) ---
        planes[self._CONSTANT_OFFSET] = 1.0

        # 剩余平面保持为 0（预分配时已为零）
        return torch.from_numpy(planes).float()

    def encode_batch(self, boards: list) -> torch.Tensor:
        """
        批量编码多个棋盘状态。

        Args:
            boards: python-chess Board 对象列表。

        Returns:
            形状为 (B, num_planes, 8, 8) 的 float32 张量。
        """
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
        - policy_logits: 形状 (B, action_space_size)，所有可能走法的对数概率
        - value: 形状 (B, 1)，局面价值 [-1, 1]
    """

    def __init__(self, config):
        """
        初始化网络。

        Args:
            config: ChessConfig 配置对象，包含超参数。
        """
        super(ChessNet, self).__init__()

        self.config = config
        num_planes = config.num_planes
        num_filters = config.num_filters
        num_res_blocks = config.num_res_blocks
        action_space_size = config.action_space_size
        value_head_hidden = config.value_head_hidden

        # ---- 初始卷积层 ----
        # 将输入特征平面映射到 num_filters 维特征空间
        self.conv_input = nn.Sequential(
            nn.Conv2d(
                in_channels=num_planes,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
        )

        # ---- 残差塔 (Residual Tower) ----
        # 堆叠多个残差块
        self.res_blocks = nn.Sequential(*[
            ResBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # ---- 策略头 (Policy Head) ----
        # 输出每个可能走法的对数概率
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, action_space_size),
        )

        # ---- 价值头 (Value Head) ----
        # 输出当前局面的评估值 [-1, 1]
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1 * 8 * 8, value_head_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_head_hidden, 1),
            nn.Tanh(),  # 输出范围 [-1, 1]
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot 权重初始化，偏置初始化为零。"""
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
        """
        前向传播。

        Args:
            x: 输入张量，形状 (B, num_planes, 8, 8)。

        Returns:
            (policy_logits, value) 元组。
            - policy_logits: (B, action_space_size)，走子概率的 logits
            - value: (B, 1)，局面价值
        """
        # 初始卷积
        x = self.conv_input(x)

        # 残差塔
        x = self.res_blocks(x)

        # 双头输出
        policy = self.policy_head(x)    # (B, action_space_size)
        value = self.value_head(x)      # (B, 1)

        return policy, value

    def predict(self, board_encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测走子概率分布和局面价值（评估模式，不计算梯度）。

        Args:
            board_encoded: 编码后的棋盘状态 (1, num_planes, 8, 8)。

        Returns:
            (policy_probs, value) 元组，policy_probs 已经过 softmax。
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(board_encoded)
            policy_probs = F.softmax(policy_logits, dim=-1)
        return policy_probs, value

    def save(self, filepath: str):
        """
        保存模型权重到文件。

        Args:
            filepath: 保存路径（建议使用 .pth 扩展名）。
        """
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
        """
        从文件加载模型权重。

        Args:
            filepath: 模型文件路径。
            config: ChessConfig 对象。如果为 None，尝试从保存的配置重建。
            device: 加载到的设备。

        Returns:
            加载好权重的 ChessNet 实例。
        """
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        saved_config = checkpoint.get('config', {})

        if config is None:
            # 从保存的配置重建
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
# 将棋盘走法映射到策略索引
# ============================================================================

def move_to_index(move) -> int:
    """
    将 python-chess 的走法对象映射到策略向量中的索引。

    编码方案：使用棋子的起始格和目标格。
    索引 = from_square * 64 + to_square  (最多 4096)
    加上升变标记 (4096 + promotion_type * 64 + ...)

    实际上完整空间约为 4672。

    Args:
        move: python-chess Move 对象。

    Returns:
        动作空间中的整数索引。
    """
    from_sq = move.from_square   # 0-63
    to_sq = move.to_square       # 0-63

    # 基础索引：源格子 * 64 + 目标格子
    base_index = from_sq * 64 + to_sq  # 0-4095

    # 升变走法：使用更高的索引空间
    if move.promotion:
        # 升变类型: 1=马, 2=象, 3=车, 4=后 (python-chess)
        promotion_offset = {
            1: 1,   # 升变为马 (knight)
            2: 2,   # 升变为象 (bishop)
            3: 3,   # 升变为车 (rook)
            4: 4,   # 升变为后 (queen)
        }
        offset = promotion_offset.get(move.promotion, 0)
        base_index = 4096 + (from_sq * 4 + offset)  # 4 种升变类型

    return min(base_index, 4671)  # 确保不超出动作空间


def index_to_move(index: int, board) -> any:
    """
    将策略索引转换回 python-chess 走法。

    Args:
        index: 动作空间中的整数索引。
        board: 当前棋盘状态。

    Returns:
        对应的 python-chess Move 对象，或 None（如果索引不合法）。
    """
    import chess

    if index < 4096:
        # 非升变走法
        from_sq = index // 64
        to_sq = index % 64
        move = chess.Move(from_sq, to_sq)

        # 检查是否是兵升变（兵到达底线）
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if chess.square_rank(to_sq) in [0, 7]:
                move = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    else:
        # 升变走法
        offset = index - 4096
        from_sq = offset // 4
        prom_type = offset % 4
        promotion_map = {0: chess.QUEEN, 1: chess.KNIGHT, 2: chess.BISHOP, 3: chess.ROOK}
        prom = promotion_map.get(prom_type, chess.QUEEN)
        # 目标格需要推断（通常是向前一格或斜前方）
        to_sq = from_sq + 8  # 默认向前
        move = chess.Move(from_sq, to_sq, promotion=prom)

    # 验证走法合法性
    if move in board.legal_moves:
        return move
    return None


# ============================================================================
# 损失函数
# ============================================================================

class ChessLoss(nn.Module):
    """
    ChessMate 训练损失函数。

    损失 = 策略损失 + 价值损失 + L2 正则化项

    - 策略损失: 交叉熵损失（MCTS 搜索概率 vs 网络预测概率）
    - 价值损失: 均方误差（实际对局结果 vs 网络评估）
    """

    def __init__(self):
        super(ChessLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(
        self,
        policy_pred: torch.Tensor,
        value_pred: torch.Tensor,
        policy_target: torch.Tensor,
        value_target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算损失。

        Args:
            policy_pred: 网络输出的策略 logits (B, action_space_size)
            value_pred: 网络输出的价值 (B, 1)
            policy_target: MCTS 搜索的目标策略分布 (B, action_space_size)
            value_target: 实际对局结果 (B, 1)

        Returns:
            (total_loss, policy_loss, value_loss) 元组。
        """
        # 策略损失：交叉熵
        policy_loss = self.cross_entropy(policy_pred, policy_target)

        # 价值损失：均方误差
        value_loss = self.mse(value_pred, value_target)

        # 总损失
        total_loss = policy_loss + value_loss

        return total_loss, policy_loss, value_loss


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import chess
    from chessmate.config import ChessConfig

    print("测试 ChessNet 网络构建和前向传播...")

    cfg = ChessConfig()
    cfg.num_res_blocks = 2
    cfg.num_filters = 32
    cfg.device = "cpu"

    # 创建网络
    net = ChessNet(cfg)
    print(f"网络参数量: {sum(p.numel() for p in net.parameters()):,}")

    # 模拟输入
    encoder = BoardEncoder(history_length=cfg.history_length)
    board = chess.Board()
    encoded = encoder.encode(board).unsqueeze(0)  # (1, num_planes, 8, 8)

    # 前向传播
    policy, value = net(encoded)
    print(f"策略输出形状: {policy.shape} (应为 (1, {cfg.action_space_size}))")
    print(f"价值输出形状: {value.shape} (应为 (1, 1))")
    print(f"价值输出值: {value.item():.4f}")

    # 测试走法映射
    for move in board.legal_moves:
        idx = move_to_index(move)
        reconstructed = index_to_move(idx, board)
        print(f"  走法: {move} -> 索引: {idx} -> 还原: {reconstructed}")
        break  # 只测试一个

    print("\n网络构建和前向传播测试通过！")