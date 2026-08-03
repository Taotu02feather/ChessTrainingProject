#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 经验回放缓冲区
========================
实现用于存储和采样训练样本的经验回放缓冲区。

每个经验样本包含：
- state: 棋盘状态编码 (num_planes, 8, 8)
- policy: MCTS 搜索得到的走法概率分布 (action_space_size,)
- value: 对局结果（+1 胜, -1 负, 0 和）

回放缓冲区使用循环缓冲区策略：
- 当缓冲区满时，最旧的样本被替换
- 支持随机采样（均匀分布），打破样本间的相关性
- 支持优先经验回放（可选扩展）
"""

import sys
import os
import random
import pickle
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch


# ============================================================================
# 经验样本数据结构
# ============================================================================

@dataclass
class Experience:
    """
    单个训练经验样本。

    属性:
        state_encoded: 棋盘编码张量 (num_planes, 8, 8)
        policy_target: MCTS 策略目标分布 (action_space_size,)
        value_target: 实际对局结果 [-1, 1]
    """
    state_encoded: torch.Tensor
    policy_target: np.ndarray
    value_target: float

    def to_dict(self) -> dict:
        """将经验样本转为可序列化的字典。"""
        return {
            'state': self.state_encoded.cpu().numpy(),
            'policy': self.policy_target,
            'value': self.value_target,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Experience':
        """从字典恢复经验样本。"""
        return cls(
            state_encoded=torch.from_numpy(data['state']),
            policy_target=np.array(data['policy'], dtype=np.float32),
            value_target=float(data['value']),
        )


# ============================================================================
# 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """
    经验回放缓冲区。

    使用 deque 实现固定容量的循环缓冲区。
    支持随机批量采样，用于神经网络训练。

    使用方式：
        buffer = ReplayBuffer(capacity=100000)
        buffer.add(state, policy, value)       # 添加单条经验
        buffer.add_batch(experiences)           # 批量添加
        batch = buffer.sample(batch_size=256)   # 随机采样
    """

    def __init__(self, capacity: int, config=None):
        """
        初始化回放缓冲区。

        Args:
            capacity: 缓冲区最大容量（样本数）。
            config: ChessConfig 对象（可选，用于获取设备信息）。
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)  # 自动管理容量的双端队列
        self.config = config

        # 统计信息
        self.total_added = 0  # 累计添加的样本数
        self.total_removed = 0  # 累计移除的样本数

    def add(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        value: float,
    ):
        """
        添加单条经验到缓冲区。

        Args:
            state: 棋盘状态编码 (num_planes, 8, 8)。
            policy: MCTS 策略分布 (action_space_size,)。
            value: 对局结果 [-1, 1]。
        """
        exp = Experience(
            state_encoded=state.cpu().clone(),
            policy_target=policy.copy(),
            value_target=value,
        )
        self._add_single(exp)

    def _add_single(self, exp: Experience):
        """内部添加方法，用于统一统计。"""
        if len(self.buffer) >= self.capacity:
            self.total_removed += 1
        self.buffer.append(exp)
        self.total_added += 1

    def add_batch(
        self,
        states: List[torch.Tensor],
        policies: List[np.ndarray],
        values: List[float],
    ):
        """
        批量添加经验到缓冲区。

        Args:
            states: 棋盘状态编码列表。
            policies: MCTS 策略分布列表。
            values: 对局结果列表。
        """
        for s, p, v in zip(states, policies, values):
            self.add(s, p, v)

    def add_game_experiences(self, game_experiences: List[Tuple[torch.Tensor, np.ndarray, float]]):
        """
        添加一整局棋的所有经验。

        Args:
            game_experiences: [(state, policy, value), ...] 列表。
                其中 value 已经是该玩家的对局结果。
        """
        self.add_batch(
            [exp[0] for exp in game_experiences],
            [exp[1] for exp in game_experiences],
            [exp[2] for exp in game_experiences],
        )

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从缓冲区随机采样一批经验。

        Args:
            batch_size: 采样批量大小。

        Returns:
            (states, policies, values) 三元组：
            - states: (B, num_planes, 8, 8)
            - policies: (B, action_space_size)
            - values: (B, 1)
        """
        batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(self.buffer, batch_size)

        states = torch.stack([exp.state_encoded for exp in batch])
        policies = torch.from_numpy(
            np.stack([exp.policy_target for exp in batch])
        ).float()
        values = torch.tensor(
            [[exp.value_target] for exp in batch], dtype=torch.float32
        )

        return states, policies, values

    def sample_numpy(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        采样一批经验并返回 numpy 数组（可用于非 PyTorch 的训练器）。

        Args:
            batch_size: 采样批量大小。

        Returns:
            (states, policies, values) 三元组（均为 numpy 数组）。
        """
        states, policies, values = self.sample(batch_size)
        return (
            states.cpu().numpy(),
            policies.cpu().numpy(),
            values.cpu().numpy(),
        )

    def get_recent(self, n: int) -> List[Experience]:
        """
        获取最近 n 条经验（用于最近经验优先回放）。

        Args:
            n: 要获取的最近经验数量。

        Returns:
            最近的 n 条经验列表。
        """
        n = min(n, len(self.buffer))
        return list(self.buffer)[-n:]

    def save(self, filepath: str):
        """
        将回放缓冲区保存到文件。

        Args:
            filepath: 保存路径（建议使用 .pkl 扩展名）。
        """
        data = {
            'experiences': [exp.to_dict() for exp in self.buffer],
            'total_added': self.total_added,
            'total_removed': self.total_removed,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

    def load(self, filepath: str):
        """
        从文件加载回放缓冲区。

        Args:
            filepath: 数据文件路径。
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        self.buffer.clear()
        for exp_dict in data['experiences']:
            self.buffer.append(Experience.from_dict(exp_dict))

        self.total_added = data.get('total_added', len(self.buffer))
        self.total_removed = data.get('total_removed', 0)

    def clear(self):
        """清空缓冲区。"""
        self.buffer.clear()
        self.total_added = 0
        self.total_removed = 0

    def __len__(self) -> int:
        """返回缓冲区中当前的样本数。"""
        return len(self.buffer)

    def __repr__(self) -> str:
        """缓冲区状态字符串。"""
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"current_size={len(self)}, "
            f"total_added={self.total_added}, "
            f"total_removed={self.total_removed})"
        )

    @property
    def is_ready(self) -> bool:
        """缓冲区是否已准备好用于训练（样本数 >= 批量大小）。"""
        if self.config:
            return len(self) >= self.config.replay_batch_size
        return len(self) >= 32  # 默认最小批量


# ============================================================================
# 游戏经验收集器（辅助工具）
# ============================================================================

class GameCollector:
    """
    用于收集单局游戏中所有步骤的经验样本。

    在自对弈过程中，每一步都会产生 (state, policy) 对，
    对局结束后统一赋上对局结果 value。

    使用方式：
        collector = GameCollector()
        collector.add_step(state, policy)   # 每步添加
        ...
        experiences = collector.finalize(result)  # 对局结束，标注结果
    """

    def __init__(self):
        self.states: List[torch.Tensor] = []
        self.policies: List[np.ndarray] = []
        self.player_turns: List[bool] = []  # True = 白方, False = 黑方

    def add_step(
        self,
        state: torch.Tensor,
        policy: np.ndarray,
        is_white: bool,
    ):
        """
        添加一步棋的经验数据。

        Args:
            state: 编码后的棋盘状态。
            policy: MCTS 策略分布。
            is_white: 当前玩家是否为白方。
        """
        self.states.append(state.cpu().clone())
        self.policies.append(policy.copy())
        self.player_turns.append(is_white)

    def finalize(self, result: str) -> List[Tuple[torch.Tensor, np.ndarray, float]]:
        """
        对局结束，为所有步骤赋上结果值。

        Args:
            result: 对局结果字符串。
                "1-0": 白方获胜
                "0-1": 黑方获胜
                "1/2-1/2": 和棋
                "*": 未结束（异常状态，视为和棋）

        Returns:
            [(state, policy, value), ...] 经验列表。
            每个样本的 value 是从对应玩家视角的结果：
            +1.0 = 胜利, -1.0 = 失败, 0.0 = 和棋
        """
        # 解析对局结果
        if result == "1-0":
            winner = "white"
        elif result == "0-1":
            winner = "black"
        else:
            winner = "draw"

        experiences = []
        for state, policy, is_white in zip(self.states, self.policies, self.player_turns):
            if winner == "draw":
                value = 0.0
            elif (winner == "white" and is_white) or (winner == "black" and not is_white):
                value = 1.0  # 当前玩家获胜
            else:
                value = -1.0  # 当前玩家失败

            experiences.append((state, policy, value))

        return experiences

    def clear(self):
        """清空收集器，准备下一局。"""
        self.states.clear()
        self.policies.clear()
        self.player_turns.clear()

    def __len__(self) -> int:
        """返回已收集的步数。"""
        return len(self.states)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import chess
    from chessmate.config import ChessConfig
    from chessmate.training.neural_net import BoardEncoder

    print("测试经验回放缓冲区...")

    cfg = ChessConfig()

    # 创建缓冲区
    buffer = ReplayBuffer(capacity=1000, config=cfg)

    # 创建编码器
    encoder = BoardEncoder()

    # 模拟添加经验
    board = chess.Board()
    for _ in range(50):
        state = encoder.encode(board)
        policy = np.random.rand(cfg.action_space_size).astype(np.float32)
        policy /= policy.sum()  # 归一化为概率分布
        value = random.uniform(-1, 1)
        buffer.add(state, policy, value)

    print(f"缓冲区状态: {buffer}")
    print(f"是否就绪: {buffer.is_ready}")

    # 采样测试
    states, policies, values = buffer.sample(16)
    print(f"\n批量采样:")
    print(f"  states 形状:  {states.shape}")
    print(f"  policies 形状: {policies.shape}")
    print(f"  values 形状:  {values.shape}")
    print(f"  values 范围: [{values.min().item():.3f}, {values.max().item():.3f}]")

    # 测试 GameCollector
    print("\n测试 GameCollector...")
    collector = GameCollector()
    board = chess.Board()

    for i in range(10):
        state = encoder.encode(board)
        policy = np.random.rand(cfg.action_space_size).astype(np.float32)
        policy /= policy.sum()
        collector.add_step(state, policy, board.turn == chess.WHITE)

        # 随机走一步
        move = random.choice(list(board.legal_moves))
        board.push(move)

    experiences = collector.finalize("1-0")
    print(f"收集了 {len(experiences)} 条经验")
    print(f"第一条经验的 value: {experiences[0][2]:.1f}")
    print(f"最后一条经验的 value: {experiences[-1][2]:.1f}")

    # 添加到缓冲区
    buffer.add_game_experiences(experiences)
    print(f"添加后的缓冲区大小: {len(buffer)}")

    print("\n经验回放缓冲区测试通过！")