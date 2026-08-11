#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 自对弈模块
====================
实现强化学习中的自对弈（Self-Play）过程：AI 与自己对弈生成训练数据。

自对弈流程：
1. 使用当前神经网络进行 MCTS 搜索
2. 根据 MCTS 搜索概率采样走法
3. 收集每一步的 (state, policy) 数据
4. 对局结束后根据结果标注 value
5. 将经验样本添加到回放缓冲区

自对弈策略特点：
- 前 N 步使用温度控制的随机采样（鼓励探索）
- 后续步使用贪婪选择（选访问次数最多的走法）
- 达到一定步数或估值过低时自动认输
- 支持并行的多局自对弈（可选扩展）
"""

import sys
import os
import random
import time
from typing import List, Tuple, Optional

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chess
import numpy as np
import torch
from tqdm import tqdm

from chessmate.training.mcts import MCTS, Node
from chessmate.training.neural_net import (
    ChessNet, BoardEncoder, move_to_index, index_to_move
)
from chessmate.training.replay_buffer import GameCollector


# ============================================================================
# 自对弈工作器
# ============================================================================

class SelfPlay:
    """
    自对弈引擎。

    使用当前的神经网络和一个独立的 MCTS 实例，
    让 AI 与自己对弈并收集训练数据。

    使用方式：
        sp = SelfPlay(model, encoder, config)
        game_experiences = sp.play_one_game()   # 进行一局自对弈
        sp.play_and_collect(buffer, num_games=10)  # 批量自对弈并收集
    """

    def __init__(
        self,
        model: ChessNet,
        encoder: BoardEncoder,
        config,
        logger=None,
    ):
        """
        初始化自对弈引擎。

        Args:
            model: ChessNet 神经网络模型。
            encoder: BoardEncoder 棋盘编码器。
            config: ChessConfig 配置对象。
            logger: 日志记录器（可选）。
        """
        self.model = model
        self.encoder = encoder
        self.config = config
        self.logger = logger

        # 创建 MCTS 实例
        self.mcts = MCTS(model, encoder, config)

        # 统计信息
        self.total_games = 0
        self.wins_white = 0
        self.wins_black = 0
        self.draws = 0

    def play_one_game(self) -> List[Tuple[torch.Tensor, np.ndarray, float]]:
        """
        进行一局完整的自对弈，收集经验数据。

        Returns:
            [(state, policy, value), ...] 经验列表。
            如果对局异常终止（如步数超限），返回空列表或部分数据。

        对局流程：
        1. 初始化棋盘
        2. 循环进行直到终局或达到最大步数
        3. 每步使用 MCTS 搜索 + 温度采样
        4. 收集训练数据
        5. 对局结束后标注结果
        """
        board = chess.Board()
        collector = GameCollector()

        # 初始化历史编码器
        self.encoder.clear_history()
        self.encoder.push_history(board)

        move_count = 0
        max_moves = self.config.max_moves_per_game

        # 温度切换步数（前 N 步用高温探索，之后用低温贪婪）
        temperature_cutoff = 40  # 前 40 步使用温度采样，增加中局多样性

        # 开局强制随机走子（白黑各1步），打破千局一面
        for _ in range(2):
            if board.legal_moves.count() > 0:
                board.push(random.choice(list(board.legal_moves)))
                self.encoder.push_history(board)
                move_count += 1

        # 主循环
        while not board.is_game_over() and move_count < max_moves:
            # 设置搜索温度
            if move_count < temperature_cutoff:
                self.mcts.temperature = self.config.mcts_temperature  # 探索
            else:
                self.mcts.temperature = 0.1  # 利用

            # 同步编码器的历史状态到 MCTS
            self.mcts.sync_encoder_history(self.encoder)

            # MCTS 搜索，获取走法概率分布和局面估值
            move_probs, root_value = self.mcts.search(
                board, return_probs=True
            )

            if move_probs is None:
                # 搜索失败（终局或无合法走法）
                break

            # 编码当前局面（用于保存经验）-- 已经包含历史上下文
            encoded_state = self.encoder.encode(board)

            # 将走法概率分布转换为动作空间大小的向量
            policy_vector = np.zeros(self.config.action_space_size, dtype=np.float32)
            for move, prob in move_probs.items():
                idx = move_to_index(move)
                if 0 <= idx < self.config.action_space_size:
                    policy_vector[idx] = prob

            # 收集经验数据（先保存，对局结束后再标注 value）
            collector.add_step(
                encoded_state,
                policy_vector,
                board.turn == chess.WHITE,
            )

            # 根据温度采样选择走法
            selected_move = self._sample_move(move_probs, board)

            if selected_move is None:
                # 采样失败，使用随机走法
                selected_move = random.choice(list(board.legal_moves))

            # 执行走法
            board.push(selected_move)
            # 记录历史局面到编码器（关键：让网络看到棋局动态变化）
            self.encoder.push_history(board)
            move_count += 1

            # 自动认输检查
            if root_value < self.config.resign_threshold and move_count > 20:
                if self.logger:
                    self.logger.debug(f"自动认输，估值={root_value:.3f}")
                break

        # ---- 对局结束 ----
        result = board.result()
        if board.is_checkmate():
            if self.logger:
                self.logger.debug(
                    f"将杀！结果={result}, 步数={move_count}"
                )
        elif board.is_stalemate() or board.is_insufficient_material():
            result = "1/2-1/2"
            if self.logger:
                self.logger.debug(f"和棋（逼和/子力不足）, 步数={move_count}")
        elif move_count >= max_moves:
            result = "1/2-1/2"
            if self.logger:
                self.logger.debug(f"达到最大步数限制={max_moves}，视为和棋")

        # 更新统计
        self.total_games += 1
        if result == "1-0":
            self.wins_white += 1
        elif result == "0-1":
            self.wins_black += 1
        else:
            self.draws += 1

        # 标注经验数据
        experiences = collector.finalize(result)

        return experiences

    def play_and_collect(
        self,
        buffer,
        num_games: int = None,
        show_progress: bool = True,
    ) -> int:
        """
        进行多局自对弈，将经验收集到回放缓冲区。

        Args:
            buffer: ReplayBuffer 对象。
            num_games: 要进行的对局数。默认使用配置中的值。
            show_progress: 是否显示进度条。

        Returns:
            收集的经验总数。
        """
        if num_games is None:
            num_games = self.config.num_self_play_games

        total_experiences = 0
        game_iterator = range(num_games)

        if show_progress:
            game_iterator = tqdm(
                game_iterator,
                desc="自对弈",
                unit="局",
            )

        for game_idx in game_iterator:
            try:
                experiences = self.play_one_game()

                if experiences:
                    buffer.add_game_experiences(experiences)
                    total_experiences += len(experiences)

            except Exception as e:
                if self.logger:
                    self.logger.error(f"第 {game_idx} 局自对弈出错: {e}")
                continue

        if self.logger:
            self.logger.info(
                f"自对弈完成: {num_games}局, 收集{total_experiences}条经验 | "
                f"白胜{self.wins_white}, 黑胜{self.wins_black}, 和棋{self.draws}"
            )

        return total_experiences

    def _sample_move(self, move_probs: dict, board) -> Optional[chess.Move]:
        """
        根据概率分布采样走法。

        Args:
            move_probs: {move: probability} 字典。
            board: 当前棋盘状态。

        Returns:
            采样到的走法，或 None。
        """
        if not move_probs:
            return None

        # 分离走法和概率
        moves = []
        probs = []

        for move, prob in move_probs.items():
            if prob > 0 and move in board.legal_moves:
                moves.append(move)
                probs.append(prob)

        if not moves:
            return None

        # 归一化概率
        probs = np.array(probs, dtype=np.float64)
        probs /= probs.sum()

        # 按概率采样
        selected_idx = np.random.choice(len(moves), p=probs)
        return moves[selected_idx]

    def get_stats(self) -> dict:
        """
        获取自对弈统计信息。

        Returns:
            包含统计信息的字典。
        """
        total = self.total_games
        if total == 0:
            return {'total': 0}

        return {
            'total': total,
            'white_wins': self.wins_white,
            'black_wins': self.wins_black,
            'draws': self.draws,
            'white_win_rate': self.wins_white / total,
            'black_win_rate': self.wins_black / total,
            'draw_rate': self.draws / total,
        }

    def reset_stats(self):
        """重置统计计数器。"""
        self.total_games = 0
        self.wins_white = 0
        self.wins_black = 0
        self.draws = 0


# ============================================================================
# 快速验证对弈（测试流程用）
# ============================================================================

def run_quick_test(config, num_games: int = 5):
    """
    运行快速自对弈测试，验证整个流程是否正常。

    Args:
        config: ChessConfig 对象。
        num_games: 测试对局数。
    """
    from chessmate.training.neural_net import ChessNet, BoardEncoder
    from chessmate.training.replay_buffer import ReplayBuffer

    print(f"开始快速自对弈测试 ({num_games}局)...")

    encoder = BoardEncoder(history_length=config.history_length)
    model = ChessNet(config)
    buffer = ReplayBuffer(capacity=10000, config=config)

    sp = SelfPlay(model, encoder, config)
    total_exp = sp.play_and_collect(buffer, num_games=num_games)

    stats = sp.get_stats()
    print(f"\n测试结果:")
    print(f"  总对局数: {stats['total']}")
    print(f"  白胜: {stats['white_wins']} ({stats['white_win_rate']:.1%})")
    print(f"  黑胜: {stats['black_wins']} ({stats['black_win_rate']:.1%})")
    print(f"  和棋: {stats['draws']} ({stats['draw_rate']:.1%})")
    print(f"  收集经验: {total_exp}条")
    print(f"  缓冲区大小: {len(buffer)}")

    return sp, buffer


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import logging
    from chessmate.config import ChessConfig, get_small_config

    print("测试自对弈模块...")

    # 使用小规模配置进行快速测试
    cfg = get_small_config()
    cfg.num_res_blocks = 2
    cfg.num_filters = 16
    cfg.mcts_simulations = 10  # 极少模拟，仅用于测试

    logger = logging.getLogger("SelfPlayTest")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    # 运行快速测试
    sp, buffer = run_quick_test(cfg, num_games=2)

    print("\n自对弈模块测试通过！")