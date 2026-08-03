#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 蒙特卡洛树搜索 (MCTS)
=============================
实现 AlphaZero 风格的 MCTS 算法，使用神经网络引导搜索。

MCTS 的核心过程：
1. Selection（选择）：从根节点开始，按 PUCT 公式选择子节点直到叶节点
2. Expansion（扩展）：到达未完全展开的节点时，创建一个新子节点
3. Simulation/Evaluation（评估）：使用神经网络评估新节点的策略和价值
4. Backpropagation（回传）：将评估结果沿路径回传到根节点

PUCT 公式：
    U(s,a) = c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    Q(s,a) = W(s,a) / N(s,a)   ... 平均价值
    选择最大化的 a: Q(s,a) + U(s,a)

参考资料：
    Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm
    (Silver et al., 2017)
"""

import sys
import os
import math
import random
from typing import Tuple, List, Optional, Dict, Any

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F

from chessmate.training.neural_net import BoardEncoder, move_to_index, index_to_move


# ============================================================================
# MCTS 节点
# ============================================================================

class Node:
    """
    MCTS 树中的一个节点，代表一个棋盘状态。

    属性：
        state: 编码后的棋盘状态张量（用于神经网络输入）
        parent: 父节点（None 表示根节点）
        children: 子节点字典 {move_index: Node}
        visit_count: 该节点的访问次数 N
        total_value: 累计价值 W（所有访问的价值之和）
        prior_prob: 先验概率 P(s,a)（来自神经网络策略头）
        is_expanded: 是否已完全展开（所有合法走法都创建了子节点）
    """

    __slots__ = (
        'state', 'parent', 'children', 'visit_count',
        'total_value', 'prior_prob', 'prior_probs',
        'is_expanded', 'legal_moves_indices', 'board',
    )

    def __init__(
        self,
        state: torch.Tensor,
        parent: Optional['Node'] = None,
        prior_prob: float = 0.0,
        board=None,
    ):
        self.state = state
        self.parent = parent
        self.children: Dict[int, 'Node'] = {}
        self.visit_count = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
        self.prior_probs = None
        self.is_expanded = False
        self.legal_moves_indices: List[int] = []
        self.board = board

    def mean_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None


class MCTS:
    """AlphaZero 风格的蒙特卡洛树搜索。"""

    def __init__(self, model, encoder: BoardEncoder, config):
        self.model = model
        self.encoder = encoder
        self.config = config

        self.num_simulations = config.mcts_simulations
        self.c_puct = config.mcts_c_puct
        self.temperature = config.mcts_temperature
        self.dirichlet_alpha = config.mcts_dirichlet_alpha
        self.dirichlet_epsilon = config.mcts_dirichlet_epsilon

        self.device = config.device
        self.model.to(self.device)

    def search(self, board, return_probs: bool = False) -> Tuple[any, float]:
        import chess

        encoded = self.encoder.encode(board)
        encoded = encoded.unsqueeze(0).to(self.device)

        root = Node(state=encoded, board=board.copy())
        root.legal_moves_indices = self._get_legal_moves_indices(board)

        if not root.legal_moves_indices:
            if board.is_checkmate():
                return None, -1.0 if board.turn == chess.WHITE else 1.0
            else:
                return None, 0.0

        with torch.no_grad():
            policy_logits, value = self.model(root.state)
            policy_probs = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
            root_value = value.item()

        root.prior_probs = policy_probs
        self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_board = board.copy()

            while not node.is_leaf() and node.is_expanded:
                best_child_idx = self._select_child(node)
                node = node.children[best_child_idx]
                search_path.append(node)
                move = index_to_move(best_child_idx, current_board)
                if move:
                    current_board.push(move)

            value = 0.0
            if not current_board.is_game_over():
                node.legal_moves_indices = self._get_legal_moves_indices(current_board)

                if node.legal_moves_indices:
                    node.state = self.encoder.encode(current_board).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        policy_logits, nn_value = self.model(node.state)
                        node.prior_probs = F.softmax(policy_logits.squeeze(0), dim=0).cpu().numpy()
                        value = -nn_value.item()
                    node.is_expanded = True
                else:
                    result = current_board.result()
                    if result == "1-0":
                        value = 1.0 if current_board.turn == chess.WHITE else -1.0
                    elif result == "0-1":
                        value = -1.0 if current_board.turn == chess.WHITE else 1.0
                    else:
                        value = 0.0
            else:
                result = current_board.result()
                if result == "1-0":
                    value = 1.0 if current_board.turn == chess.BLACK else -1.0
                elif result == "0-1":
                    value = -1.0 if current_board.turn == chess.BLACK else 1.0
                else:
                    value = 0.0

            for node_in_path in reversed(search_path):
                node_in_path.visit_count += 1
                node_in_path.total_value += value
                value = -value

        if return_probs:
            return self._get_move_probs(root, board), root.mean_value()
        return self._select_best_move(root, board), root.mean_value()

    def _get_legal_moves_indices(self, board) -> List[int]:
        indices = []
        for move in board.legal_moves:
            idx = move_to_index(move)
            if 0 <= idx < self.config.action_space_size:
                indices.append(idx)
        return indices

    def _add_dirichlet_noise(self, node: Node):
        if node.prior_probs is None:
            return
        legal_indices = node.legal_moves_indices
        if not legal_indices:
            return
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
        epsilon = self.dirichlet_epsilon
        for i, idx in enumerate(legal_indices):
            node.prior_probs[idx] = (1 - epsilon) * node.prior_probs[idx] + epsilon * noise[i]

    def _select_child(self, node: Node) -> int:
        best_score = -float('inf')
        best_idx = -1
        sqrt_parent_n = math.sqrt(max(node.visit_count, 1))

        for idx in node.legal_moves_indices:
            child = node.children.get(idx)
            if child is None:
                prior = node.prior_probs[idx] if node.prior_probs is not None else 0.0
                child = Node(state=None, parent=node, prior_prob=prior)
                node.children[idx] = child

            if child.visit_count > 0:
                q_value = child.total_value / child.visit_count
            else:
                q_value = 0.0

            if child.prior_prob > 0 and node.visit_count > 0:
                u_value = self.c_puct * child.prior_prob * sqrt_parent_n / (1 + child.visit_count)
            else:
                u_value = 0.0

            score = q_value + u_value
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def _select_best_move(self, root: Node, board) -> any:
        if not root.children:
            return random.choice(list(board.legal_moves))
        best_idx = max(root.children.keys(), key=lambda k: root.children[k].visit_count)
        move = index_to_move(best_idx, board)
        if move and move in board.legal_moves:
            return move
        return next(iter(board.legal_moves))

    def _get_move_probs(self, root: Node, board) -> Dict[any, float]:
        probs = {}
        total_n = sum(child.visit_count ** (1.0 / self.temperature) for child in root.children.values())
        if total_n == 0:
            legal_moves = list(board.legal_moves)
            prob = 1.0 / len(legal_moves)
            return {move: prob for move in legal_moves}
        for idx, child in root.children.items():
            move = index_to_move(idx, board)
            if move and move in board.legal_moves:
                probs[move] = (child.visit_count ** (1.0 / self.temperature)) / total_n
        for move in board.legal_moves:
            if move not in probs:
                probs[move] = 0.0
        return probs


class MatchMCTS(MCTS):
    """用于实际对战的 MCTS 变体。"""

    def __init__(self, model, encoder, config):
        super().__init__(model, encoder, config)
        self.temperature = 0.1
        self.dirichlet_epsilon = 0.0
        self.num_simulations = max(100, self.num_simulations)


if __name__ == "__main__":
    import chess
    import time
    from chessmate.config import ChessConfig
    from chessmate.training.neural_net import ChessNet

    print("测试 MCTS 搜索...")
    cfg = ChessConfig()
    cfg.num_res_blocks = 2
    cfg.num_filters = 32
    cfg.mcts_simulations = 20
    cfg.device = "cpu"

    net = ChessNet(cfg)
    encoder = BoardEncoder(history_length=8)
    mcts = MCTS(net, encoder, cfg)

    board = chess.Board()
    print(f"初始局面: {board.fen()}")
    print(f"合法走法数: {board.legal_moves.count()}")

    start = time.time()
    best_move, value = mcts.search(board)
    elapsed = time.time() - start

    print(f"搜索耗时: {elapsed:.2f}s")
    print(f"MCTS 推荐走法: {best_move}")
    print(f"局面评估值: {value:.4f}")
    board.push(best_move)
    print(f"新局面: {board.fen()}")
    print("\nMCTS 测试完成！")