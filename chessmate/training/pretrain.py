#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 监督学习预训练模块
============================
从人类棋谱（PGN 文件）训练神经网络，让模型先学会"模仿人类走子"，
再通过 AlphaZero 强化学习微调。

背景：从零开始的 AlphaZero 需要数百万局自对弈，个人算力无法复现。
监督学习预训练是 LeetCode 等项目标准做法——先学人类走法，再强化学习。

使用方式：
    python main.py pretrain --pgn data/games.pgn --epochs 3
"""

import sys
import os
import glob
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import chess
import chess.pgn
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from chessmate.training.neural_net import ChessNet, BoardEncoder, move_to_index


class Pretrainer:
    """监督学习预训练器：从 PGN 棋谱训练策略头模仿人类走法。"""

    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or config.setup_logging("Pretrainer")

        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"预训练设备: {self.device}")

        self.encoder = BoardEncoder(history_length=config.history_length)
        self.model = ChessNet(config).to(self.device)
        self.logger.info(
            f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.total_games = 0

    def _extract_training_data(self, pgn_file, max_games: int = None):
        """从 PGN 文件流式提取 (state, policy_idx, value) 三元组。"""
        game_count = 0

        with open(pgn_file, "r", encoding="utf-8", errors="ignore") as f:
            while True:
                if max_games is not None and game_count >= max_games:
                    break

                game = chess.pgn.read_game(f)
                if game is None:
                    break

                game_count += 1
                self.total_games += 1

                result_header = game.headers.get("Result", "*")
                if result_header == "1-0":
                    game_value = 1.0
                elif result_header == "0-1":
                    game_value = -1.0
                else:
                    game_value = 0.0

                board = game.board()
                self.encoder.clear_history()
                self.encoder.push_history(board)

                try:
                    for node in game.mainline():
                        move = node.move
                        if move is None:
                            break

                        # 验证走法合法性，非法则跳过该局
                        if move not in board.legal_moves:
                            break

                        encoded = self.encoder.encode(board)
                        policy_idx = move_to_index(move)

                        if board.turn == chess.WHITE:
                            value = game_value
                        else:
                            value = -game_value

                        yield encoded, policy_idx, value

                        board.push(move)
                        self.encoder.push_history(board)
                except (ValueError, AssertionError):
                    # 遇到无法解析的走法，跳过该局
                    pass

        self.logger.info(f"已处理 {game_count} 局棋谱")

    def train(self, pgn_files, epochs: int = 3, max_games: int = None):
        """执行监督学习预训练。"""
        files = self._resolve_pgn_files(pgn_files)
        if not files:
            self.logger.error("未找到 PGN 文件，请检查路径")
            return

        self.logger.info("=" * 60)
        self.logger.info("监督学习预训练开始")
        self.logger.info(f"PGN 文件数: {len(files)}, 训练轮数: {epochs}")
        self.logger.info("=" * 60)

        self.model.train()

        for epoch in range(epochs):
            epoch_p_loss = 0.0
            epoch_v_loss = 0.0
            num_batches = 0

            pbar = tqdm(files, desc=f"Epoch {epoch+1}/{epochs}", unit="文件")

            for pgn_file in pbar:
                batch_states, batch_policies, batch_values = [], [], []
                batch_size = self.config.replay_batch_size

                for encoded, policy_idx, value in self._extract_training_data(
                    pgn_file, max_games
                ):
                    batch_states.append(encoded)
                    batch_policies.append(policy_idx)
                    batch_values.append(value)

                    if len(batch_states) >= batch_size:
                        lp, lv = self._train_batch(
                            batch_states, batch_policies, batch_values
                        )
                        epoch_p_loss += lp
                        epoch_v_loss += lv
                        num_batches += 1
                        batch_states, batch_policies, batch_values = [], [], []
                        pbar.set_postfix({'p': f'{lp:.4f}', 'v': f'{lv:.4f}'})

                if batch_states:
                    lp, lv = self._train_batch(
                        batch_states, batch_policies, batch_values
                    )
                    epoch_p_loss += lp
                    epoch_v_loss += lv
                    num_batches += 1

            avg_p = epoch_p_loss / max(1, num_batches)
            avg_v = epoch_v_loss / max(1, num_batches)
            self.logger.info(
                f"Epoch {epoch+1}/{epochs} | 策略损失: {avg_p:.4f} | "
                f"价值损失: {avg_v:.4f}"
            )

        self.model.eval()
        save_path = os.path.join(self.config.model_dir, self.config.best_model_name)
        self.model.save(save_path)
        self.logger.info(f"预训练模型已保存: {save_path}")

    def _train_batch(self, states, policies, values):
        """训练一个批次，返回 (policy_loss, value_loss)。"""
        states_tensor = torch.stack(states).to(self.device)
        policies_tensor = torch.tensor(policies, dtype=torch.long).to(self.device)
        values_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(1).to(self.device)

        self.optimizer.zero_grad()
        policy_logits, value_pred = self.model(states_tensor)

        policy_loss = self.policy_loss_fn(policy_logits, policies_tensor)
        value_loss = self.value_loss_fn(value_pred, values_tensor)

        total_loss = policy_loss + value_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return policy_loss.item(), value_loss.item()

    def _resolve_pgn_files(self, pgn_files) -> List[str]:
        """解析 PGN 路径，支持文件/目录/glob 模式。"""
        if isinstance(pgn_files, str):
            pgn_files = [pgn_files]

        files = []
        for path in pgn_files:
            if os.path.isdir(path):
                files.extend(glob.glob(os.path.join(path, "*.pgn")))
            elif os.path.isfile(path):
                files.append(path)
            elif "*" in path or "?" in path:
                files.extend(glob.glob(path))
            else:
                self.logger.warning(f"路径不存在: {path}")

        return sorted(files)


