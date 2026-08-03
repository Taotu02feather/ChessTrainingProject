#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 训练器
================
实现完整的 AlphaZero 风格强化学习训练循环。

训练流程（每轮迭代）：
1. 自对弈阶段：使用当前模型进行多局自对弈，收集经验数据
2. 训练阶段：从回放缓冲区采样，更新神经网络权重
3. 评估阶段（可选）：新模型与旧模型对战，评估棋力提升
4. 保存检查点：定期保存模型和数据

核心训练循环采用 AlphaZero 的方法：
    - 使用 MCTS 搜索引导自对弈
    - 策略目标来自 MCTS 访问计数（而非原始网络输出）
    - 价值目标来自实际对局结果
    - 损失 = 交叉熵(策略) + MSE(价值) + L2 正则化
"""

import sys
import os
import time
import json
from typing import Optional, Dict, Any

# 允许直接运行此文件时也能找到 chessmate 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from chessmate.training.neural_net import (
    ChessNet, BoardEncoder, ChessLoss, move_to_index
)
from chessmate.training.mcts import MCTS
from chessmate.training.replay_buffer import ReplayBuffer
from chessmate.training.self_play import SelfPlay


# ============================================================================
# 训练器
# ============================================================================

class Trainer:
    """
    ChessMate 训练器。

    管理完整的训练流程：自对弈 -> 经验收集 -> 神经网络更新 -> 评估。

    使用方式：
        trainer = Trainer(config)
        trainer.train()  # 开始完整训练循环

    或者逐步控制：
        trainer.self_play_phase()
        trainer.train_phase()
        trainer.save_checkpoint()
    """

    def __init__(self, config, logger=None):
        """
        初始化训练器。

        Args:
            config: ChessConfig 配置对象。
            logger: 日志记录器（可选，不提供则自动创建）。
        """
        self.config = config
        self.logger = logger or config.setup_logging("Trainer")

        # 设备配置
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.logger.info(f"训练设备: {self.device}")

        # ---- 初始化核心组件 ----
        self.encoder = BoardEncoder(history_length=config.history_length)

        # 神经网络
        self.model = ChessNet(config).to(self.device)
        self.logger.info(
            f"模型参数量: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=config.replay_buffer_capacity,
            config=config,
        )

        # 自对弈引擎
        self.self_play = SelfPlay(
            self.model, self.encoder, config, self.logger
        )

        # 优化器（使用 AdamW，支持权重衰减）
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 学习率调度器（余弦退火）
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_training_iterations,
            eta_min=config.learning_rate * 0.01,
        )

        # 损失函数
        self.loss_fn = ChessLoss()

        # ---- 训练状态 ----
        self.current_iteration = 0
        self.total_steps = 0
        self.best_loss = float('inf')

        # TensorBoard 日志（可选）
        self.writer = None
        try:
            log_dir = os.path.join(config.log_dir, "tensorboard")
            self.writer = SummaryWriter(log_dir=log_dir)
            self.logger.info(f"TensorBoard 日志目录: {log_dir}")
        except Exception as e:
            self.logger.warning(f"无法创建 TensorBoard writer: {e}")

        # 确保目录存在
        config.ensure_dirs()

    def train(self):
        """
        执行完整的训练循环。

        每轮迭代包含：
        1. 自对弈生成数据
        2. 神经网络训练
        3. 检查点保存
        """
        self.logger.info("=" * 60)
        self.logger.info("ChessMate 训练开始")
        self.logger.info(f"最大迭代轮数: {self.config.max_training_iterations}")
        self.logger.info(f"每轮自对弈局数: {self.config.num_self_play_games}")
        self.logger.info(f"MCTS 模拟次数: {self.config.mcts_simulations}")
        self.logger.info(f"回放缓冲区容量: {self.config.replay_buffer_capacity}")
        self.logger.info("=" * 60)

        start_time = time.time()

        for iteration in range(self.config.max_training_iterations):
            self.current_iteration = iteration + 1
            iter_start = time.time()

            self.logger.info(f"\n--- 第 {self.current_iteration}/{self.config.max_training_iterations} 轮训练 ---")

            # ---- Phase 1: 自对弈 ----
            self.logger.info("阶段1: 自对弈生成数据...")
            num_experiences = self.self_play_phase()

            # 检查缓冲区是否准备好
            if not self.replay_buffer.is_ready:
                self.logger.warning(
                    f"缓冲区样本不足 ({len(self.replay_buffer)} < "
                    f"{self.config.replay_batch_size})，跳过训练阶段"
                )
                continue

            # ---- Phase 2: 训练神经网络 ----
            self.logger.info("阶段2: 训练神经网络...")
            train_metrics = self.train_phase()

            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # ---- 记录日志 ----
            iter_time = time.time() - iter_start
            self._log_iteration(iteration, train_metrics, num_experiences, iter_time, current_lr)

            # ---- 保存检查点 ----
            if self.current_iteration % self.config.checkpoint_frequency == 0:
                self.save_checkpoint()

            # 如果是最佳模型则保存
            avg_loss = train_metrics.get('total_loss', float('inf'))
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.save_best_model()

        # 训练结束
        total_time = time.time() - start_time
        self.logger.info("=" * 60)
        self.logger.info(f"训练完成！总耗时: {total_time/3600:.2f} 小时")
        self.logger.info(f"最佳损失: {self.best_loss:.6f}")

        # 保存最终模型
        self.save_checkpoint(filename="final_model.pth")
        self.save_best_model()

        # 关闭日志写入器
        if self.writer:
            self.writer.close()

    def self_play_phase(self) -> int:
        """
        执行自对弈阶段。

        Returns:
            本轮收集的经验样本数。
        """
        self.model.eval()  # 评估模式（自对弈时不需要梯度）
        num_exp = self.self_play.play_and_collect(
            self.replay_buffer,
            show_progress=True,
        )

        # 打印统计
        stats = self.self_play.get_stats()
        self.logger.info(
            f"自对弈统计: {stats.get('total', 0)}局 -> "
            f"白胜率={stats.get('white_win_rate', 0):.1%}, "
            f"黑胜率={stats.get('black_win_rate', 0):.1%}, "
            f"和棋率={stats.get('draw_rate', 0):.1%}"
        )

        return num_exp

    def train_phase(self) -> Dict[str, float]:
        """
        执行神经网络训练阶段。

        Returns:
            包含训练指标（loss等）的字典。
        """
        self.model.train()
        num_epochs = self.config.num_train_epochs
        batch_size = self.config.replay_batch_size
        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        # 计算每个 epoch 的批次数
        batches_per_epoch = max(1, len(self.replay_buffer) // batch_size)

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # 使用进度条显示训练进度
            pbar = tqdm(
                range(batches_per_epoch),
                desc=f"训练 Epoch {epoch+1}/{num_epochs}",
                unit="batch",
            )

            for _ in pbar:
                # 采样一个批次
                states, policy_targets, value_targets = self.replay_buffer.sample(
                    batch_size
                )
                states = states.to(self.device)
                policy_targets = policy_targets.to(self.device)
                value_targets = value_targets.to(self.device)

                # 前向传播
                self.optimizer.zero_grad()
                policy_pred, value_pred = self.model(states)

                # 计算损失
                total_loss, policy_loss, value_loss = self.loss_fn(
                    policy_pred, value_pred, policy_targets, value_targets
                )

                # 反向传播
                total_loss.backward()

                # 梯度裁剪（防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                # 更新参数
                self.optimizer.step()

                # 累计损失
                batch_loss = total_loss.item()
                epoch_loss += batch_loss
                total_loss_sum += batch_loss
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                num_batches += 1

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'p_loss': f'{policy_loss.item():.4f}',
                    'v_loss': f'{value_loss.item():.4f}',
                })

                self.total_steps += 1

                # TensorBoard 记录（每 10 步）
                if self.writer and self.total_steps % 10 == 0:
                    self.writer.add_scalar('Train/TotalLoss', batch_loss, self.total_steps)
                    self.writer.add_scalar('Train/PolicyLoss', policy_loss.item(), self.total_steps)
                    self.writer.add_scalar('Train/ValueLoss', value_loss.item(), self.total_steps)

            avg_epoch_loss = epoch_loss / batches_per_epoch
            self.logger.debug(
                f"Epoch {epoch+1}/{num_epochs} 平均损失: {avg_epoch_loss:.6f}"
            )

        # 计算平均指标
        metrics = {
            'total_loss': total_loss_sum / num_batches,
            'policy_loss': policy_loss_sum / num_batches,
            'value_loss': value_loss_sum / num_batches,
            'batches': num_batches,
        }

        return metrics

    def _log_iteration(
        self,
        iteration: int,
        metrics: dict,
        num_exp: int,
        iter_time: float,
        lr: float,
    ):
        """
        记录一轮训练的日志。

        Args:
            iteration: 当前迭代轮次。
            metrics: 训练指标字典。
            num_exp: 收集的经验数。
            iter_time: 本轮耗时（秒）。
            lr: 当前学习率。
        """
        self.logger.info(
            f"第 {iteration+1} 轮完成 | "
            f"损失: {metrics['total_loss']:.4f} | "
            f"策略损失: {metrics['policy_loss']:.4f} | "
            f"价值损失: {metrics['value_loss']:.4f} | "
            f"经验数: {num_exp} | "
            f"耗时: {iter_time:.1f}s | "
            f"学习率: {lr:.2e}"
        )

        # TensorBoard 记录
        if self.writer:
            self.writer.add_scalar('Iteration/TotalLoss', metrics['total_loss'], iteration)
            self.writer.add_scalar('Iteration/PolicyLoss', metrics['policy_loss'], iteration)
            self.writer.add_scalar('Iteration/ValueLoss', metrics['value_loss'], iteration)
            self.writer.add_scalar('Iteration/Experiences', num_exp, iteration)
            self.writer.add_scalar('Iteration/LearningRate', lr, iteration)
            self.writer.add_scalar('Iteration/Time', iter_time, iteration)

    def save_checkpoint(self, filename: str = None):
        """
        保存训练检查点（包含模型、优化器状态、训练进度）。

        Args:
            filename: 文件名。默认使用 latest_model_name。
        """
        if filename is None:
            filename = self.config.latest_model_name

        filepath = os.path.join(self.config.model_dir, filename)

        checkpoint = {
            'iteration': self.current_iteration,
            'total_steps': self.total_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config.to_dict(),
        }

        torch.save(checkpoint, filepath)
        self.logger.info(f"检查点已保存: {filepath}")

    def load_checkpoint(self, filepath: str):
        """
        加载训练检查点。

        Args:
            filepath: 检查点文件路径。
        """
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_iteration = checkpoint.get('iteration', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))

        self.logger.info(
            f"检查点已加载: {filepath} (第 {self.current_iteration} 轮)"
        )

    def save_best_model(self):
        """保存当前最优模型。"""
        filepath = os.path.join(self.config.model_dir, self.config.best_model_name)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': {
                'num_planes': self.config.num_planes,
                'num_filters': self.config.num_filters,
                'num_res_blocks': self.config.num_res_blocks,
                'action_space_size': self.config.action_space_size,
                'value_head_hidden': self.config.value_head_hidden,
            }
        }, filepath)
        self.logger.info(f"最优模型已保存: {filepath} (loss={self.best_loss:.6f})")

    def evaluate(self, num_games: int = 10) -> Dict[str, float]:
        """
        评估当前模型：让当前模型与随机走法对弈。

        Args:
            num_games: 评估对局数。

        Returns:
            包含胜率等指标的字典。
        """
        import chess
        import random

        wins = 0
        losses = 0
        draws = 0

        self.model.eval()
        encoder = self.encoder
        mcts = MCTS(self.model, encoder, self.config)

        for _ in tqdm(range(num_games), desc="评估模型", unit="局"):
            board = chess.Board()

            while not board.is_game_over():
                if board.turn == chess.WHITE:
                    # 当前模型执白
                    move, _ = mcts.search(board)
                    if move:
                        board.push(move)
                    else:
                        break
                else:
                    # 对手走随机
                    if board.legal_moves.count() > 0:
                        move = random.choice(list(board.legal_moves))
                        board.push(move)
                    else:
                        break

            result = board.result()
            if result == "1-0":
                wins += 1
            elif result == "0-1":
                losses += 1
            else:
                draws += 1

        metrics = {
            'wins': wins,
            'losses': losses,
            'draws': draws,
            'win_rate': wins / num_games,
            'games': num_games,
        }

        self.logger.info(
            f"评估结果 ({num_games}局): "
            f"胜={wins}, 负={losses}, 和={draws}, "
            f"胜率={metrics['win_rate']:.1%}"
        )

        return metrics


# ============================================================================
# 便捷训练函数
# ============================================================================

def quick_train(config=None, iterations: int = 5, games_per_iter: int = 5):
    """
    快速训练函数，用于测试训练流程。

    Args:
        config: ChessConfig 对象。如果为 None，使用小规模配置。
        iterations: 训练迭代轮数。
        games_per_iter: 每轮自对弈局数。

    Returns:
        训练好的 Trainer 对象。
    """
    if config is None:
        from chessmate.config import get_small_config
        config = get_small_config()

    config.max_training_iterations = iterations
    config.num_self_play_games = games_per_iter
    config.mcts_simulations = 20  # 极少的模拟，加速测试
    config.num_res_blocks = 2
    config.num_filters = 32
    config.replay_batch_size = 32
    config.num_train_epochs = 1  # 每个 epoch 的批次数较少

    trainer = Trainer(config)
    trainer.train()

    return trainer


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import logging
    from chessmate.config import get_small_config

    print("测试训练器模块...")

    # 使用最小规模配置以快速完成测试
    cfg = get_small_config()
    cfg.max_training_iterations = 2
    cfg.num_self_play_games = 2
    cfg.mcts_simulations = 10
    cfg.num_res_blocks = 2
    cfg.num_filters = 16
    cfg.replay_buffer_capacity = 2000
    cfg.replay_batch_size = 16
    cfg.num_train_epochs = 1
    cfg.log_level = logging.INFO

    trainer = quick_train(
        config=cfg,
        iterations=2,
        games_per_iter=2,
    )

    print(f"\n训练完成！共 {trainer.total_steps} 步，最佳损失: {trainer.best_loss:.4f}")
    print("训练器模块测试通过！")