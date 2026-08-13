#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""强化学习微调启动脚本（从预训练模型继续训练）"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chessmate.config import ChessConfig
from chessmate.training.trainer import Trainer

if __name__ == "__main__":
    cfg = ChessConfig()
    trainer = Trainer(cfg)
    trainer.train()
