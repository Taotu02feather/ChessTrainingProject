#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 全局配置文件
======================
集中管理所有可调参数，涵盖训练、视觉识别、网页交互、GUI 等模块。
可通过修改此文件或从外部 YAML/JSON 文件加载配置来调整运行参数。

使用方式：
    from chessmate.config import ChessConfig
    cfg = ChessConfig()
    print(cfg.learning_rate)
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ============================================================================
# 路径工具：获取项目根目录
# ============================================================================

def get_project_root() -> str:
    """返回项目根目录（包含 main.py 的目录）。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================================
# 主配置类
# ============================================================================

@dataclass
class ChessConfig:
    """
    ChessMate 全局配置数据类。

    所有参数均有默认值，可直接实例化使用。
    如需自定义，可在创建实例后修改对应属性。
    """

    # ------------------------------------------------------------------
    # 项目路径
    # ------------------------------------------------------------------
    project_root: str = field(default_factory=get_project_root)
    """项目根目录的绝对路径。"""

    # ------------------------------------------------------------------
    # 设备配置 (训练用)
    # ------------------------------------------------------------------
    device: str = "cuda"  # 可选: "cuda", "cpu", "mps"(macOS)
    """PyTorch 使用的计算设备。如果 CUDA 不可用，训练器会自动回退到 CPU。"""

    # ------------------------------------------------------------------
    # 棋盘与规则参数
    # ------------------------------------------------------------------
    board_size: int = 8
    """棋盘大小（标准国际象棋为 8x8）。"""

    action_space_size: int = 4672
    """动作空间大小。
    国际象棋走法空间约为 4672（所有可能的 from_square + to_square + promotion 组合）。
    实际合法的走法由 python-chess 引擎过滤。"""

    num_planes: int = 119
    """输入特征平面的数量（神经网络输入通道数）。
    AlphaZero 原始论文使用 119 个特征平面（包括棋子位置、颜色、历史状态等）。"""

    # ------------------------------------------------------------------
    # 神经网络参数
    # ------------------------------------------------------------------
    num_res_blocks: int = 12
    """残差块的数量。AlphaZero 原始使用 19 或 39，小规模实验建议 4~8。"""

    num_filters: int = 128
    """卷积层的滤波器数量。AlphaZero 原始使用 256，小规模实验建议 64~128。"""

    value_head_hidden: int = 256
    """价值头（Value Head）隐藏层大小。"""

    # ------------------------------------------------------------------
    # MCTS（蒙特卡洛树搜索）参数
    # ------------------------------------------------------------------
    mcts_simulations: int = 400
    """每步棋的 MCTS 搜索模拟次数。越多越强但越慢。训练时建议 200~800。"""

    mcts_c_puct: float = 1.0
    """PUCT 公式中的探索常数 c_puct。控制探索与利用的平衡。"""

    mcts_temperature: float = 1.0
    """MCTS 根节点的温度参数。
    温度 > 1 鼓励探索（更多随机性），温度 < 1 鼓励利用（更多确定性）。
    训练初期可以设为 1.0，对弈时可以设为 0.1。"""

    mcts_dirichlet_alpha: float = 0.3
    """Dirichlet 噪声的 alpha 参数，用于根节点的先验概率。"""

    mcts_dirichlet_epsilon: float = 0.25
    """Dirichlet 噪声的混合比例 epsilon。"""

    # ------------------------------------------------------------------
    # 自对弈参数
    # ------------------------------------------------------------------
    num_self_play_games: int = 40
    """每轮训练的自我对弈局数。小规模实验建议 10~100。"""

    max_moves_per_game: int = 200
    """每局棋的最大步数（防止无限循环）。"""

    resign_threshold: float = -0.95
    """自动认输的估值阈值。当模型对当前局面的估值低于此值时自动认输。"""

    # ------------------------------------------------------------------
    # 经验回放池参数
    # ------------------------------------------------------------------
    replay_buffer_capacity: int = 100000
    """经验回放池的最大容量（存储的状态-策略-价值样本数）。"""

    replay_batch_size: int = 256
    """每次训练从回放池中采样的批量大小。"""

    # ------------------------------------------------------------------
    # 训练参数
    # ------------------------------------------------------------------
    learning_rate: float = 0.001
    """初始学习率。"""

    weight_decay: float = 1e-4
    """权重衰减（L2 正则化）系数。"""

    momentum: float = 0.9
    """SGD 动量（如果使用 SGD 优化器）。"""

    num_train_epochs: int = 5
    """每轮数据上的训练轮数（epochs）。"""

    max_training_iterations: int = 50
    """最大训练迭代轮数（每轮包含若干自对弈局数 + 一次训练）。"""

    checkpoint_frequency: int = 5
    """每隔多少轮保存一次模型检查点。"""

    # ------------------------------------------------------------------
    # 视觉识别参数
    # ------------------------------------------------------------------
    vision_screenshot_region: Optional[Tuple[int, int, int, int]] = None
    """屏幕截图区域 (left, top, width, height)。
    若为 None 则截取全屏。建议在使用网页对战时手动设置为棋盘所在区域。
    例如: (100, 200, 600, 600) 表示从 (100, 200) 开始截取 600x600 的区域。"""

    vision_board_light_color: Tuple[int, int, int] = (240, 217, 181)
    """棋盘浅色（白格）的参考 RGB 颜色，用于颜色校准。"""

    vision_board_dark_color: Tuple[int, int, int] = (181, 136, 99)
    """棋盘深色（黑格）的参考 RGB 颜色，用于颜色校准。"""

    vision_square_size_min: int = 40
    """棋盘格子的最小像素大小（用于检测棋盘尺寸）。"""

    vision_square_size_max: int = 120
    """棋盘格子的最大像素大小。"""

    vision_use_template_matching: bool = True
    """是否使用模板匹配来识别棋子。True 使用模板匹配，False 使用颜色分类。"""

    vision_templates_dir: str = field(default_factory=lambda: os.path.join(get_project_root(), "data", "templates"))
    """棋子模板图片的存放目录。"""

    vision_confidence_threshold: float = 0.6
    """棋子模板匹配的置信度阈值。低于此值则认为无法识别。"""

    # ------------------------------------------------------------------
    # 网页交互参数
    # ------------------------------------------------------------------
    web_click_delay: float = 0.3
    """两次鼠标点击之间的延迟（秒），避免网页检测到过快操作。"""

    web_move_duration: float = 0.5
    """鼠标拖拽走子时的移动持续时间（秒）。"""

    web_board_top_left: Tuple[int, int] = (100, 200)
    """网页棋盘左上角的屏幕坐标 (x, y)。需根据实际网页位置调整。"""

    web_square_size: int = 75
    """网页上每个棋盘格子的像素大小。需根据实际网页调整。"""

    # ------------------------------------------------------------------
    # GUI 参数
    # ------------------------------------------------------------------
    gui_window_width: int = 900
    """GUI 窗口宽度。"""

    gui_window_height: int = 700
    """GUI 窗口高度。"""

    gui_square_size: int = 70
    """GUI 棋盘每个格子的像素大小。"""

    gui_player_color: bool = True  # True = 玩家执白
    """玩家执子颜色。True 表示玩家执白，False 表示玩家执黑（AI 先行）。"""

    gui_ai_thinking_time: float = 1.0
    """AI 思考时的最短显示时间（秒），让玩家看到 AI 在思考而非瞬间落子。"""

    gui_show_coordinates: bool = True
    """是否在棋盘边缘显示坐标标签（a-h, 1-8）。"""

    # ------------------------------------------------------------------
    # 日志与调试
    # ------------------------------------------------------------------
    log_level: int = logging.INFO
    """日志级别。可选: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR。"""

    log_dir: str = field(default_factory=lambda: os.path.join(get_project_root(), "logs"))
    """日志文件存放目录。"""

    log_to_file: bool = True
    """是否将日志输出到文件（同时保留控制台输出）。"""

    log_filename: str = "chessmate.log"
    """日志文件名。"""

    # ------------------------------------------------------------------
    # 模型保存与加载
    # ------------------------------------------------------------------
    model_dir: str = field(default_factory=lambda: os.path.join(get_project_root(), "models"))
    """模型权重文件的存放目录。"""

    best_model_name: str = "best_model.pth"
    """最优模型的保存文件名。"""

    latest_model_name: str = "latest_model.pth"
    """最新模型的保存文件名。"""

    # ------------------------------------------------------------------
    # 棋盘状态编码参数
    # ------------------------------------------------------------------
    history_length: int = 8
    """棋盘历史状态编码时保留的最近步数。
    AlphaZero 使用 8 步历史，即当前局面 + 过去 7 步的局面。"""

    # ------------------------------------------------------------------
    # 实用方法
    # ------------------------------------------------------------------

    def ensure_dirs(self):
        """确保所有需要的目录都存在（自动创建）。"""
        dirs = [self.log_dir, self.model_dir, self.vision_templates_dir]
        for d in dirs:
            os.makedirs(d, exist_ok=True)

    def setup_logging(self, name: str = "ChessMate") -> logging.Logger:
        """
        根据配置初始化日志系统。

        Args:
            name: 日志记录器的名称。

        Returns:
            配置好的 logging.Logger 实例。
        """
        logger = logging.getLogger(name)
        logger.setLevel(self.log_level)

        # 避免重复添加 handler
        if logger.handlers:
            return logger

        # 格式化器
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 文件输出
        if self.log_to_file:
            self.ensure_dirs()
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, self.log_filename),
                encoding="utf-8"
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger

    def update_from_dict(self, config_dict: dict):
        """
        从字典更新配置参数（用于从外部文件加载配置）。

        Args:
            config_dict: 包含配置键值对的字典。
        """
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_dict(self) -> dict:
        """
        将当前配置导出为字典（用于保存配置到文件）。

        Returns:
            包含所有配置参数的字典。
        """
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith("_"):
                # 跳过不可序列化的字段
                if key in ("project_root",):
                    continue
                if isinstance(value, type(logging.INFO)):
                    result[key] = logging.getLevelName(value)
                elif callable(value):
                    continue
                else:
                    result[key] = value
        return result


# ============================================================================
# 预定义配置方案（快速切换）
# ============================================================================

def get_small_config() -> ChessConfig:
    """返回适合快速测试的小规模配置（CPU 上也可快速运行）。"""
    cfg = ChessConfig()
    cfg.num_res_blocks = 4
    cfg.num_filters = 64
    cfg.mcts_simulations = 50
    cfg.num_self_play_games = 5
    cfg.replay_buffer_capacity = 10000
    cfg.replay_batch_size = 64
    cfg.num_train_epochs = 2
    cfg.max_training_iterations = 10
    cfg.device = "cpu"
    return cfg


def get_medium_config() -> ChessConfig:
    """返回中等规模配置（适合有 GPU 的环境）。"""
    cfg = ChessConfig()
    cfg.num_res_blocks = 8
    cfg.num_filters = 128
    cfg.mcts_simulations = 200
    cfg.num_self_play_games = 20
    cfg.replay_buffer_capacity = 50000
    cfg.replay_batch_size = 256
    cfg.num_train_epochs = 3
    cfg.max_training_iterations = 30
    return cfg


def get_full_config() -> ChessConfig:
    """返回完整规模配置（AlphaZero 级别，需要强大 GPU）。"""
    cfg = ChessConfig()
    cfg.num_res_blocks = 19
    cfg.num_filters = 256
    cfg.mcts_simulations = 800
    cfg.num_self_play_games = 100
    cfg.replay_buffer_capacity = 500000
    cfg.replay_batch_size = 1024
    cfg.num_train_epochs = 5
    cfg.max_training_iterations = 100
    return cfg


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    # 测试配置加载
    cfg = ChessConfig()
    print("=== ChessMate 默认配置 ===")
    for key, value in cfg.to_dict().items():
        print(f"  {key}: {value}")

    print("\n=== 小规模测试配置 ===")
    small = get_small_config()
    for key, value in small.to_dict().items():
        if value != getattr(ChessConfig(), key, None):
            print(f"  {key}: {value}")