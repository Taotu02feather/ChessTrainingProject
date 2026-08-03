#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ChessMate 主入口脚本
====================
提供命令行和交互式菜单选择三种运行模式：

  1. 训练模式：启动强化学习训练流程
  2. 本地对弈模式：启动 PyQt5 GUI 窗口进行人机对弈
  3. 网页对战模式：通过屏幕截图和鼠标控制进行网页自动对弈

使用方式：
    python main.py              # 交互式菜单选择模式
    python main.py train        # 直接启动训练模式
    python main.py gui          # 直接启动本地对弈
    python main.py web          # 直接启动网页对战
    python main.py check        # 仅运行环境检测
    python main.py calibrate    # 启动网页棋盘位置校准工具
    python main.py test         # 运行快速测试（验证所有模块）
"""

import sys
import os
import argparse
import logging

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def show_banner():
    """显示 ChessMate ASCII 标题。"""
    print("""
     ██████╗██╗  ██╗███████╗███████╗███████╗███╗   ███╗ █████╗ ████████╗███████╗
    ██╔════╝██║  ██║██╔════╝██╔════╝██╔════╝████╗ ████║██╔══██╗╚══██╔══╝██╔════╝
    ██║     ███████║█████╗  ███████╗███████╗██╔████╔██║███████║   ██║   █████╗
    ██║     ██╔══██║██╔══╝  ╚════██║╚════██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝
    ╚██████╗██║  ██║███████╗███████║███████║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗
     ╚═════╝╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝

         国际象棋 AI 系统 v0.1.0
         训练 | 对弈 | 征服
    """)


def show_menu() -> str:
    """
    显示交互式菜单并获取用户选择。

    Returns:
        用户选择的模式字符串：'train', 'gui', 'web', 'check', 'calibrate', 'test', 'exit'
    """
    show_banner()
    print("请选择运行模式：")
    print("  [1] 训练模式      - 启动 AlphaZero 风格强化学习训练")
    print("  [2] 本地对弈 (GUI) - 启动图形界面进行人机对弈")
    print("  [3] 网页对战       - 屏幕截图识别 + 自动走子")
    print("  [4] 环境检测       - 检查依赖安装状态")
    print("  [5] 位置校准       - 校准网页棋盘位置")
    print("  [6] 快速测试       - 运行全模块诊断测试")
    print("  [0] 退出")
    print()

    mode_map = {
        '1': 'train', '2': 'gui', '3': 'web',
        '4': 'check', '5': 'calibrate', '6': 'test', '0': 'exit'
    }

    while True:
        try:
            choice = input("请输入选项 [0-6]: ").strip()
            if choice in mode_map:
                return mode_map[choice]
            print("无效选项，请重新输入。")
        except (KeyboardInterrupt, EOFError):
            return 'exit'


def run_check():
    """运行环境检测脚本。"""
    print("\n正在运行环境检测...")
    from check_env import run_check as check
    check()


def run_train():
    """启动训练模式。"""
    from chessmate.config import ChessConfig, get_small_config
    from chessmate.training.trainer import Trainer

    print("\n" + "=" * 60)
    print("ChessMate 训练模式")
    print("=" * 60)

    # 询问训练规模
    print("\n选择训练规模：")
    print("  [1] 小规模测试 (快速验证，CPU 友好)")
    print("  [2] 中等规模 (需要 GPU)")
    print("  [3] 自定义 (使用默认 config.py 中的设置)")

    choice = input("请选择 [1-3] (默认: 1): ").strip() or "1"

    if choice == "1":
        config = get_small_config()
        print("使用小规模配置")
    elif choice == "2":
        from chessmate.config import get_medium_config
        config = get_medium_config()
        print("使用中等规模配置")
    else:
        config = ChessConfig()
        print("使用 config.py 中的默认配置")

    # 显示关键参数
    print(f"\n训练参数:")
    print(f"  迭代轮数: {config.max_training_iterations}")
    print(f"  每轮对局: {config.num_self_play_games}")
    print(f"  MCTS 模拟: {config.mcts_simulations}")
    print(f"  残差块数: {config.num_res_blocks}")
    print(f"  滤波器数: {config.num_filters}")
    print(f"  设备: {config.device}")

    confirm = input("\n开始训练? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("已取消训练。")
        return

    # 创建训练器并开始训练
    trainer = Trainer(config)
    trainer.train()

    print("\n训练完成！")


def run_gui():
    """启动本地对弈 GUI。"""
    print("\n正在启动 ChessMate GUI...")
    try:
        from chessmate.config import ChessConfig
        from chessmate.gui.chess_window import launch_gui

        config = ChessConfig()

        # 询问玩家执子颜色
        choice = input("玩家执白? [Y/n]: ").strip().lower()
        config.gui_player_color = (choice != 'n')

        print(f"玩家执{'白' if config.gui_player_color else '黑'}方")
        print("\n启动 GUI 窗口...")
        print("提示：关闭窗口即可退出。")

        launch_gui(config=config)

    except ImportError as e:
        print(f"错误：无法导入 PyQt5。请确保已安装：pip install PyQt5")
        print(f"详细错误: {e}")
    except Exception as e:
        print(f"启动 GUI 失败: {e}")


def run_web():
    """启动网页对战模式。"""
    print("\n" + "=" * 60)
    print("ChessMate 网页对战模式")
    print("=" * 60)

    try:
        import pyautogui
    except ImportError:
        print("错误：pyautogui 未安装。请运行：pip install pyautogui")
        return

    from chessmate.config import ChessConfig
    from chessmate.training.neural_net import ChessNet
    from chessmate.web.web_player import WebPlayer

    config = ChessConfig()

    # 检查棋盘位置是否已配置
    print(f"\n当前网页棋盘配置：")
    print(f"  左上角坐标: {config.web_board_top_left}")
    print(f"  格子大小: {config.web_square_size} px")

    need_calibrate = input("\n需要重新校准棋盘位置? [y/N]: ").strip().lower()
    if need_calibrate == 'y':
        # 创建临时 WebPlayer 进行校准
        model = ChessNet(config)
        model.eval()
        player = WebPlayer(model, config)
        player.calibrate_position()

        # 更新配置
        config.web_board_top_left = player.board_top_left
        config.web_square_size = player.square_size

    # 选择对弈角色
    print("\n选择对弈角色：")
    print("  [1] 执白 (AI 后手) - 等待对手走子后 AI 应对")
    print("  [2] 执黑 (AI 先手) - AI 先行，然后等待对手")
    print("  [3] 仅走一步 - 测试单步走子")

    role_choice = input("请选择 [1-3]: ").strip() or "1"

    # 创建模型（尝试加载已训练的模型）
    model = ChessNet(config)
    model_path = os.path.join(config.model_dir, config.best_model_name)
    if os.path.exists(model_path):
        try:
            model = ChessNet.load(model_path, config=config, device=config.device)
            print(f"已加载训练模型: {model_path}")
        except Exception as e:
            print(f"无法加载模型 ({e})，使用未训练模型")
    else:
        print("未找到已训练模型，使用未训练模型（AI 走法随机）")

    model.eval()
    player = WebPlayer(model, config)

    if role_choice == "1":
        num_moves = int(input("最大步数 (默认: 40): ").strip() or "40")
        player.play_as_white(num_moves=num_moves)
    elif role_choice == "2":
        num_moves = int(input("最大步数 (默认: 40): ").strip() or "40")
        player.play_as_black(num_moves=num_moves)
    else:
        print("AI 正在思考单步走法...")
        player.make_one_move()


def run_calibrate():
    """启动网页棋盘位置校准。"""
    print("\n正在启动棋盘位置校准工具...")
    try:
        import pyautogui
    except ImportError:
        print("错误：pyautogui 未安装。请运行：pip install pyautogui")
        return

    from chessmate.config import ChessConfig
    from chessmate.training.neural_net import ChessNet
    from chessmate.web.web_player import WebPlayer

    config = ChessConfig()
    model = ChessNet(config)
    model.eval()

    player = WebPlayer(model, config)
    player.calibrate_position()

    print("\n校准完成！请将以上坐标填入 config.py 或在启动时设置。")


def run_test():
    """运行快速诊断测试，验证所有模块的基本功能。"""
    print("\n" + "=" * 60)
    print("ChessMate 系统诊断测试")
    print("=" * 60)

    import traceback

    # 测试 1：配置
    print("\n[1/6] 测试配置模块...")
    try:
        from chessmate.config import ChessConfig, get_small_config
        cfg = get_small_config()
        assert cfg.num_res_blocks > 0
        print("  ✅ 配置模块正常")
    except Exception as e:
        print(f"  ❌ 配置模块异常: {e}")

    # 测试 2：神经网络
    print("\n[2/6] 测试神经网络...")
    try:
        import torch
        from chessmate.training.neural_net import ChessNet, BoardEncoder, move_to_index
        cfg.num_filters = 16
        cfg.num_res_blocks = 2
        net = ChessNet(cfg)
        encoder = BoardEncoder()
        import chess
        encoded = encoder.encode(chess.Board()).unsqueeze(0)
        policy, value = net(encoded)
        assert policy.shape[1] == cfg.action_space_size
        assert value.shape[1] == 1
        print(f"  ✅ 神经网络正常 (参数: {sum(p.numel() for p in net.parameters()):,})")
    except Exception as e:
        print(f"  ❌ 神经网络异常: {e}")
        traceback.print_exc()

    # 测试 3：MCTS
    print("\n[3/6] 测试 MCTS...")
    try:
        from chessmate.training.mcts import MCTS
        cfg.mcts_simulations = 10
        mcts = MCTS(net, encoder, cfg)
        board = chess.Board()
        move, value = mcts.search(board)
        assert move is not None
        print(f"  ✅ MCTS 正常 (推荐走法: {move}, 估值: {value:.3f})")
    except Exception as e:
        print(f"  ❌ MCTS 异常: {e}")
        traceback.print_exc()

    # 测试 4：经验回放
    print("\n[4/6] 测试经验回放...")
    try:
        from chessmate.training.replay_buffer import ReplayBuffer, GameCollector
        import numpy as np

        buffer = ReplayBuffer(capacity=100, config=cfg)
        board = chess.Board()
        for _ in range(20):
            state = encoder.encode(board)
            policy = np.random.rand(cfg.action_space_size).astype(np.float32)
            policy /= policy.sum()
            buffer.add(state, policy, 0.5)

        states, policies, values = buffer.sample(8)
        assert states.shape[0] == 8
        print(f"  ✅ 经验回放正常 (缓冲区大小: {len(buffer)})")
    except Exception as e:
        print(f"  ❌ 经验回放异常: {e}")
        traceback.print_exc()

    # 测试 5：视觉识别
    print("\n[5/6] 测试视觉识别...")
    try:
        from chessmate.vision.board_detector import BoardDetector
        from chessmate.vision.piece_classifier import PieceClassifier, validate_fen

        detector = BoardDetector(cfg)
        classifier = PieceClassifier(cfg)

        # 测试手动区域
        test_image = np.zeros((500, 500, 3), dtype=np.uint8)
        region = detector.detect_manual(test_image)

        center = region.get_square_center(0, 0)
        assert center[0] > 0 and center[1] > 0
        print(f"  ✅ 棋盘检测正常 (a1 中心: {center})")

        # 测试 FEN 验证
        assert validate_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        print(f"  ✅ FEN 验证正常")
    except Exception as e:
        print(f"  ❌ 视觉识别异常: {e}")
        traceback.print_exc()

    # 测试 6：网页对弈（坐标映射）
    print("\n[6/6] 测试网页对弈坐标...")
    try:
        from chessmate.web.web_player import WebPlayer
        import chess as ch

        player = WebPlayer(net, cfg)
        a1 = player._get_screen_coords(ch.A1)
        h8 = player._get_screen_coords(ch.H8)
        assert a1[0] >= 0 and a1[1] >= 0
        assert h8[0] > a1[0]  # h8 在 a1 右边
        print(f"  ✅ 坐标映射正常 (a1: {a1}, h8: {h8})")
    except Exception as e:
        print(f"  ❌ 坐标映射异常: {e}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    print("诊断测试完成！如果所有项目均为 ✅，系统可正常运行。")
    print("=" * 60)


def main():
    """主函数：解析参数并路由到相应的处理函数。"""
    parser = argparse.ArgumentParser(
        description="ChessMate 国际象棋 AI 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py              # 交互式菜单
  python main.py train        # 训练模式
  python main.py gui          # 本地对弈 GUI
  python main.py web          # 网页对战
  python main.py check        # 环境检测
  python main.py calibrate    # 位置校准
  python main.py test         # 诊断测试
        """
    )
    parser.add_argument(
        'mode', nargs='?', default=None,
        choices=['train', 'gui', 'web', 'check', 'calibrate', 'test'],
        help='运行模式 (无参数时显示交互式菜单)'
    )
    parser.add_argument(
        '--config', type=str, default=None,
        help='自定义配置文件路径 (YAML/JSON，未来扩展)'
    )
    parser.add_argument(
        '--model', type=str, default=None,
        help='要加载的模型文件路径'
    )
    parser.add_argument(
        '--small', action='store_true',
        help='使用小规模配置（用于快速测试）'
    )
    parser.add_argument(
        '--cpu', action='store_true',
        help='强制使用 CPU（不使用 GPU）'
    )

    args = parser.parse_args()

    # 确定运行模式
    mode = args.mode
    if mode is None:
        show_banner()
        mode = show_menu()

    if mode == 'exit':
        print("再见！")
        return

    # 执行对应的模式
    mode_handlers = {
        'check': run_check,
        'train': run_train,
        'gui': run_gui,
        'web': run_web,
        'calibrate': run_calibrate,
        'test': run_test,
    }

    handler = mode_handlers.get(mode)
    if handler:
        try:
            handler()
        except KeyboardInterrupt:
            print("\n\n用户中断。")
        except Exception as e:
            print(f"\n运行出错: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"未知模式: {mode}")


if __name__ == "__main__":
    main()