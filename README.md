# ChessMate - 国际象棋 AI 系统

<p align="center">
  <strong>训练 | 对弈 | 征服</strong>
</p>

ChessMate 是一个完整的国际象棋 AI 系统，基于 AlphaZero 风格的强化学习算法构建。它包含三大核心功能：自行训练模型、本地 GUI 对弈、以及通过屏幕截图识别网页棋局并自动走子。

---

## 目录

- [核心功能](#核心功能)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装步骤](#安装步骤)
  - [环境检测](#环境检测)
- [使用指南](#使用指南)
  - [训练模式](#训练模式)
  - [本地对弈 (GUI)](#本地对弈-gui)
  - [网页对战](#网页对战)
  - [命令行参数](#命令行参数)
- [配置说明](#配置说明)
  - [训练参数](#训练参数)
  - [MCTS 参数](#mcts-参数)
  - [视觉识别参数](#视觉识别参数)
  - [网页交互参数](#网页交互参数)
  - [GUI 参数](#gui-参数)
- [模块说明](#模块说明)
- [算法简介](#算法简介)
- [常见问题 (FAQ)](#常见问题-faq)
- [未来计划](#未来计划)
- [更新日志](#更新日志)
- [许可证](#许可证)

---

## 核心功能

### 1. 强化学习训练 🧠
基于 AlphaZero 算法的完整训练流程：
- **神经网络**：残差卷积网络，双头架构（策略头 + 价值头）
- **MCTS**：蒙特卡洛树搜索，PUCT 公式引导探索
- **自对弈**：AI 与自己对弈生成训练数据
- **经验回放**：循环缓冲区存储和采样训练样本
- **完整训练循环**：自对弈 → 经验收集 → 网络更新 → 评估

### 2. 本地 GUI 对弈 ♟️
基于 PyQt5 的图形界面：
- 渲染标准 8×8 棋盘，Unicode 棋子符号
- 鼠标点击走子（点击起始格 + 目标格）
- 合法走法高亮提示
- 走子历史记录显示
- 支持玩家选择执白或执黑
- AI 后台线程搜索，不冻结界面
- 棋盘翻转（黑方视角）
- 对局状态实时显示

### 3. 网页自动对战 🌐
通过屏幕截图识别棋局 + 自动鼠标操作：
- 截图分析网页棋盘，提取 FEN 字符串
- AI 计算最优走法
- 自动鼠标点击走子
- 交互式棋盘位置校准工具
- 支持执白/执黑两种角色

---

## 项目结构

```
ChessTrainingProject/
├── main.py                          # 主入口脚本
├── check_env.py                     # 环境检测脚本
├── requirements.txt                 # Python 依赖清单
├── README.md                        # 本文档
├── CHANGELOG.md                     # 更新日志
│
├── chessmate/                       # 核心包
│   ├── __init__.py
│   ├── config.py                    # 全局配置（所有可调参数）
│   │
│   ├── training/                    # 训练模块
│   │   ├── __init__.py
│   │   ├── neural_net.py           # 神经网络定义（ChessNet + BoardEncoder）
│   │   ├── mcts.py                 # MCTS 蒙特卡洛树搜索
│   │   ├── self_play.py            # 自对弈引擎
│   │   ├── replay_buffer.py        # 经验回放缓冲区
│   │   └── trainer.py              # 训练循环编排器
│   │
│   ├── vision/                      # 视觉识别模块
│   │   ├── __init__.py
│   │   ├── board_detector.py       # 棋盘检测（定位 + 坐标映射）
│   │   └── piece_classifier.py     # 棋子识别（颜色分析 + 模板匹配）
│   │
│   ├── gui/                         # GUI 模块
│   │   ├── __init__.py
│   │   └── chess_window.py         # PyQt5 对弈窗口
│   │
│   └── web/                         # 网页交互模块
│       ├── __init__.py
│       └── web_player.py           # 网页自动对弈器
│
├── models/                          # 模型权重文件存放目录
├── logs/                            # 训练日志和 TensorBoard 数据
├── data/
│   └── templates/                   # 棋子模板图片（可选）
└── .gitignore
```

---

## 快速开始

### 环境要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 11（也可用于 Linux/macOS） |
| **Python** | >= 3.9 |
| **Conda 环境** | 推荐使用 `chess_ai` |
| **GPU (可选)** | NVIDIA GPU + CUDA 12.x（用于加速训练） |
| **内存** | 至少 8 GB RAM（训练建议 16 GB+） |

### 安装步骤

**1. 创建并激活 conda 环境（推荐）**
```bash
conda create -n chess_ai python=3.10 -y
conda activate chess_ai
```

**2. 安装 PyTorch（GPU 版本）**
```bash
# CUDA 12.x
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或仅 CPU 版本
pip install torch torchvision torchaudio
```

**3. 安装 ChessMate 依赖**
```bash
pip install -r requirements.txt

# 以及可选依赖（如果不使用 requirements.txt 中的全部）:
pip install PyQt5 opencv-python pyautogui scikit-learn pyyaml
```

**4. 克隆/进入项目目录**
```bash
cd ChessTrainingProject
```

### 环境检测

安装完成后，运行环境检测脚本确保一切就绪：

```bash
python check_env.py
```

预期输出：
```
=================================================================
  ChessMate 环境检测报告
=================================================================
  Python 版本 : 3.10.19
  Conda 环境   : chess_ai

=================================================================
  3) 核心依赖检查 (必需)
=================================================================
  ✅ numpy                     2.1.2           -- 数值计算基础库
  ✅ torch                     2.8.0+cu128     -- PyTorch 深度学习框架
  ✅ python-chess              1.999           -- 国际象棋规则引擎
  ✅ tqdm                      4.67.1          -- 进度条显示

=================================================================
  6) 汇总与建议
=================================================================
  ✅ 所有必需依赖已安装，环境就绪！
```

---

## 使用指南

### 训练模式

启动训练最简单的方式：

```bash
python main.py train
```

系统会询问训练规模：
- **小规模测试**：CPU 友好，快速验证流程（推荐首次使用）
- **中等规模**：需要 GPU，训练更有意义的模型
- **自定义**：使用 `chessmate/config.py` 中的默认参数

**训练参数示例（小规模测试）：**
- 迭代轮数：10
- 每轮自对弈：5 局
- MCTS 模拟：50 次
- 残差块：4 个
- 滤波器：64

**监控训练进度：**
训练日志会同时输出到控制台和 `logs/chessmate.log` 文件。

启用 TensorBoard 可视化（需要安装 tensorboard）：
```bash
pip install tensorboard
tensorboard --logdir=logs/tensorboard
```

**从检查点恢复训练：**
检查点自动保存在 `models/latest_model.pth`，训练器会自动使用。

---

### 本地对弈 (GUI)

```bash
python main.py gui
```

**操作说明：**

1. 选择执子颜色（执白/执黑）
2. 点击棋盘上的棋格进行走子：
   - **第一次点击**：选中棋子（该棋子的合法走法会高亮显示）
   - **第二次点击**：点击目标格执行走子
3. AI 自动回应（后台搜索，不冻结界面）
4. 走子历史实时显示在右侧面板
5. 点击"翻转棋盘"可从黑方视角查看
6. 点击"新对局"重置棋盘

**键盘快捷键：**
- `Ctrl+N`：新对局（计划中）
- `Ctrl+Z`：悔棋（计划中）

---

### 网页对战

**前置准备：**
1. 在浏览器中打开国际象棋网站（如 chess.com、lichess.org）
2. 确保棋盘完整可见

**步骤：**

1. **校准棋盘位置**
   ```bash
   python main.py calibrate
   ```
   按提示将鼠标移到棋盘的 a1 格和 h8 格，系统自动计算棋盘参数。

2. **启动网页对战**
   ```bash
   python main.py web
   ```
   选择执白（AI 后手）或执黑（AI 先手）。

3. AI 会自动截图、识别局面、计算并走子。

**注意：**
- 确保目标网站允许使用自动化工具
- 建议在友谊赛或与机器人对弈时使用
- 遵守网站的使用条款
- 首次使用需仔细校准，可多次微调

---

### 命令行参数

```bash
python main.py [模式] [选项]

模式:
  train        训练模式
  gui          本地对弈 GUI
  web          网页对战
  check        环境检测
  calibrate    网页棋盘位置校准
  test         全模块诊断测试

选项:
  --config PATH    自定义配置文件路径
  --model PATH     加载指定的模型文件
  --small          使用小规模配置
  --cpu            强制使用 CPU
```

**使用示例：**
```bash
python main.py              # 交互式菜单
python main.py test         # 运行诊断测试
python main.py train --small  # 使用小规模配置训练
python main.py gui          # 启动 GUI
python main.py web          # 启动网页对战
```

---

## 配置说明

所有可调参数集中在 `chessmate/config.py` 的 `ChessConfig` 类中。

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_res_blocks` | 6 | 残差块数量。AlphaZero 用 19/39，小实验建议 4-8 |
| `num_filters` | 128 | 卷积滤波器数量。越大模型越强但越慢 |
| `learning_rate` | 0.001 | 初始学习率（AdamW 优化器） |
| `weight_decay` | 1e-4 | L2 正则化系数 |
| `num_train_epochs` | 5 | 每轮数据的训练轮数 |
| `max_training_iterations` | 50 | 最大训练迭代轮数 |
| `num_self_play_games` | 10 | 每轮自对弈局数 |
| `replay_buffer_capacity` | 100000 | 经验回放池最大容量 |
| `replay_batch_size` | 256 | 训练批量大小 |
| `checkpoint_frequency` | 5 | 每隔多少轮保存检查点 |

### MCTS 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mcts_simulations` | 200 | 每步棋的搜索模拟次数。越多越强但越慢 |
| `mcts_c_puct` | 1.0 | PUCT 探索常数 |
| `mcts_temperature` | 1.0 | 温度参数（>1 探索，<1 利用） |
| `mcts_dirichlet_alpha` | 0.3 | Dirichlet 噪声 alpha |
| `mcts_dirichlet_epsilon` | 0.25 | Dirichlet 噪声混合比例 |
| `max_moves_per_game` | 200 | 每局棋最大步数 |
| `resign_threshold` | -0.95 | 自动认输估值阈值 |

### 视觉识别参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `web_board_top_left` | (100, 200) | 网页棋盘左上角屏幕坐标 |
| `web_square_size` | 75 | 网页棋盘每个格子的像素大小 |
| `vision_board_light_color` | (240, 217, 181) | 浅色格参考 RGB |
| `vision_board_dark_color` | (181, 136, 99) | 深色格参考 RGB |
| `vision_square_size_min` | 40 | 格子最小像素（自动检测用） |
| `vision_square_size_max` | 120 | 格子最大像素（自动检测用） |

### 网页交互参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `web_click_delay` | 0.3 | 两次点击间隔（秒），避免过快被检测 |
| `web_move_duration` | 0.5 | 拖拽走子持续时长（秒） |

### GUI 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `gui_window_width` | 900 | 窗口宽度 |
| `gui_window_height` | 700 | 窗口高度 |
| `gui_square_size` | 70 | 棋盘格子像素大小 |
| `gui_player_color` | True | True=玩家执白, False=玩家执黑 |
| `gui_ai_thinking_time` | 1.0 | AI 思考最短显示时间（秒） |
| `gui_show_coordinates` | True | 是否显示坐标标签 |

---

## 模块说明

### 训练模块 (`chessmate/training/`)

| 文件 | 说明 |
|------|------|
| `neural_net.py` | AlphaZero 风格残差 CNN：输入(119,8,8)→残差塔→策略头+价值头。含 BoardEncoder 棋盘编码器和 ChessLoss 损失函数 |
| `mcts.py` | 蒙特卡洛树搜索：Selection→Expansion→Evaluation→Backpropagation。使用 PUCT 公式选择走法 |
| `self_play.py` | 自对弈引擎：AI 与自己对弈，使用温度控制探索，收集 (state, policy, value) 经验 |
| `replay_buffer.py` | 经验回放缓冲区：循环缓冲 + 随机采样。含 GameCollector 对局收集器 |
| `trainer.py` | 训练编排器：自对弈→训练→检查点。含 AdamW 优化器、余弦学习率调度、梯度裁剪 |

### 视觉识别模块 (`chessmate/vision/`)

| 文件 | 说明 |
|------|------|
| `board_detector.py` | 棋盘检测：支持手动区域配置和自动轮廓检测。提供方格坐标映射 |
| `piece_classifier.py` | 棋子识别：颜色分析（简单）+ 模板匹配（可选）。输出 FEN 字符串。含 VisionPipeline 流水线 |

### GUI 模块 (`chessmate/gui/`)

| 文件 | 说明 |
|------|------|
| `chess_window.py` | PyQt5 对弈窗口：棋盘绘制+Unicode棋子+交互逻辑+AI后台线程+走子历史 |

### 网页交互模块 (`chessmate/web/`)

| 文件 | 说明 |
|------|------|
| `web_player.py` | 网页对弈器：截图识别→AI搜索→鼠标走子。支持执白/执黑。含校准工具 |

---

## 算法简介

### AlphaZero 核心思想

ChessMate 的训练算法基于 DeepMind 的 AlphaZero 论文：

1. **神经网络 fθ(s) = (p, v)**
   - 输入：棋盘状态 s（119 特征平面）
   - 输出：走子概率 p（策略头）+ 局面价值 v（价值头）

2. **MCTS 搜索**
   - 每步棋执行 N 次模拟
   - 每次模拟用神经网络评估叶节点
   - 使用 PUCT 公式平衡探索与利用

3. **自对弈训练**
   - 当前最佳模型与自己对弈
   - MCTS 搜索概率作为策略目标
   - 对局结果作为价值目标

4. **神经网络更新**
   - 损失 = 交叉熵(策略) + MSE(价值)

### 训练数据流

```
自对弈 → (state, policy, value) → 回放缓冲区 → 随机采样 → 训练网络 → 更新模型 → 自对弈...
```

---

## 常见问题 (FAQ)

### Q: 训练需要多长时间？
**A:** 取决于配置。小规模测试（CPU）约需数十分钟。要训练有意义的模型，建议在有 GPU 的环境下运行 50+ 轮，每轮 20 局以上。

### Q: GPU 显存需要多大？
**A:** 默认配置约需 2-4 GB 显存。降低 `num_filters` 和 `replay_batch_size` 可减少显存占用。

### Q: 网页对战无法识别棋子怎么办？
**A:** 
1. 重新运行校准工具 `python main.py calibrate`
2. 确保棋盘在截图区域内完全可见
3. 调整 `vision_board_light_color` 和 `vision_board_dark_color` 以匹配目标网站的棋盘颜色
4. 在 `data/templates/` 中添加目标网站的棋子截图作为模板

### Q: GUI 中 Unicode 棋子显示为方框？
**A:** 安装支持 Unicode 国际象棋符号的字体（如 Segoe UI Symbol）。Windows 11 通常已自带。

### Q: 如何在两个 AI 模型之间对战？
**A:** 当前版本尚未直接支持。可以通过修改 `chess_window.py` 中的 AI 初始化逻辑来实现。

### Q: 可以从棋谱（PGN）初始化训练数据吗？
**A:** 可以！使用 `ReplayBuffer.add_game_experiences()` 方法，先将 PGN 棋谱转换为 (state, policy, value) 格式后导入。

---

## 未来计划

- [ ] 并行自对弈（多进程加速）
- [ ] PGN 棋谱导入/导出
- [ ] 更精确的棋子识别（CNN 分类器 + 训练脚本）
- [ ] 拖拽走子模式（网页对战）
- [ ] GUI 悔棋和提示功能
- [ ] 模型评估对战（Elo 分计算）
- [ ] Web 界面（Flask/FastAPI）
- [ ] 多 GPU 分布式训练支持
- [ ] 开局库集成
- [ ] 残局表基（Endgame Tablebase）支持

---

## 更新日志

详细的版本更新记录请参阅 [CHANGELOG.md](CHANGELOG.md)。

### v0.1.0 (2026-08-03)

- 🎉 初始版本发布
- ✅ AlphaZero 风格神经网络（残差 CNN 双头架构）
- ✅ MCTS 搜索（PUCT 公式 + Dirichlet 噪声）
- ✅ 自对弈引擎（温度控制 + 自动认输）
- ✅ 经验回放缓冲区（循环缓冲 + 随机采样）
- ✅ 完整训练循环（AdamW + 余弦退火 + 检查点）
- ✅ PyQt5 GUI 本地对弈窗口
- ✅ 屏幕截图识别 + 鼠标自动走子
- ✅ 交互式棋盘位置校准工具
- ✅ 环境检测脚本
- ✅ 全局配置文件系统
- ✅ 详细中文文档

---

## 许可证

MIT License

Copyright (c) 2026 ChessMate Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

---

<p align="center">
  Made with ♟️ by ChessMate Team
</p>