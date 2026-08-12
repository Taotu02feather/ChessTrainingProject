# ChessMate 更新日志 (CHANGELOG)

本文档记录 ChessMate 国际象棋 AI 系统的所有重要版本更新。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [v0.2.0] - 2026-08-10

### 🧠 BoardEncoder 历史编码（重大修复）

**问题**：v0.1.0 的 BoardEncoder 只填充了 119 个平面中的 19 个（84% 是零）。神经网络收到的输入绝大多数是垃圾数据，导致模型完全无法学习有意义的棋局评估。

**修复**：重写 BoardEncoder 为 AlphaZero 标准的"局面历史序列"编码——19 个基础平面 × 7 个时间步 = 119 总平面。与旧检查点完全兼容。

### ♟️ 对战 MCTS 优化

**问题**：GUI 和网页对战使用训练模式的 `mcts_simulations=200`，导致每步搜索耗时 60-90 秒，界面卡死。

**修复**：新增 `config.play_mcts_simulations = 40`，`MatchMCTS` 使用对战专用值，模型加载时显式 `model.to("cuda")` 确保 GPU。

### 📊 训练配置增强

mcts_simulations: 60→200 | num_self_play_games: 40→100 | mcts_c_puct: 1.0→1.5 | max_moves_per_game: 150→100 | replay_buffer_capacity: 100k→500k | replay_batch_size: 256→512 | learning_rate: 0.001→0.0005 | max_training_iterations: 50→200 | checkpoint_frequency: 5→2 | num_res_blocks: 12→15 | num_filters: 128→192

### 🔧 统一模型管理器

新建 `chessmate/model_manager.py`：`list_available_models()` 扫描模型目录、`select_model(config)` 交互式选择、`check_and_restore_training(trainer)` 自动恢复训练。

### 🐛 Bug 修复

- **config 序列化破坏整数**：`to_dict()` 中 `isinstance(value, type(logging.INFO))` 错误匹配所有整数，导致 `board_size=8` 被保存为 `"Level 8"`。已修复并添加 `_sanitize_config_value()` 兼容现有检查点。
- **"not all arguments converted during string formatting"**：上述 bug 导致加载模型时报错，已修复。
- **GUI 棋子位置偏移**：`_draw_pieces()` 中阴影偏移量错误，已重写绘制逻辑。
- **直接运行子模块报错 `ModuleNotFoundError`**：所有子模块顶部添加 `sys.path.insert` 自动修正项目根目录。

### 📝 文档

- 更新 README.md：完整使用说明、参数表格、FAQ
- 更新 CHANGELOG.md（本文档）

### 🚀 训练加速与多样性增强（2026-08-10 后续）

**问题**：BoardEncoder 历史栈在 self_play 中未更新，导致 7 个时间步的历史始终为初始局面，119 平面编码形同虚设。同时和棋率 99.8%、每局仅 22 步。

**修复（4 项改动）**：

- **历史栈激活**（`self_play.py`）：每步 `board.push()` 后调用 `self.encoder.push_history(board)`，T=-1 到 T=-6 真正反映棋局动态变化
- **开局随机化**（`self_play.py`）：白黑各随机走 1 步作为开局，打破"千局一面"
- **探索延长**：`temperature_cutoff: 15→40`，`dirichlet_epsilon: 0.25→0.50`，MCTS 更积极尝试非常规走法
- **学习率翻倍**：`learning_rate: 0.0005→0.001`，加速早期收敛

**参数调整汇总**：

| 参数 | 旧值 | 新值 | 说明 |
|------|------|------|------|
| temperature_cutoff | 15 | 40 | 探索延伸到中局 |
| dirichlet_epsilon | 0.25 | 0.50 | MCTS 50% 噪声覆盖 |
| learning_rate | 0.0005 | 0.001 | 学习率翻倍 |
| resign_threshold | -0.95 | -0.99 | 减少误判认输 |
| max_moves_per_game | 100 | 120 | 减少和棋截断 |
| num_self_play_games | 100 | 20 | 每轮更快完成，频繁检查点 |
| checkpoint_frequency | 2 | 1 | 每轮都保存 |

---

## [v0.3.0] - 2026-08-12

### 🎓 监督学习预训练模块（重大新增）

**背景**：从零开始的 AlphaZero 强化学习需要数百万局自对弈（DeepMind 用了 440 万局、5000 个 TPU），个人算力无法复现。实测从零训练 70 轮策略损失仍卡在 3.35（接近随机水平），模型纯随机走子。

**方案**：采用 Leela Chess Zero 等开源项目的标准做法——先监督学习预训练，再强化学习微调。

**新增**：`chessmate/training/pretrain.py` 的 `Pretrainer` 类

- **数据流**：流式读取 PGN 棋谱，逐局提取 (局面, 人类走法, 对局结果) 三元组
- **策略头训练**：交叉熵损失，让模型学会预测人类大师的走法
- **价值头训练**：MSE 损失，预测对局结果（白胜 +1 / 黑胜 -1 / 和 0）
- **历史编码**：复用 BoardEncoder 的 history_stack，每步同步历史状态
- **支持**：单文件 / 目录 / glob 模式，`--max-games` 限制处理量

**集成**：
- `main.py` 新增 `pretrain` 命令，支持 `--pgn`、`--epochs`、`--max-games` 参数
- 新增 `download_pgn.py` 脚本，自动下载大师棋谱（pgnmentor.com）

**实测效果**：
- 快速验证（200 局 1 epoch）：策略损失 8.7 → 5.9
- 完整预训练（2 万局）：策略损失从 8.66 降至 3.9 以下，首次突破强化学习 70 轮未达到的水平

**使用方式**：
```bash
python main.py pretrain --pgn data/pgn --epochs 3   # 监督预训练
python main.py train                                  # 强化学习微调
```

---

## [v0.1.0] - 2026-08-03

### 🎉 初始版本发布

首次公开发布 ChessMate，包含完整的三大核心模块和辅助工具。

### ✨ 新增功能

#### 训练模块 (`chessmate/training/`)
- **神经网络** (`neural_net.py`)
  - AlphaZero 风格残差卷积神经网络（ChessNet）
  - 双头架构：策略头（走子概率）+ 价值头（局面评估）
  - ResBlock 残差块，支持可配置的层数和通道数
  - BoardEncoder 棋盘状态编码器（119 特征平面）
  - 走法 ↔ 动作空间索引的双向映射（move_to_index / index_to_move）
  - ChessLoss 组合损失函数（交叉熵 + MSE）

- **MCTS 搜索** (`mcts.py`)
  - 完整四阶段实现：Selection → Expansion → Evaluation → Backpropagation
  - PUCT 公式节点选择（含可配置的 c_puct 常数）
  - Dirichlet 噪声注入（根节点探索增强）
  - 温度控制采样（支持训练/对战两种模式）
  - MatchMCTS 对战模式变体（低温度 + 无噪声）

- **自对弈引擎** (`self_play.py`)
  - AI 自我对弈生成训练数据
  - 温度切换策略（前 15 步探索，后续利用）
  - 自动认输机制（估值低于阈值）
  - 最大步数保护
  - 对局统计（白胜/黑胜/和棋比例）
  - 批量自对弈（progress bar 显示）

- **经验回放** (`replay_buffer.py`)
  - 循环缓冲区（deque，固定容量）
  - 随机批量采样
  - 序列化/反序列化（pickle 支持）
  - GameCollector 对局经验收集器
  - 统计信息追踪

- **训练器** (`trainer.py`)
  - 完整训练循环编排（self-play → train → checkpoint）
  - AdamW 优化器 + 余弦退火学习率调度
  - 梯度裁剪（max_norm=1.0）
  - TensorBoard 日志支持
  - 检查点保存/恢复（含优化器和调度器状态）
  - 模型评估（vs 随机走法）
  - 最佳模型自动保存
  - quick_train() 便捷训练函数

#### 视觉识别模块 (`chessmate/vision/`)
- **棋盘检测** (`board_detector.py`)
  - 手动区域配置（指定棋盘左上角 + 格子大小）
  - 自动检测（Canny 边缘 + 轮廓四边形近似）
  - 角点排序（左上→右上→右下→左下）
  - 方格坐标映射（square → 屏幕像素）
  - 子图像裁剪提取
  - ChessboardRegion 数据类

- **棋子识别** (`piece_classifier.py`)
  - 颜色分析法（中心区域 vs 边缘比较）
  - 模板匹配法（可选的模板图片对比）
  - FEN 字符串输出
  - FEN 验证（python-chess 校验）
  - FEN ↔ 网格双向转换
  - VisionPipeline 完整流水线

#### 网页交互模块 (`chessmate/web/`)
- **网页对弈器** (`web_player.py`)
  - 截图 + 识别 + AI 搜索 + 鼠标走子完整流水线
  - 两种走子方式：点击（click-click）+ 拖拽（drag）
  - 执白/执黑模式（等待对手 vs 先行）
  - 全自动双模式（测试用）
  - 交互式棋盘位置校准工具（calibrate_position）
  - 坐标映射（python-chess square → 屏幕像素）
  - 延时控制（避免被检测）

#### GUI 模块 (`chessmate/gui/`)
- **PyQt5 对弈窗口** (`chess_window.py`)
  - 标准 8×8 棋盘绘制（QPainter）
  - Unicode 国际象棋符号渲染（♔♕♖♗♘♙ / ♚♛♜♝♞♟）
  - 鼠标交互走子（选中高亮 + 合法走法指示）
  - 兵自动升变为后（简化处理）
  - AI 后台线程搜索（QThread，不冻结 UI）
  - 最短思考时间显示
  - 走子历史（标准代数记谱法，SAN）
  - 棋盘翻转（黑方视角）
  - 玩家执子颜色选择
  - 对局结束处理（将杀/逼和/子力不足）
  - 状态栏实时提示

#### 系统基础
- **主入口** (`main.py`)
  - 交互式菜单（6 种模式）
  - 命令行参数解析（argparse）
  - 训练规模选择（小/中/自定义）
  - ASCII 艺术标题
  - 全模块诊断测试

- **配置系统** (`chessmate/config.py`)
  - ChessConfig 数据类（50+ 可调参数）
  - 预定义配置方案（小/中/完整规模）
  - 日志系统自动配置
  - 目录自动创建
  - 字典导入/导出

- **环境检测** (`check_env.py`)
  - Python 版本检查
  - Conda 环境检测
  - 依赖包逐一验证（含版本号）
  - CUDA/GPU 可用性检测
  - 缺失包安装命令提示
  - 分级报告（必需/可选）

- **依赖管理** (`requirements.txt`)
  - 分类注释（核心/可选/可视化）
  - 最低版本约束

### 📚 文档
- 完整中文 README（安装、使用、配置、FAQ）
- 所有源码含中英文注释
- 每个类和函数均有 docstring
- 每个模块含测试代码（`if __name__ == "__main__"`）

### 🔧 技术细节
- **技术栈**：Python 3.10、PyTorch 2.x、python-chess、PyQt5、OpenCV、pyautogui
- **神经网络**：残差 CNN、Batch Normalization、双头输出（策略 + 价值）
- **优化器**：AdamW with weight decay、余弦退火学习率调度
- **数据流**：自对弈 → GameCollector → ReplayBuffer → 随机采样 → Trainer
- **并行**：GUI 使用 QThread 后台搜索，训练支持 GPU（CUDA）
- **编码规范**：PEP8 风格、类型提示、dataclass、slots 优化

### ⚠️ 已知限制
- 棋盘编码器使用简化特征平面（非完整 AlphaZero 119 平面实现）
- 视觉识别的颜色分析法只能区分空格/棋子/颜色，无法细分棋子类型
- 自对弈为单线程，不支持并行
- GUI 不支持悔棋和提示功能
- 网页对弈仅支持点击走子，拖拽模式尚未完全测试
- 棋子升变自动选后，不支持用户选择

### 🔜 计划中（v0.2.0）
- 完整 AlphaZero 119 平面特征编码（含历史状态）
- 并行自对弈（多进程）
- CNN 棋子分类器（替代颜色分析）
- GUI 悔棋和走子提示
- PGN 导入/导出
- 模型 Elo 评估对战
- 拖拽走子在网页对战中的完整实现

---

## 版本规范

版本号格式：`主版本号.次版本号.修订号`

- **主版本号**：不兼容的 API 修改
- **次版本号**：向下兼容的功能新增
- **修订号**：向下兼容的问题修复

---

## 贡献者

- Taotu - 初始开发和维护

---

*本日志最后更新于 2026-08-12*