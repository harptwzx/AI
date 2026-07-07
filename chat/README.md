# English ChatAI — 从零构建生成式 AI

纯 TensorFlow 实现，不依赖任何外部模型框架。

## 项目结构

```
.
├── data/
│   ├── raw/              # 原始数据（手动下载）
│   │   ├── tatoeba/
│   │   ├── openwebtext/
│   │   ├── wikipedia/
│   │   ├── gutenberg/
│   │   └── c4/
│   ├── processed/        # 预处理后的数据
│   │   ├── train.txt
│   │   └── val.txt
│   └── tokenizer/        # Tokenizer 文件
│       ├── vocab.json
│       ├── merges.txt
│       └── tokenizer_config.json
├── models/
│   ├── checkpoints/      # 定期保存的检查点
│   ├── best_model/       # 验证集上最好的模型
│   └── final_model/      # 最终模型
├── logs/                 # TensorBoard 日志
├── config.json           # 训练配置
├── data_downloader.py    # 数据下载说明
├── data_processor.py     # 数据预处理
├── tokenizer.py          # BPE Tokenizer
├── model.py              # Transformer 模型
├── train.py              # 训练脚本
├── generate.py           # 文本生成
├── run_pipeline.py       # 一键运行
└── requirements.txt      # 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载数据

运行 `data_downloader.py` 查看下载说明，然后手动下载数据到 `data/raw/` 对应目录：

```bash
python data_downloader.py
```

推荐下载：
- **Tatoeba** (https://tatoeba.org/en/downloads) — 语法规范
- **OpenWebText** (https://skylion007.github.io/OpenWebTextCorpus/) — 语料丰富
- **Wikipedia** (https://dumps.wikimedia.org/enwiki/latest/) — 知识密集

### 3. 一键运行完整流程

```bash
# 全部流程：数据处理 -> Tokenizer 训练 -> 模型训练
python run_pipeline.py --stage all

# 或分步执行
python run_pipeline.py --stage data       # 仅数据预处理
python run_pipeline.py --stage tokenizer  # 仅训练 Tokenizer
python run_pipeline.py --stage train      # 仅训练模型
```

### 4. 断点续训

```bash
# 从检查点继续训练
python run_pipeline.py --stage train --resume ./models/checkpoints/checkpoint_epoch_010

# 或直接用 train.py
python train.py --resume ./models/checkpoints/checkpoint_epoch_010
```

### 5. 文本生成

```bash
# 交互式对话
python generate.py --model ./models/best_model --tokenizer ./data/tokenizer

# 单次生成
python generate.py --model ./models/best_model --prompt "Once upon a time" --max-tokens 50
```

## 训练配置

编辑 `config.json` 调整超参数：

```json
{
  "model": {
    "vocab_size": 32000,
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 6,
    "dff": 2048,
    "max_seq_len": 256
  },
  "training": {
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 5
  }
}
```

## 模型保存

- **SaveModel 格式**: TensorFlow 原生格式，支持完整保存模型结构和权重
- **检查点**: 每 epoch 自动保存到 `models/checkpoints/`
- **最佳模型**: 验证 loss 最低时自动保存到 `models/best_model/`
- **最终模型**: 训练结束时保存到 `models/final_model/`

## 防止过拟合

- **Early Stopping**: 验证 loss 5 个 epoch 不改善则停止
- **Dropout**: 模型中已内置
- **验证集**: 自动划分 5% 数据用于验证
- **学习率衰减**: Warmup + Cosine Decay

## Tokenizer 文件

训练完成后会在 `data/tokenizer/` 生成：
- `vocab.json`: `{token_id: token_str}` 对照表
- `merges.txt`: BPE 合并规则
- `tokenizer_config.json`: 配置信息
