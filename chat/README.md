# English ChatAI — 从零构建生成式 AI

纯 TensorFlow 实现，不依赖任何外部模型框架。

## 快速开始（三步搞定）

### 第一步：安装依赖

```bash
pip install -r requirements.txt
```

### 第二步：下载数据（全自动，约 25MB）

```bash
python download_data.py
```

这会从 ManyThings.org 自动下载 10 个英语句子数据集：
- English-German (~7.4MB, 20万句)
- English-French (~5.2MB, 19万句)
- English-Spanish (~3.8MB, 14万句)
- English-Japanese (~2.2MB, 5万句)
- English-Russian (~2.8MB, 50万句)
- English-Portuguese (~2.0MB, 4万句)
- English-Italian (~1.8MB, 3.5万句)
- English-Korean (~1.5MB, 2.5万句)
- English-Dutch (~1.2MB, 2万句)
- English-Chinese (~1.0MB, 1.5万句)

**特点：**
- 句子简短、语法规范（来自 Tatoeba 项目，经人工校对）
- 适合训练语法和语感
- 自动断点续传，下载失败自动重试

### 第三步：处理数据 → 训练 Tokenizer → 训练模型

```bash
# 1. 处理数据（提取英语句子，生成 train.txt / val.txt）
python process_data.py

# 2. 训练 Tokenizer（生成 vocab.json + merges.txt）
python build_tokenizer.py

# 3. 训练模型
python train.py

# 或一键运行全部
python run_pipeline.py --stage all
```

## 断点续训

```bash
# 从检查点继续训练
python train.py --resume ./models/checkpoints/checkpoint_epoch_010
```

## 文本生成

```bash
# 交互式对话
python generate.py --model ./models/best_model --tokenizer ./data/tokenizer

# 单次生成
python generate.py --model ./models/best_model --prompt "Hello" --max-tokens 50
```

## 项目结构

```
.
├── data/
│   ├── raw/
│   │   └── manythings/          # ManyThings 原始数据
│   │       ├── eng_deu/deu.txt
│   │       ├── eng_fra/fra.txt
│   │       ├── eng_spa/spa.txt
│   │       └── ...
│   ├── processed/               # 处理后的数据
│   │   ├── train.txt
│   │   └── val.txt
│   └── tokenizer/               # Tokenizer 文件
│       ├── vocab.json
│       ├── merges.txt
│       ├── tokenizer_config.json
│       └── vocab_readable.txt   # 可读的 token 对照表
├── models/
│   ├── checkpoints/             # 定期保存的检查点
│   ├── best_model/              # 验证集上最好的模型
│   └── final_model/             # 最终模型
├── logs/                        # TensorBoard 日志
├── config.json                  # 训练配置
├── download_data.py             # 自动下载 ManyThings 数据
├── process_data.py              # 数据处理
├── build_tokenizer.py           # 训练 Tokenizer
├── tokenizer.py                 # BPE Tokenizer 实现
├── model.py                     # Transformer 模型
├── train.py                     # 训练脚本
├── generate.py                  # 文本生成
├── run_pipeline.py              # 一键运行
├── requirements.txt             # 依赖
└── README.md                    # 本文件
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
- **Dropout**: 模型中已内置 (rate=0.1)
- **验证集**: 自动划分 5% 数据用于验证
- **学习率衰减**: Warmup + Cosine Decay

## Tokenizer 文件说明

运行 `build_tokenizer.py` 后在 `data/tokenizer/` 生成：
- `vocab.json`: `{token_id: token_str}` 对照表（JSON 格式）
- `merges.txt`: BPE 合并规则（每行一个 pair）
- `tokenizer_config.json`: 配置信息
- **`vocab_readable.txt`**: **纯文本可读版**，方便查看每个 token 是什么
