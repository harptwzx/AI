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

这会从 ManyThings.org 自动下载 10 个英语句子数据集。

### 第三步：处理数据 → 训练 Tokenizer → 训练模型

```bash
# 1. 处理数据
python process_data.py

# 2. 训练 Tokenizer（默认 CharTokenizer，瞬间完成）
python build_tokenizer.py

# 3. 训练模型
python train.py
```

## Tokenizer 说明

本项目支持两种 Tokenizer：

### 1. CharTokenizer（默认，推荐）
- **训练时间**: 瞬间完成（< 1 秒）
- **vocab 大小**: ~100-150（取决于数据中的字符种类）
- **原理**: 每个字符一个 token
- **优点**: 零训练时间，简单可靠
- **缺点**: 序列较长（每个字符一个 token）

```bash
python build_tokenizer.py --type char
```

### 2. BPETokenizer（可选）
- **训练时间**: 数小时~数天（取决于数据量和 vocab 大小）
- **vocab 大小**: 可配置（默认 32000）
- **原理**: 合并高频字符对，逐步构建词表
- **优点**: 序列更短，压缩率高
- **缺点**: 训练极慢，纯 Python 实现不适合大数据集

```bash
# 完整 BPE（非常慢！）
python build_tokenizer.py --type bpe

# 限制合并次数加速（只合并 1000 次）
python build_tokenizer.py --type bpe --max-merges 1000
```

> 💡 **建议**: 先用 CharTokenizer 快速开始训练模型，后续如果需要更好的效果再尝试 BPE。

## 断点续训

```bash
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
│   ├── raw/manythings/          # ManyThings 原始数据
│   ├── processed/               # train.txt / val.txt
│   └── tokenizer/                 # Tokenizer 文件
│       ├── tokenizer_config.json
│       └── vocab_readable.txt
├── models/
│   ├── checkpoints/
│   ├── best_model/
│   └── final_model/
├── download_data.py             # 自动下载数据
├── process_data.py              # 数据处理
├── build_tokenizer.py           # Tokenizer 训练
├── tokenizer.py                 # Tokenizer 实现（Char + BPE）
├── model.py                     # Transformer 模型
├── train.py                     # 训练脚本
├── generate.py                  # 文本生成
├── run_pipeline.py              # 一键运行
└── config.json                  # 配置
```

## 训练配置

编辑 `config.json` 调整超参数。

## 模型保存

- **SaveModel 格式**: TensorFlow 原生格式
- **检查点**: `models/checkpoints/`
- **最佳模型**: `models/best_model/`（验证 loss 最低）
- **最终模型**: `models/final_model/`

## 防止过拟合

- Early Stopping (patience=5)
- Dropout (rate=0.1)
- 5% 验证集
- Warmup + Cosine Decay 学习率
