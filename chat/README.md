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

### 第三步：处理数据 → 训练 Tokenizer → 训练模型

```bash
# 1. 处理数据（默认全部，可折半）
python process_data.py              # 全部数据
python process_data.py --sample 0.5  # 只用 50% 数据（约 9MB）

# 2. 训练 Tokenizer（默认 WordTokenizer，瞬间完成）
python build_tokenizer.py                    # 全部数据
python build_tokenizer.py --sample 0.5      # 只用 50% 数据
python build_tokenizer.py --min-freq 10     # 提高单词频率阈值（减少 vocab）

# 3. 训练模型
python train.py
```

## Tokenizer 说明

### WordTokenizer（默认，强烈推荐）

**策略：**
- 高频单词（频率 >= 5）→ 独立 token
- 中频单词 → 前缀/后缀拆分（如 "unhappiness" → "un" + "happiness"）
- 罕见词/专有名词 → 拆为字符

**优势：**
- 训练时间：**< 1 秒**
- 序列长度：比字符级短 **3-5 倍**
- 能学到单词级语法模式
- 自动统计词频，生成完整 token 映射文件

**生成的文件：**
```
data/tokenizer/
├── tokenizer_config.json    # 配置 + 完整映射表
├── vocab_readable.txt       # 可读 token 对照表（含频率）
├── word_list.txt           # 纯单词列表（按频率排序）
└── test_results.txt        # 编码解码测试结果
```

```bash
# 查看生成的 token 文件
cat data/tokenizer/word_list.txt        # 所有单词 token
cat data/tokenizer/vocab_readable.txt   # 完整对照表
```

### CharTokenizer

- 每个字符一个 token
- 训练时间：0 秒
- 序列很长，训练慢

```bash
python build_tokenizer.py --type char
```

### BPETokenizer

- 训练极慢（数小时~数天）
- 不推荐

```bash
python build_tokenizer.py --type bpe --max-merges 1000
```

## 数据量控制

```bash
# 处理数据时折半
python process_data.py --sample 0.5

# Tokenizer 也折半（保持一致）
python build_tokenizer.py --sample 0.5

# 同时提高单词频率阈值（进一步减少 vocab）
python build_tokenizer.py --sample 0.5 --min-freq 10
```

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
│   └── tokenizer/               # Tokenizer 文件
│       ├── tokenizer_config.json
│       ├── vocab_readable.txt
│       ├── word_list.txt
│       └── test_results.txt
├── models/
│   ├── checkpoints/
│   ├── best_model/
│   └── final_model/
├── download_data.py
├── process_data.py
├── build_tokenizer.py
├── tokenizer.py
├── model.py
├── train.py
├── generate.py
└── config.json
```

## 训练配置

编辑 `config.json`：

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

- **SaveModel 格式**: TensorFlow 原生格式
- **检查点**: `models/checkpoints/`
- **最佳模型**: `models/best_model/`（验证 loss 最低）
- **最终模型**: `models/final_model/`

## 防止过拟合

- Early Stopping (patience=5)
- Dropout (rate=0.1)
- 5% 验证集
- Warmup + Cosine Decay 学习率
