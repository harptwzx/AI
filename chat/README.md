# English ChatAI — 从零构建生成式 AI

## 快速开始（解决 accuracy 卡住问题）

### 问题：accuracy 卡在 ~0.19 不动

**原因**：模型太小（256维4层）学不动，或学习率不合适

**解决**：用中等模型 + 更高学习率 + 更少数据

### 推荐流程

```bash
# 1. 下载数据
python download_data.py

# 2. 只用 1000-2000 句（先验证模型能学习）
python process_data.py --max-lines 1500

# 3. 训练 WordTokenizer
python build_tokenizer.py --max-lines 1500 --min-freq 2

# 4. 训练（config.json 已更新为中等模型 + 高学习率）
python train.py
```

### 模型配置（config.json）

```json
{
  "model": {
    "d_model": 384,      // 模型维度
    "num_heads": 6,      // 注意力头
    "num_layers": 5,     // Transformer 层数
    "dff": 1536,         // 前馈网络维度
    "max_seq_len": 128   // 序列长度
  },
  "training": {
    "batch_size": 16,
    "learning_rate": 0.001,    // 更高学习率
    "warmup_steps": 500        // 更短 warmup
  }
}
```

### 数据量建议

| 数据量 | 用途 | 预估 1 epoch |
|--------|------|-------------|
| 1000 句 | 快速验证模型能学习 | ~5-10 分钟 |
| 3000 句 | 小规模训练 | ~20-30 分钟 |
| 10000 句 | 中等训练 | ~1-2 小时 |
| 全部 | 完整训练 | ~5+ 小时 |

### 判断模型是否在学习的指标

**正常训练**：
- epoch 1: accuracy ~0.05-0.15, loss ~4-5
- epoch 3-5: accuracy 上升到 ~0.3-0.5, loss 降到 ~2-3
- epoch 10+: accuracy 继续上升

**如果卡住不动**：
- 减小数据量到 1000 句测试
- 检查学习率是否太高/太低
- 增大模型维度

### 完整命令

```bash
# 最小测试（1000句，验证能学习）
python process_data.py --max-lines 1000
python build_tokenizer.py --max-lines 1000
python train.py

# 小训练（3000句）
python process_data.py --max-lines 3000
python build_tokenizer.py --max-lines 3000
python train.py

# 中等训练（10%数据）
python process_data.py --sample 0.1
python build_tokenizer.py --sample 0.1
python train.py
```
