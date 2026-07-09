"""
English ChatAI 训练配置
所有超参数集中管理，方便调整
"""

import os

# ========== 路径配置 ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best_model")

# 创建目录
for d in [DATA_DIR, MODEL_DIR, CHECKPOINT_DIR, BEST_MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

# 数据文件路径
RAW_TEXT_FILE = os.path.join(DATA_DIR, "training_corpus.txt")
TOKENIZER_FILE = os.path.join(DATA_DIR, "tokenizer.json")
VOCAB_FILE = os.path.join(DATA_DIR, "vocab.txt")          # token -> id 对照
MERGES_FILE = os.path.join(DATA_DIR, "merges.txt")        # BPE merges 规则

# ========== Tokenizer 配置 ==========
VOCAB_SIZE = 13947          # 词汇表大小（Oxford 3000 + 常见词 + subword）
MIN_FREQ = 2                # 最小词频

# ========== 模型架构配置 ==========
D_MODEL = 256               # 嵌入维度
NUM_HEADS = 4               # 注意力头数
NUM_LAYERS = 3              # Transformer 层数
DFF = 1024                  # FFN 中间层维度
MAX_SEQ_LEN = 128           # 最大序列长度
DROPOUT_RATE = 0.1          # Dropout 率

# ========== 训练配置 ==========
BATCH_SIZE = 32
EPOCHS = 50                 # 最大 epoch 数（早停会提前终止）
LEARNING_RATE = 3e-4
WARMUP_STEPS = 2000

# 早停 & 模型保存配置
EARLY_STOPPING_PATIENCE = 5     # 验证 loss 不改善的 patience
SAVE_BEST_ONLY = True           # 只保存最佳模型
CHECKPOINT_FREQUENCY = "epoch"  # 每个 epoch 保存检查点

# 验证集比例
VALIDATION_SPLIT = 0.1

# ========== 生成配置 ==========
GENERATION_TEMPERATURE = 0.8
GENERATION_TOP_K = 50
GENERATION_MAX_TOKENS = 100
