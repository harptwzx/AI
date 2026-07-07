"""
Transformer 语言模型（纯 TensorFlow 实现）
=========================================
不依赖任何外部模型框架，纯手搓 Transformer Decoder。
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Tuple

# ============================================================
# 1. 位置编码
# ============================================================
class PositionalEncoding(tf.keras.layers.Layer):
    """正弦/余弦位置编码"""

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.d_model = d_model

        # 预计算位置编码矩阵
        pos = np.arange(max_seq_len)[:, np.newaxis]  # (max_seq_len, 1)
        i = np.arange(d_model)[np.newaxis, :]        # (1, d_model)

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # 偶数索引用 sin，奇数索引用 cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = tf.constant(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]


# ============================================================
# 2. 多头自注意力
# ============================================================
class MultiHeadAttention(tf.keras.layers.Layer):
    """多头因果自注意力"""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        """将最后一个维度拆分为 (num_heads, depth)"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])  # (batch, heads, seq, depth)

    def scaled_dot_product_attention(
        self, 
        q: tf.Tensor, 
        k: tf.Tensor, 
        v: tf.Tensor, 
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        缩放点积注意力

        Args:
            q: query, shape (..., seq_len_q, depth)
            k: key, shape (..., seq_len_k, depth)
            v: value, shape (..., seq_len_v, depth)
            mask: 掩码
        """
        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_q, seq_k)

        # 缩放
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        # 应用掩码（因果掩码）
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # softmax
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, v)  # (..., seq_q, depth)
        return output, attention_weights

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False):
        batch_size = tf.shape(x)[0]

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask, training
        )

        # 合并多头
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)
        return output, attention_weights


# ============================================================
# 3. 前馈网络
# ============================================================
class FeedForward(tf.keras.layers.Layer):
    """位置-wise 前馈网络"""

    def __init__(self, d_model: int, dff: int, dropout_rate: float = 0.1):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(dff, activation="relu")
        self.dense2 = tf.keras.layers.Dense(d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, training: bool = False):
        x = self.dense1(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        return x


# ============================================================
# 4. Transformer 解码器块
# ============================================================
class TransformerBlock(tf.keras.layers.Layer):
    """单个 Transformer Decoder 块"""

    def __init__(
        self, 
        d_model: int, 
        num_heads: int, 
        dff: int, 
        dropout_rate: float = 0.1
    ):
        super().__init__()

        self.att = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForward(d_model, dff, dropout_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False):
        # 自注意力 + 残差连接
        attn_output, _ = self.att(x, mask, training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        # 前馈 + 残差连接
        ffn_output = self.ffn(out1, training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# ============================================================
# 5. 完整 GPT 模型
# ============================================================
class GPTModel(tf.keras.Model):
    """
    GPT 风格的 Transformer 语言模型

    Args:
        vocab_size: 词表大小
        d_model: 模型维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        dff: 前馈网络隐藏层维度
        max_seq_len: 最大序列长度
        dropout_rate: Dropout 比率
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dff: int = 2048,
        max_seq_len: int = 512,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token 嵌入
        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model)

        # 位置编码
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)

        # Dropout
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        # Transformer 块堆叠
        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        # 最终 LayerNorm
        self.final_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        # 输出层（投影到词表）
        self.output_projection = tf.keras.layers.Dense(vocab_size)

    def create_look_ahead_mask(self, size: int) -> tf.Tensor:
        """
        创建因果（look-ahead）掩码
        上三角为 1（遮挡），下三角为 0（可见）
        """
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)

    def call(self, inputs: tf.Tensor, training: bool = False):
        """
        前向传播

        Args:
            inputs: (batch_size, seq_len) token ids
            training: 是否训练模式

        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        seq_len = tf.shape(inputs)[1]

        # Token 嵌入 + 缩放
        x = self.token_embedding(inputs)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # 位置编码
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        # 因果掩码
        look_ahead_mask = self.create_look_ahead_mask(seq_len)

        # 通过所有 Transformer 块
        for block in self.transformer_blocks:
            x = block(x, mask=look_ahead_mask, training=training)

        # 最终 LayerNorm
        x = self.final_layernorm(x)

        # 投影到词表
        logits = self.output_projection(x)

        return logits

    def get_config(self):
        """用于模型保存/加载"""
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": len(self.transformer_blocks),
            "dff": self.transformer_blocks[0].ffn.dense1.units,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout.rate,
        })
        return config


# ============================================================
# 6. 学习率调度（Warmup + Cosine Decay）
# ============================================================
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """带 Warmup 的余弦衰减学习率调度"""

    def __init__(
        self,
        d_model: int,
        warmup_steps: int = 4000,
        max_steps: int = 100000,
        min_lr_ratio: float = 0.1
    ):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup 阶段
        warmup_lr = step * (self.d_model ** -0.5) / self.warmup_steps

        # Cosine decay 阶段
        progress = tf.minimum((step - self.warmup_steps) / (self.max_steps - self.warmup_steps), 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * progress))
        decayed = (1 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio

        base_lr = self.d_model ** -0.5
        decay_lr = base_lr * decayed

        # 选择 warmup 或 decay
        lr = tf.where(step < self.warmup_steps, warmup_lr, decay_lr)
        return lr

    def get_config(self):
        return {
            "d_model": int(self.d_model),
            "warmup_steps": self.warmup_steps,
            "max_steps": self.max_steps,
            "min_lr_ratio": self.min_lr_ratio,
        }


# ============================================================
# 7. 模型工具函数
# ============================================================
def create_model(config: dict) -> GPTModel:
    """根据配置创建模型"""
    return GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dff=config["dff"],
        max_seq_len=config["max_seq_len"],
        dropout_rate=config.get("dropout_rate", 0.1),
    )


def count_parameters(model: tf.keras.Model) -> int:
    """统计模型参数量"""
    return sum([tf.keras.utils.count_params(w) for w in model.trainable_weights])


if __name__ == "__main__":
    # 测试模型
    print("=" * 60)
    print("测试 GPTModel")
    print("=" * 60)

    config = {
        "vocab_size": 1000,
        "d_model": 128,
        "num_heads": 4,
        "num_layers": 2,
        "dff": 512,
        "max_seq_len": 64,
        "dropout_rate": 0.1,
    }

    model = create_model(config)

    # 测试前向传播
    test_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
    output = model(test_input, training=False)
    print(f"输入 shape: {test_input.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")

    # 测试学习率调度
    lr_schedule = WarmupCosineDecay(d_model=128, warmup_steps=100, max_steps=1000)
    print(f"\n学习率调度测试:")
    for step in [0, 50, 100, 500, 1000]:
        print(f"  step={step}: lr={lr_schedule(step).numpy():.6f}")
