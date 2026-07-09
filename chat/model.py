"""
Transformer 语言模型（纯 TensorFlow 实现）
=========================================
不依赖任何外部模型框架，纯手搓 Transformer Decoder。
兼容 Keras 2.x 和 Keras 3.x
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

        pos = np.arange(max_seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]

        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

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
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def scaled_dot_product_attention(
        self, 
        q: tf.Tensor, 
        k: tf.Tensor, 
        v: tf.Tensor, 
        mask: Optional[tf.Tensor] = None,
        training: bool = False
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        output = tf.matmul(attention_weights, v)
        return output, attention_weights

    def call(self, x: tf.Tensor, mask: Optional[tf.Tensor] = None, training: bool = False):
        batch_size = tf.shape(x)[0]

        q = self.split_heads(self.wq(x), batch_size)
        k = self.split_heads(self.wk(x), batch_size)
        v = self.split_heads(self.wv(x), batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(
            q, k, v, mask=mask, training=training
        )

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
        attn_output, _ = self.att(x, mask=mask, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2


# ============================================================
# 5. 完整 GPT 模型
# ============================================================
class GPTModel(tf.keras.Model):
    """
    GPT 风格的 Transformer 语言模型
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

        # ===== 修复：补上缺失的实例属性 =====
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        # ====================================

        self.token_embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=False)
        self.pos_encoding = PositionalEncoding(max_seq_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.transformer_blocks = [
            TransformerBlock(d_model, num_heads, dff, dropout_rate)
            for _ in range(num_layers)
        ]

        self.final_layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.output_projection = tf.keras.layers.Dense(vocab_size)

    def create_look_ahead_mask(self, size: int) -> tf.Tensor:
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

    def call(self, inputs: tf.Tensor, training: bool = False):
        seq_len = tf.shape(inputs)[1]

        x = self.token_embedding(inputs)
        x = x * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)

        look_ahead_mask = self.create_look_ahead_mask(seq_len)

        for block in self.transformer_blocks:
            x = block(x, mask=look_ahead_mask, training=training)

        x = self.final_layernorm(x)
        logits = self.output_projection(x)

        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dff": self.dff,
            "max_seq_len": self.max_seq_len,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ============================================================
# 6. 学习率调度
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

        warmup_lr = step * (self.d_model ** -0.5) / self.warmup_steps

        progress = tf.minimum((step - self.warmup_steps) / (self.max_steps - self.warmup_steps), 1.0)
        cosine_decay = 0.5 * (1 + tf.cos(np.pi * progress))
        decayed = (1 - self.min_lr_ratio) * cosine_decay + self.min_lr_ratio

        base_lr = self.d_model ** -0.5
        decay_lr = base_lr * decayed

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
# 7. 工具函数
# ============================================================
def count_parameters(model: tf.keras.Model) -> int:
    """手动计算参数量（兼容 Keras 2/3）"""
    total = 0
    for weight in model.trainable_weights:
        total += int(tf.reduce_prod(weight.shape))
    return total


def create_model(config: dict) -> GPTModel:
    return GPTModel(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dff=config["dff"],
        max_seq_len=config["max_seq_len"],
        dropout_rate=config.get("dropout_rate", 0.1),
    )


if __name__ == "__main__":
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

    test_input = tf.constant([[1, 2, 3, 4, 5]], dtype=tf.int32)
    output = model(test_input, training=False)
    print(f"输入 shape: {test_input.shape}")
    print(f"输出 shape: {output.shape}")
    print(f"参数量: {count_parameters(model):,}")

    # 测试保存/加载
    print("\n测试保存/加载...")
    model.save("/tmp/test_model.keras")
    loaded = tf.keras.models.load_model("/tmp/test_model.keras", 
        custom_objects={"GPTModel": GPTModel, "WarmupCosineDecay": WarmupCosineDecay})
    print(f"加载成功！参数量: {count_parameters(loaded):,}")

    lr_schedule = WarmupCosineDecay(d_model=128, warmup_steps=100, max_steps=1000)
    print(f"\n学习率调度测试:")
    for step in [0, 50, 100, 500, 1000]:
        print(f"  step={step}: lr={lr_schedule(step).numpy():.6f}")