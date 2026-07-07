"""
训练脚本
========
功能:
- 从 SaveModel 格式加载已有模型继续训练
- 每 epoch 保存检查点
- 保存验证集上表现最好的模型
- Early Stopping 防止过拟合
- TensorBoard 日志
"""

import os
import json
import time
import tensorflow as tf
import numpy as np
from datetime import datetime

from model import GPTModel, WarmupCosineDecay, count_parameters
from tokenizer import BPETokenizer


# ============================================================
# 配置
# ============================================================
class TrainingConfig:
    """训练配置"""

    # 模型配置
    vocab_size = 32000
    d_model = 512
    num_heads = 8
    num_layers = 6
    dff = 2048
    max_seq_len = 256
    dropout_rate = 0.1

    # 训练配置
    batch_size = 32
    epochs = 100  # 最大 epoch 数（会被 early stopping 打断）
    learning_rate = 3e-4
    warmup_steps = 2000
    max_train_steps = 500000

    # 早停配置
    early_stopping_patience = 5  # 验证 loss 不改善的 patience
    early_stopping_min_delta = 0.001

    # 保存配置
    save_best_only = True
    checkpoint_freq = 1  # 每 N 个 epoch 保存一次

    # 路径配置
    model_dir = "./models"
    checkpoint_dir = "./models/checkpoints"
    best_model_dir = "./models/best_model"
    log_dir = "./logs"
    tokenizer_dir = "./data/tokenizer"
    train_data_path = "./data/processed/train.txt"
    val_data_path = "./data/processed/val.txt"


# ============================================================
# 数据管道
# ============================================================
class TextDataPipeline:
    """文本数据管道"""

    def __init__(self, tokenizer: BPETokenizer, seq_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

    def load_text_file(self, filepath: str) -> list:
        """加载文本文件"""
        print(f"[Data] 加载 {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"[Data] 共 {len(lines):,} 行")
        return lines

    def create_dataset(self, texts: list, shuffle: bool = True, buffer_size: int = 10000):
        """
        创建 tf.data.Dataset

        将文本编码为 token ids，然后创建 (input, target) 对
        target 是 input 向右偏移一位（next token prediction）
        """
        print("[Data] 编码文本...")
        all_ids = []
        for text in texts:
            ids = self.tokenizer.encode(text)
            # 添加 endoftext token
            ids.append(self.tokenizer.special_tokens["<|endoftext|>"])
            all_ids.extend(ids)

        print(f"[Data] 总 token 数: {len(all_ids):,}")

        # 创建滑动窗口样本
        def gen():
            for i in range(0, len(all_ids) - self.seq_len):
                x = all_ids[i : i + self.seq_len]
                y = all_ids[i + 1 : i + self.seq_len + 1]
                yield x, y

        dataset = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(self.seq_len,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.seq_len,), dtype=tf.int32),
            ),
        )

        if shuffle:
            dataset = dataset.shuffle(buffer_size)

        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


# ============================================================
# 自定义回调
# ============================================================
class SaveBestModelCallback(tf.keras.callbacks.Callback):
    """保存验证集上最好的模型（SaveModel 格式）"""

    def __init__(self, save_dir: str, monitor: str = "val_loss", mode: str = "min"):
        super().__init__()
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        improved = (self.mode == "min" and current < self.best_value - 1e-7) or \
                   (self.mode == "max" and current > self.best_value + 1e-7)

        if improved:
            self.best_value = current
            self.best_epoch = epoch

            # 保存为 SaveModel 格式
            save_path = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            self.model.save(save_path, save_format="tf")

            # 同时保存到 best_model 目录（覆盖）
            best_path = os.path.join(self.save_dir, "..", "best_model")
            self.model.save(best_path, save_format="tf")

            print(f"\n[SaveBest] Epoch {epoch+1}: {self.monitor}={current:.6f} "
                  f"(新最佳) -> 已保存到 {save_path}")

    def on_train_end(self, logs=None):
        print(f"\n[SaveBest] 训练结束。最佳模型在第 {self.best_epoch+1} epoch，"
              f"{self.monitor}={self.best_value:.6f}")


class CheckpointCallback(tf.keras.callbacks.Callback):
    """定期保存检查点"""

    def __init__(self, save_dir: str, freq: int = 1):
        super().__init__()
        self.save_dir = save_dir
        self.freq = freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            save_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1:03d}")
            self.model.save(save_path, save_format="tf")
            print(f"[Checkpoint] Epoch {epoch+1}: 检查点已保存到 {save_path}")


class TrainingLogger(tf.keras.callbacks.Callback):
    """自定义训练日志"""

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        elapsed = time.time() - self.epoch_start_time
        print(f"[Epoch {epoch+1}] loss={logs.get('loss', 0):.6f}, "
              f"val_loss={logs.get('val_loss', 0):.6f}, "
              f"acc={logs.get('accuracy', 0):.4f}, "
              f"val_acc={logs.get('val_accuracy', 0):.4f}, "
              f"time={elapsed:.1f}s")


# ============================================================
# 训练器
# ============================================================
class Trainer:
    """模型训练器"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.history = None

    def load_tokenizer(self):
        """加载或训练 tokenizer"""
        tokenizer_dir = self.config.tokenizer_dir

        if os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            print(f"[Tokenizer] 从 {tokenizer_dir} 加载...")
            self.tokenizer = BPETokenizer.load(tokenizer_dir)
        else:
            print("[Tokenizer] 未找到已有 tokenizer，需要训练...")
            # 这里需要训练数据，实际运行时从 train.txt 加载
            print("[Tokenizer] 请先运行 tokenizer.py 训练 tokenizer")
            raise FileNotFoundError("Tokenizer 文件未找到")

        return self.tokenizer

    def build_model(self, from_checkpoint: str = None):
        """
        构建或加载模型

        Args:
            from_checkpoint: 已有 SaveModel 路径，None 则创建新模型
        """
        if from_checkpoint and os.path.exists(from_checkpoint):
            print(f"[Model] 从检查点加载: {from_checkpoint}")
            self.model = tf.keras.models.load_model(
                from_checkpoint,
                custom_objects={
                    "GPTModel": GPTModel,
                    "PositionalEncoding": __import__("model").PositionalEncoding,
                    "MultiHeadAttention": __import__("model").MultiHeadAttention,
                    "FeedForward": __import__("model").FeedForward,
                    "TransformerBlock": __import__("model").TransformerBlock,
                    "WarmupCosineDecay": WarmupCosineDecay,
                }
            )
        else:
            print("[Model] 创建新模型...")
            self.model = GPTModel(
                vocab_size=self.config.vocab_size,
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                num_layers=self.config.num_layers,
                dff=self.config.dff,
                max_seq_len=self.config.max_seq_len,
                dropout_rate=self.config.dropout_rate,
            )

            # 构建模型（通过一次前向传播）
            dummy_input = tf.zeros((1, self.config.max_seq_len), dtype=tf.int32)
            _ = self.model(dummy_input, training=False)

        print(f"[Model] 参数量: {count_parameters(self.model):,}")
        return self.model

    def compile_model(self):
        """编译模型"""
        # 学习率调度
        lr_schedule = WarmupCosineDecay(
            d_model=self.config.d_model,
            warmup_steps=self.config.warmup_steps,
            max_steps=self.config.max_train_steps,
        )

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=0.9,
            beta_2=0.98,
            epsilon=1e-9,
        )

        # 损失函数（忽略 padding token）
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )

        def masked_loss(y_true, y_pred):
            """带 mask 的损失函数"""
            loss = loss_fn(y_true, y_pred)
            # 创建 mask（非 padding 位置）
            mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
            loss *= mask
            return tf.reduce_sum(loss) / tf.reduce_sum(mask)

        self.model.compile(
            optimizer=optimizer,
            loss=masked_loss,
            metrics=["accuracy"],
        )

        print("[Model] 编译完成")
        return self.model

    def train(self, from_checkpoint: str = None):
        """
        开始训练

        Args:
            from_checkpoint: 从已有 SaveModel 继续训练
        """
        # 1. 加载 tokenizer
        self.load_tokenizer()

        # 2. 构建/加载模型
        self.build_model(from_checkpoint)
        self.compile_model()

        # 3. 准备数据
        pipeline = TextDataPipeline(
            self.tokenizer,
            self.config.max_seq_len,
            self.config.batch_size
        )

        train_texts = pipeline.load_text_file(self.config.train_data_path)
        val_texts = pipeline.load_text_file(self.config.val_data_path)

        train_dataset = pipeline.create_dataset(train_texts, shuffle=True)
        val_dataset = pipeline.create_dataset(val_texts, shuffle=False)

        # 4. 回调函数
        callbacks = [
            # TensorBoard
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1,
                update_freq="epoch",
            ),

            # 保存最佳模型
            SaveBestModelCallback(
                save_dir=self.config.best_model_dir,
                monitor="val_loss",
                mode="min",
            ),

            # 定期保存检查点
            CheckpointCallback(
                save_dir=self.config.checkpoint_dir,
                freq=self.config.checkpoint_freq,
            ),

            # 早停
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1,
            ),

            # 学习率记录
            tf.keras.callbacks.LearningRateScheduler(
                lambda epoch, lr: lr,
                verbose=0,
            ),

            # 自定义日志
            TrainingLogger(),
        ]

        # 5. 训练
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)

        self.history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.epochs,
            callbacks=callbacks,
            verbose=1,
        )

        # 6. 保存最终模型
        final_path = os.path.join(self.config.model_dir, "final_model")
        self.model.save(final_path, save_format="tf")
        print(f"\n[Done] 最终模型已保存到 {final_path}")

        return self.history


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练 GPT 模型")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从已有 SaveModel 路径继续训练"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径 (JSON)"
    )

    args = parser.parse_args()

    # 加载配置
    config = TrainingConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            for k, v in custom_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    # 创建训练器
    trainer = Trainer(config)

    # 开始训练
    trainer.train(from_checkpoint=args.resume)
