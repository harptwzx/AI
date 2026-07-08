#!/usr/bin/env python3
"""
训练脚本
========
支持 WordCharTokenizer
"""

import os
import json
import time
import tensorflow as tf
import numpy as np
from datetime import datetime

from model import GPTModel, WarmupCosineDecay, count_parameters
from tokenizer import load_tokenizer


# ============================================================
# 配置
# ============================================================
class TrainingConfig:
    vocab_size = 5000
    d_model = 256
    num_heads = 4
    num_layers = 4
    dff = 1024
    max_seq_len = 128
    dropout_rate = 0.1

    batch_size = 16
    epochs = 100
    learning_rate = 3e-4
    warmup_steps = 1000
    max_train_steps = 100000

    early_stopping_patience = 5
    early_stopping_min_delta = 0.001

    save_best_only = True
    checkpoint_freq = 1

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
    def __init__(self, tokenizer, seq_len: int, batch_size: int):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size

    def load_text_file(self, filepath: str) -> list:
        print(f"[Data] 加载 {filepath}...")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"[Data] 共 {len(lines):,} 行")
        return lines

    def create_dataset(self, texts: list, shuffle: bool = True, buffer_size: int = 10000):
        print("[Data] 编码文本...")
        all_ids = []

        # 获取 endoftext token id
        eot_id = self.tokenizer.SPECIAL_TOKENS.get("<endoftext>", 2)

        for text in texts:
            ids = self.tokenizer.encode(text)
            ids.append(eot_id)
            all_ids.extend(ids)

        print(f"[Data] 总 token 数: {len(all_ids):,}")

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
# 回调函数
# ============================================================
class SaveBestModelCallback(tf.keras.callbacks.Callback):
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

            save_path = os.path.join(self.save_dir, f"epoch_{epoch+1:03d}")
            self.model.save(save_path, save_format="tf")

            best_path = os.path.join(self.save_dir, "..", "best_model")
            self.model.save(best_path, save_format="tf")

            print(f"\n[SaveBest] Epoch {epoch+1}: {self.monitor}={current:.6f} (新最佳)")

    def on_train_end(self, logs=None):
        print(f"\n[SaveBest] 最佳模型在第 {self.best_epoch+1} epoch, {self.monitor}={self.best_value:.6f}")


class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_dir: str, freq: int = 1):
        super().__init__()
        self.save_dir = save_dir
        self.freq = freq
        os.makedirs(save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.freq == 0:
            save_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch+1:03d}")
            self.model.save(save_path, save_format="tf")
            print(f"[Checkpoint] Epoch {epoch+1}: 已保存")


class TrainingLogger(tf.keras.callbacks.Callback):
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
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.history = None

    def load_tokenizer(self):
        print(f"[Tokenizer] 从 {self.config.tokenizer_dir} 加载...")
        self.tokenizer = load_tokenizer(self.config.tokenizer_dir)
        self.config.vocab_size = self.tokenizer.vocab_size
        print(f"[Tokenizer] vocab_size={self.config.vocab_size}")
        return self.tokenizer

    def build_model(self, from_checkpoint: str = None):
        if from_checkpoint and os.path.exists(from_checkpoint):
            print(f"[Model] 从检查点加载: {from_checkpoint}")
            self.model = tf.keras.models.load_model(
                from_checkpoint,
                custom_objects={
                    "GPTModel": GPTModel,
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

            dummy_input = tf.zeros((1, self.config.max_seq_len), dtype=tf.int32)
            _ = self.model(dummy_input, training=False)

        print(f"[Model] 参数量: {count_parameters(self.model):,}")
        return self.model

    def compile_model(self):
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

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction="none"
        )

        def masked_loss(y_true, y_pred):
            loss = loss_fn(y_true, y_pred)
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
        self.load_tokenizer()
        self.build_model(from_checkpoint)
        self.compile_model()

        pipeline = TextDataPipeline(
            self.tokenizer,
            self.config.max_seq_len,
            self.config.batch_size
        )

        train_texts = pipeline.load_text_file(self.config.train_data_path)
        val_texts = pipeline.load_text_file(self.config.val_data_path)

        train_dataset = pipeline.create_dataset(train_texts, shuffle=True)
        val_dataset = pipeline.create_dataset(val_texts, shuffle=False)

        callbacks = [
            tf.keras.callbacks.TensorBoard(
                log_dir=os.path.join(self.config.log_dir, datetime.now().strftime("%Y%m%d-%H%M%S")),
                histogram_freq=1,
                update_freq="epoch",
            ),
            SaveBestModelCallback(
                save_dir=self.config.best_model_dir,
                monitor="val_loss",
                mode="min",
            ),
            CheckpointCallback(
                save_dir=self.config.checkpoint_dir,
                freq=self.config.checkpoint_freq,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                restore_best_weights=True,
                verbose=1,
            ),
            TrainingLogger(),
        ]

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

        final_path = os.path.join(self.config.model_dir, "final_model")
        self.model.save(final_path, save_format="tf")
        print(f"\n[Done] 最终模型已保存到 {final_path}")

        return self.history


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="训练 GPT 模型")
    parser.add_argument("--resume", type=str, default=None, help="从已有 SaveModel 继续训练")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径 (JSON)")

    args = parser.parse_args()

    config = TrainingConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            custom_config = json.load(f)
            for k, v in custom_config.items():
                if hasattr(config, k):
                    setattr(config, k, v)

    trainer = Trainer(config)
    trainer.train(from_checkpoint=args.resume)
