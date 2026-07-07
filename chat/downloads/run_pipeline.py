#!/usr/bin/env python3
"""
完整训练流程
============
一键运行: 数据处理 -> Tokenizer 训练 -> 模型训练

用法:
    python run_pipeline.py --stage all
    python run_pipeline.py --stage tokenizer
    python run_pipeline.py --stage train --resume ./models/checkpoints/checkpoint_epoch_010
"""

import os
import sys
import argparse
import json

# 确保可以导入同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor
from tokenizer import BPETokenizer
from train import TrainingConfig, Trainer


def run_data_processing():
    """步骤 1: 数据预处理"""
    print("\n" + "=" * 60)
    print("步骤 1: 数据预处理")
    print("=" * 60)

    processor = DataProcessor()
    processor.process_all()

    print("\n[OK] 数据预处理完成")


def run_tokenizer_training(vocab_size: int = 32000):
    """步骤 2: 训练 Tokenizer"""
    print("\n" + "=" * 60)
    print("步骤 2: 训练 Tokenizer")
    print("=" * 60)

    # 加载训练文本
    train_path = "./data/processed/train.txt"
    if not os.path.exists(train_path):
        print(f"[Error] 未找到 {train_path}，请先运行数据预处理")
        return

    print(f"[Tokenizer] 加载训练文本...")
    with open(train_path, "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f if line.strip()]

    print(f"[Tokenizer] 文本数: {len(texts):,}")

    # 训练 BPE（如果数据量大，可以只采样一部分）
    sample_size = min(500000, len(texts))
    if len(texts) > sample_size:
        import random
        random.seed(42)
        sample_texts = random.sample(texts, sample_size)
        print(f"[Tokenizer] 使用随机采样 {sample_size:,} 条训练 tokenizer")
    else:
        sample_texts = texts

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(sample_texts, save_dir="./data/tokenizer")

    # 测试
    test_text = "Hello, world! This is a test sentence."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n[Tokenizer] 测试:")
    print(f"  原文: {test_text}")
    print(f"  编码: {encoded[:20]}...")
    print(f"  解码: {decoded}")

    print("\n[OK] Tokenizer 训练完成")


def run_training(resume_from: str = None):
    """步骤 3: 训练模型"""
    print("\n" + "=" * 60)
    print("步骤 3: 训练模型")
    print("=" * 60)

    # 加载配置
    config = TrainingConfig()
    if os.path.exists("config.json"):
        with open("config.json", "r") as f:
            custom = json.load(f)
            # 合并配置
            for section in ["model", "training"]:
                if section in custom:
                    for k, v in custom[section].items():
                        if hasattr(config, k):
                            setattr(config, k, v)

    trainer = Trainer(config)
    trainer.train(from_checkpoint=resume_from)

    print("\n[OK] 训练完成")


def main():
    parser = argparse.ArgumentParser(description="完整训练流程")
    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "data", "tokenizer", "train"],
        default="all",
        help="运行阶段"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="从检查点恢复训练（仅 train 阶段有效）"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Tokenizer 词表大小"
    )

    args = parser.parse_args()

    if args.stage in ("all", "data"):
        run_data_processing()

    if args.stage in ("all", "tokenizer"):
        run_tokenizer_training(args.vocab_size)

    if args.stage in ("all", "train"):
        run_training(resume_from=args.resume)

    print("\n" + "=" * 60)
    print("全部完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
