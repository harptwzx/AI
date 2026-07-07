#!/usr/bin/env python3
"""
Tokenizer 训练脚本
==================
默认使用 CharTokenizer（瞬间完成），可选 BPE（训练慢）。

用法:
    python build_tokenizer.py                    # 默认 CharTokenizer
    python build_tokenizer.py --type char        # 显式使用字符级
    python build_tokenizer.py --type bpe         # 使用 BPE（慢！）
    python build_tokenizer.py --type bpe --max-merges 1000  # BPE 只合并1000次
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import CharTokenizer, BPETokenizer, load_tokenizer


def load_train_text(filepath: str, max_lines: int = None) -> list:
    """加载训练文本"""
    print(f"[Data] 加载 {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[Data] 总句子数: {len(lines):,}")

    if max_lines and len(lines) > max_lines:
        import random
        random.seed(42)
        sampled = random.sample(lines, max_lines)
        print(f"[Data] 采样 {max_lines:,} 句用于训练 tokenizer")
        return sampled

    return lines


def train_char_tokenizer(texts: list, save_dir: str = "./data/tokenizer"):
    """训练字符级 Tokenizer（瞬间完成）"""
    print("\n" + "=" * 60)
    print("CharTokenizer 训练")
    print("=" * 60)

    start = time.time()
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts, save_dir=save_dir)
    elapsed = time.time() - start

    print(f"\n[CharTokenizer] 训练完成！耗时: {elapsed:.2f}s")

    # 测试
    print("\n[CharTokenizer] 测试:")
    test_sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I love learning English every day.",
    ]

    for text in test_sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"  原文: {text}")
        print(f"  编码: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"  解码: {decoded}")
        print()

    return tokenizer


def train_bpe_tokenizer(texts: list, vocab_size: int = 32000, max_merges: int = None, save_dir: str = "./data/tokenizer"):
    """训练 BPE Tokenizer（很慢！）"""
    print("\n" + "=" * 60)
    print("BPETokenizer 训练")
    print("=" * 60)
    print("[WARN] BPE 训练非常慢，大数据集可能需要数小时！")
    print("[WARN] 建议先用 CharTokenizer 快速开始。\n")

    all_text = "\n".join(texts)

    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(all_text, save_dir=save_dir, max_merges=max_merges)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练 Tokenizer")
    parser.add_argument("--type", type=str, choices=["char", "bpe"], default="char",
                        help="Tokenizer 类型: char(快) 或 bpe(慢)")
    parser.add_argument("--vocab", type=int, default=32000, help="BPE 词表大小")
    parser.add_argument("--max-merges", type=int, default=None, help="BPE 最大合并次数（减少可加速）")
    parser.add_argument("--sample", type=int, default=None, help="采样句子数")
    parser.add_argument("--save-dir", type=str, default="./data/tokenizer", help="保存目录")
    parser.add_argument("--data", type=str, default="./data/processed/train.txt", help="训练数据路径")

    args = parser.parse_args()

    print("=" * 60)
    print("  English ChatAI — Tokenizer 训练")
    print("=" * 60)

    if not os.path.exists(args.data):
        print(f"[ERROR] 未找到 {args.data}")
        print("[HINT] 请先运行: python process_data.py")
        return

    texts = load_train_text(args.data, max_lines=args.sample)

    if args.type == "char":
        tokenizer = train_char_tokenizer(texts, save_dir=args.save_dir)
    else:
        tokenizer = train_bpe_tokenizer(texts, vocab_size=args.vocab, 
                                        max_merges=args.max_merges, save_dir=args.save_dir)

    print("\n" + "=" * 60)
    print("[OK] Tokenizer 训练完成！")
    print(f"  类型: {args.type}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    print(f"  保存位置: {args.save_dir}/")
    print(f"    - tokenizer_config.json")
    print(f"    - vocab_readable.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
