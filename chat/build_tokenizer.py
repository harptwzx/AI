#!/usr/bin/env python3
"""
Tokenizer 训练脚本
==================
默认使用 WordTokenizer（常见单词映射 + 罕见词字符拆分）
支持数据采样（折半）

用法:
    python build_tokenizer.py                    # 默认 WordTokenizer
    python build_tokenizer.py --type word        # 显式使用 WordTokenizer
    python build_tokenizer.py --type char        # 字符级
    python build_tokenizer.py --type bpe         # BPE（慢！）
    python build_tokenizer.py --sample 0.5      # 只用 50% 数据
    python build_tokenizer.py --min-freq 10      # 单词最小频率阈值
"""

import os
import sys
import argparse
import time
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import WordTokenizer, CharTokenizer, BPETokenizer, load_tokenizer


def load_train_text(filepath: str, sample_ratio: float = 1.0, max_lines: int = None) -> list:
    """加载训练文本，支持采样"""
    print(f"[Data] 加载 {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[Data] 总句子数: {len(lines):,}")

    # 按 sample_ratio 采样
    if sample_ratio < 1.0:
        random.seed(42)
        sample_size = int(len(lines) * sample_ratio)
        lines = random.sample(lines, sample_size)
        print(f"[Data] 采样 {sample_ratio*100:.0f}% -> {len(lines):,} 句")

    if max_lines and len(lines) > max_lines:
        random.seed(42)
        lines = random.sample(lines, max_lines)
        print(f"[Data] 限制到 {max_lines:,} 句")

    return lines


def train_word_tokenizer(texts: list, min_freq: int = 5, save_dir: str = "./data/tokenizer"):
    """训练 WordTokenizer（瞬间完成）"""
    print("\n" + "=" * 60)
    print("WordTokenizer 训练")
    print("=" * 60)

    start = time.time()
    tokenizer = WordTokenizer()
    tokenizer.min_word_freq = min_freq
    tokenizer.build_vocab(texts, save_dir=save_dir)
    elapsed = time.time() - start

    print(f"\n[WordTokenizer] 训练完成！耗时: {elapsed:.2f}s")

    # 统计
    print("\n[WordTokenizer] 词表统计:")
    word_count = sum(1 for k, v in tokenizer.id_to_word.items() if v not in tokenizer.SPECIAL_TOKENS and len(v) > 1)
    char_count = sum(1 for k, v in tokenizer.id_to_word.items() if len(v) == 1 and v not in tokenizer.SPECIAL_TOKENS)
    print(f"  单词 token: {word_count}")
    print(f"  字符 token: {char_count}")
    print(f"  总 vocab: {tokenizer.vocab_size}")

    # 测试
    print("\n[WordTokenizer] 测试编码/解码:")
    test_sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "I love learning English every day.",
        "Unhappiness is not a good feeling.",
        "Running quickly through the forest.",
    ]

    for text in test_sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        tokens = [tokenizer.id_to_word.get(i, "<unk>") for i in encoded]
        print(f"\n  原文: {text}")
        print(f"  编码: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"  Tokens: {tokens[:15]}{'...' if len(tokens) > 15 else ''}")
        print(f"  解码: {decoded}")
        print(f"  压缩: {len(text)} chars -> {len(encoded)} tokens = {len(text)/len(encoded):.2f}x")

    return tokenizer


def train_char_tokenizer(texts: list, save_dir: str = "./data/tokenizer"):
    """训练 CharTokenizer"""
    print("\n" + "=" * 60)
    print("CharTokenizer 训练")
    print("=" * 60)

    start = time.time()
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts, save_dir=save_dir)
    elapsed = time.time() - start

    print(f"\n[CharTokenizer] 训练完成！耗时: {elapsed:.2f}s")
    return tokenizer


def train_bpe_tokenizer(texts: list, vocab_size: int = 32000, max_merges: int = None, save_dir: str = "./data/tokenizer"):
    """训练 BPE Tokenizer（很慢！）"""
    print("\n" + "=" * 60)
    print("BPETokenizer 训练")
    print("=" * 60)
    print("[WARN] BPE 训练非常慢，大数据集可能需要数小时！\n")

    all_text = "\n".join(texts)
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    tokenizer.train(all_text, save_dir=save_dir, max_merges=max_merges)

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练 Tokenizer")
    parser.add_argument("--type", type=str, choices=["word", "char", "bpe"], default="word",
                        help="Tokenizer 类型: word(推荐)/char/bpe(慢)")
    parser.add_argument("--vocab", type=int, default=32000, help="BPE 词表大小")
    parser.add_argument("--max-merges", type=int, default=None, help="BPE 最大合并次数")
    parser.add_argument("--min-freq", type=int, default=5, help="WordTokenizer 单词最小频率")
    parser.add_argument("--sample", type=float, default=1.0, help="数据采样比例 (0.5=折半)")
    parser.add_argument("--max-lines", type=int, default=None, help="最大句子数")
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

    # 加载文本（支持采样）
    texts = load_train_text(args.data, sample_ratio=args.sample, max_lines=args.max_lines)

    # 训练
    if args.type == "word":
        tokenizer = train_word_tokenizer(texts, min_freq=args.min_freq, save_dir=args.save_dir)
    elif args.type == "char":
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
    print(f"    - word_list.txt")
    print(f"    - test_results.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
