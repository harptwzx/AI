#!/usr/bin/env python3
"""
全自动 Tokenizer 训练脚本
==========================
从 train.txt 训练 BPE Tokenizer，生成 vocab.json + merges.txt

用法:
    python build_tokenizer.py              # 使用默认配置
    python build_tokenizer.py --vocab 16000 --sample 200000
"""

import os
import sys
import argparse
import json
import time
import random
from typing import List, Dict, Tuple

# 将项目根目录加入路径，导入 tokenizer 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import BPETokenizer


def load_train_text(filepath: str, max_lines: int = None) -> List[str]:
    """加载训练文本"""
    print(f"[Data] 加载 {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"[Data] 总句子数: {len(lines):,}")

    if max_lines and len(lines) > max_lines:
        random.seed(42)
        sampled = random.sample(lines, max_lines)
        print(f"[Data] 采样 {max_lines:,} 句用于训练 tokenizer")
        return sampled

    return lines


def train_tokenizer(
    texts: List[str],
    vocab_size: int = 32000,
    save_dir: str = "./data/tokenizer"
):
    """训练 BPE Tokenizer"""
    print("\n" + "=" * 60)
    print("Tokenizer 训练")
    print("=" * 60)

    # 合并所有文本
    print("[Tokenizer] 合并文本...")
    all_text = "\n".join(texts)
    print(f"[Tokenizer] 总字符数: {len(all_text):,}")

    # 创建并训练 tokenizer
    tokenizer = BPETokenizer(vocab_size=vocab_size)

    start_time = time.time()
    tokenizer.train(all_text, save_dir=save_dir)
    elapsed = time.time() - start_time

    print(f"\n[Tokenizer] 训练完成！耗时: {elapsed:.1f}s")

    # 测试
    print("\n[Tokenizer] 测试编码/解码:")
    test_sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "To be or not to be, that is the question.",
    ]

    for text in test_sentences:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"  原文: {text}")
        print(f"  编码: {encoded[:20]}{'...' if len(encoded) > 20 else ''}")
        print(f"  解码: {decoded}")
        print()

    # 保存 token 对照表（纯文本版，方便查看）
    vocab_txt_path = os.path.join(save_dir, "vocab_readable.txt")
    with open(vocab_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# Tokenizer Vocab\n")
        f.write(f"# vocab_size: {vocab_size}\n")
        f.write(f"# actual_size: {len(tokenizer.vocab)}\n")
        f.write(f"# merges: {len(tokenizer.merges)}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'ID':<8} {'Token':<30} {'Type'}\n")
        f.write("-" * 50 + "\n")

        for token_id in sorted(tokenizer.vocab.keys()):
            token_str = tokenizer.vocab[token_id]
            token_type = "BYTE" if token_id < 256 else "MERGE" if token_id < vocab_size else "SPECIAL"
            # 处理不可见字符
            display_str = repr(token_str) if any(ord(c) < 32 for c in token_str) else token_str
            f.write(f"{token_id:<8} {display_str:<30} {token_type}\n")

    print(f"[Tokenizer] 可读 vocab 表已保存: {vocab_txt_path}")

    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="训练 Tokenizer")
    parser.add_argument("--vocab", type=int, default=32000, help="词表大小")
    parser.add_argument("--sample", type=int, default=None, help="采样句子数（默认全部）")
    parser.add_argument("--save-dir", type=str, default="./data/tokenizer", help="保存目录")
    parser.add_argument("--data", type=str, default="./data/processed/train.txt", help="训练数据路径")

    args = parser.parse_args()

    print("=" * 60)
    print("  English ChatAI — Tokenizer 训练")
    print("=" * 60)

    # 检查数据是否存在
    if not os.path.exists(args.data):
        print(f"[ERROR] 未找到 {args.data}")
        print("[HINT] 请先运行: python process_data.py")
        return

    # 加载文本
    texts = load_train_text(args.data, max_lines=args.sample)

    # 训练
    tokenizer = train_tokenizer(
        texts=texts,
        vocab_size=args.vocab,
        save_dir=args.save_dir
    )

    print("\n" + "=" * 60)
    print("[OK] Tokenizer 训练完成！")
    print(f"  vocab.json: {args.save_dir}/vocab.json")
    print(f"  merges.txt: {args.save_dir}/merges.txt")
    print(f"  tokenizer_config.json: {args.save_dir}/tokenizer_config.json")
    print(f"  vocab_readable.txt: {args.save_dir}/vocab_readable.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
