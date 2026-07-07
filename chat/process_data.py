#!/usr/bin/env python3
"""
数据处理脚本 — ManyThings 版
==============================
处理从 ManyThings.org 下载的英语句子数据，生成 train.txt 和 val.txt

用法:
    python process_data.py              # 处理所有下载的数据
    python process_data.py --max-lines 50000  # 限制处理行数
"""

import os
import re
import random
import argparse
from pathlib import Path
from typing import List, Set

random.seed(42)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def clean_text(text: str) -> str:
    """清洗文本"""
    if not text or not isinstance(text, str):
        return ""

    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)

    # 移除控制字符
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)

    # 移除 URL
    text = re.sub(r'https?://\S+', '', text)

    # 移除多余标点
    text = re.sub(r'([!?.]){3,}', r'\1\1\1', text)

    return text.strip()


def is_valid_sentence(text: str, min_len: int = 10, max_len: int = 200) -> bool:
    """判断是否为有效英语句子"""
    if not text:
        return False
    if len(text) < min_len or len(text) > max_len:
        return False
    # 至少包含一个字母
    if not re.search(r'[a-zA-Z]', text):
        return False
    # 非英语字符比例不能太高
    non_ascii = len(re.findall(r'[^\x00-\x7F]', text))
    if non_ascii / len(text) > 0.1:
        return False
    # 至少包含 3 个单词
    words = text.split()
    if len(words) < 3:
        return False
    return True


def extract_english_from_manythings(txt_path: str) -> List[str]:
    """
    从 ManyThings 格式的 txt 文件中提取英语句子

    ManyThings 格式: English\tOtherLanguage\tAttribution
    例如: "Hello.\tHallo.\tCC-BY 2.0 ..."
    """
    sentences = []

    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # ManyThings 格式是 tab 分隔
            parts = line.split("\t")
            if len(parts) >= 2:
                # 第一列是英语句子
                english = parts[0].strip()
                english = clean_text(english)

                if is_valid_sentence(english):
                    sentences.append(english)

    return sentences


def process_all_manythings(data_dir: str = "./data/raw/manythings", max_lines: int = None) -> List[str]:
    """处理所有 ManyThings 数据"""
    print("=" * 60)
    print("  处理 ManyThings 英语句子数据")
    print("=" * 60)

    all_sentences = []

    # 查找所有 txt 文件
    txt_files = list(Path(data_dir).rglob("*.txt"))

    if not txt_files:
        print(f"[ERROR] 未在 {data_dir} 找到数据文件")
        print("[HINT] 请先运行: python download_data.py")
        return []

    print(f"找到 {len(txt_files)} 个数据文件\n")

    for txt_path in sorted(txt_files):
        print(f"处理: {txt_path.name} ...", end=" ")
        sentences = extract_english_from_manythings(str(txt_path))
        all_sentences.extend(sentences)
        print(f"提取 {len(sentences):,} 句")

        if max_lines and len(all_sentences) >= max_lines:
            print(f"\n达到限制 {max_lines} 句，停止处理")
            all_sentences = all_sentences[:max_lines]
            break

    return all_sentences


def save_data(sentences: List[str], train_ratio: float = 0.95):
    """保存训练集和验证集"""
    processed_dir = ensure_dir("./data/processed")

    # 去重
    print(f"\n去重前: {len(sentences):,} 句")
    seen: Set[str] = set()
    unique = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    print(f"去重后: {len(unique):,} 句")

    # 打乱
    random.shuffle(unique)

    # 划分
    split_idx = int(len(unique) * train_ratio)
    train = unique[:split_idx]
    val = unique[split_idx:]

    # 保存
    train_path = os.path.join(processed_dir, "train.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train:
            f.write(s + "\n")

    val_path = os.path.join(processed_dir, "val.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val:
            f.write(s + "\n")

    # 统计
    total_chars = sum(len(s) for s in unique)
    print(f"\n{'='*60}")
    print("  数据统计")
    print(f"{'='*60}")
    print(f"  总句子数:    {len(unique):,}")
    print(f"  总字符数:    {total_chars:,}")
    print(f"  平均句长:    {total_chars / len(unique):.1f} 字符")
    print(f"  训练集:      {len(train):,} ({len(train)/len(unique)*100:.1f}%)")
    print(f"  验证集:      {len(val):,} ({len(val)/len(unique)*100:.1f}%)")
    print(f"  文件大小:    ~{total_chars/1024/1024:.1f}MB")
    print(f"{'='*60}")
    print(f"\n保存位置:")
    print(f"  {train_path}")
    print(f"  {val_path}")


def main():
    parser = argparse.ArgumentParser(description="处理 ManyThings 数据")
    parser.add_argument("--max-lines", type=int, default=None, help="最大处理行数")
    parser.add_argument("--data-dir", type=str, default="./data/raw/manythings", help="数据目录")

    args = parser.parse_args()

    sentences = process_all_manythings(args.data_dir, args.max_lines)

    if not sentences:
        print("\n[ERROR] 没有收集到任何数据！")
        return

    save_data(sentences)

    print("\n[OK] 数据处理完成！下一步: python build_tokenizer.py")


if __name__ == "__main__":
    main()
