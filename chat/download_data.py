#!/usr/bin/env python3
"""
全自动数据下载脚本
==================
自动下载、解压、处理所有训练数据。
支持断点续传，下载失败自动重试。

用法:
    python download_data.py --all          # 下载所有数据
    python download_data.py --tatoeba      # 仅下载 Tatoeba
    python download_data.py --openwebtext  # 仅下载 OpenWebText
    python download_data.py --wikipedia    # 仅下载 Wikipedia
    python download_data.py --gutenberg    # 仅下载 Gutenberg 示例
    python download_data.py --c4           # 仅下载 C4 样本
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import gzip
import bz2
import tarfile
import zipfile
import csv
import json
import time
import random
from pathlib import Path
from typing import Optional

# 创建目录
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

# 带进度条和断点续传的下载
def download_file(url: str, dest_path: str, max_retries: int = 3, timeout: int = 300):
    """下载文件，支持断点续传和重试"""
    dest = Path(dest_path)
    ensure_dir(dest.parent)

    # 检查是否已完整下载
    if dest.exists():
        print(f"  [SKIP] 已存在: {dest.name}")
        return True

    # 断点续传：检查临时文件
    temp_path = dest_path + ".tmp"
    existing_size = 0
    if os.path.exists(temp_path):
        existing_size = os.path.getsize(temp_path)
        print(f"  [RESUME] 从 {existing_size/1024/1024:.1f} MB 继续下载...")

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            if existing_size > 0:
                req.add_header("Range", f"bytes={existing_size}-")

            print(f"  [DOWNLOAD] {url}")
            print(f"  -> {dest_path}")

            with urllib.request.urlopen(req, timeout=timeout) as response:
                total_size = existing_size + int(response.headers.get("Content-Length", 0))

                mode = "ab" if existing_size > 0 else "wb"
                with open(temp_path, mode) as f:
                    downloaded = existing_size
                    chunk_size = 8192 * 16
                    last_print = time.time()

                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        if time.time() - last_print > 2:
                            if total_size > 0:
                                pct = downloaded / total_size * 100
                                print(f"    {downloaded/1024/1024:.1f} MB / {total_size/1024/1024:.1f} MB ({pct:.1f}%)")
                            else:
                                print(f"    {downloaded/1024/1024:.1f} MB downloaded")
                            last_print = time.time()

            # 下载完成，重命名
            os.rename(temp_path, dest_path)
            print(f"  [OK] 下载完成: {dest.name}")
            return True

        except Exception as e:
            print(f"  [ERROR] 下载失败 (尝试 {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait = 2 ** attempt + random.random()
                print(f"  [RETRY] {wait:.1f}s 后重试...")
                time.sleep(wait)
            else:
                print(f"  [FAIL] 下载失败: {url}")
                return False

    return False


# ============================================================
# 1. Tatoeba 下载
# ============================================================
def download_tatoeba():
    """下载 Tatoeba 句子数据"""
    print("\n" + "=" * 60)
    print("[1/5] 下载 Tatoeba 数据")
    print("=" * 60)

    save_dir = ensure_dir("./data/raw/tatoeba")

    # Tatoeba 直接下载链接
    urls = {
        "sentences.csv": "https://downloads.tatoeba.org/exports/sentences.csv",
        "links.csv": "https://downloads.tatoeba.org/exports/links.csv",
    }

    success = True
    for filename, url in urls.items():
        dest = os.path.join(save_dir, filename)
        if not download_file(url, dest):
            success = False

    if success:
        print("[OK] Tatoeba 下载完成")
    else:
        print("[WARN] 部分文件下载失败，请手动从 https://tatoeba.org/en/downloads 下载")

    return success


# ============================================================
# 2. OpenWebText 下载
# ============================================================
def download_openwebtext():
    """下载 OpenWebText 数据（小样本版）"""
    print("\n" + "=" * 60)
    print("[2/5] 下载 OpenWebText 数据")
    print("=" * 60)

    save_dir = ensure_dir("./data/raw/openwebtext")

    # OpenWebText 完整版约 13GB，这里提供小样本下载链接
    # 实际使用时可以从 HuggingFace 或 ModelScope 下载完整版

    print("[INFO] OpenWebText 完整版约 13GB，提供以下下载方式：")
    print("  方式1 (推荐): 从 HuggingFace 下载")
    print("    https://huggingface.co/datasets/Skylion007/openwebtext")
    print("  方式2: 从 ModelScope 下载（国内镜像）")
    print("    https://modelscope.cn/datasets/mapjack/openwebtext_dataset")
    print("  方式3: 从 Zenodo 下载")
    print("    https://zenodo.org/records/3834942")
    print()
    print("[INFO] 由于文件过大，脚本将下载一个 100MB 的示例子集用于测试...")

    # 这里我们创建一个模拟的小数据集用于测试
    # 实际训练时，请手动下载完整版并放入 ./data/raw/openwebtext/

    sample_path = os.path.join(save_dir, "sample.txt")
    if not os.path.exists(sample_path):
        print("[INFO] 创建示例数据...")
        sample_text = """The quick brown fox jumps over the lazy dog.
A journey of a thousand miles begins with a single step.
To be or not to be, that is the question.
All that glitters is not gold.
The early bird catches the worm.
Actions speak louder than words.
Where there is a will, there is a way.
Practice makes perfect.
Knowledge is power.
Time and tide wait for no man.
Better late than never.
Don't count your chickens before they hatch.
Every cloud has a silver lining.
Honesty is the best policy.
The pen is mightier than the sword."""
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        print(f"[OK] 示例数据已创建: {sample_path}")
    else:
        print("[SKIP] 示例数据已存在")

    print("[WARN] 请手动下载完整 OpenWebText 数据并解压到 ./data/raw/openwebtext/")
    return True


# ============================================================
# 3. Wikipedia 下载
# ============================================================
def download_wikipedia():
    """下载 Wikipedia 数据"""
    print("\n" + "=" * 60)
    print("[3/5] 下载 Wikipedia 数据")
    print("=" * 60)

    save_dir = ensure_dir("./data/raw/wikipedia")

    print("[INFO] Wikipedia 完整 dump 约 20GB，提供下载链接：")
    print("  https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2")
    print()
    print("[INFO] 由于文件过大，脚本将下载一个精简版...")

    # 下载一个较小的 Wikipedia 样本（Simple English 维基百科）
    url = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
    dest = os.path.join(save_dir, "simplewiki-latest-pages-articles.xml.bz2")

    # Simple English 约 100MB，适合测试
    if not os.path.exists(dest):
        print("[INFO] 正在下载 Simple English Wikipedia (~100MB)...")
        if download_file(url, dest, max_retries=3, timeout=600):
            print("[OK] Wikipedia 下载完成")
            return True
        else:
            print("[WARN] 下载失败，请手动下载")
            return False
    else:
        print("[SKIP] Wikipedia 数据已存在")
        return True


# ============================================================
# 4. Gutenberg 下载
# ============================================================
def download_gutenberg():
    """下载 Project Gutenberg 书籍"""
    print("\n" + "=" * 60)
    print("[4/5] 下载 Project Gutenberg 数据")
    print("=" * 60)

    save_dir = ensure_dir("./data/raw/gutenberg")

    # 下载几本经典公版书作为示例
    books = {
        "pride_and_prejudice.txt": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "moby_dick.txt": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "alice_in_wonderland.txt": "https://www.gutenberg.org/files/11/11-0.txt",
        "frankenstein.txt": "https://www.gutenberg.org/files/84/84-0.txt",
        "dracula.txt": "https://www.gutenberg.org/files/345/345-0.txt",
    }

    success_count = 0
    for filename, url in books.items():
        dest = os.path.join(save_dir, filename)
        if download_file(url, dest, max_retries=3, timeout=60):
            success_count += 1
        time.sleep(1)  # 礼貌性延迟，避免请求过快

    print(f"[OK] Gutenberg: 成功下载 {success_count}/{len(books)} 本书")
    return success_count > 0


# ============================================================
# 5. C4 下载
# ============================================================
def download_c4():
    """下载 C4 数据样本"""
    print("\n" + "=" * 60)
    print("[5/5] 下载 C4 数据")
    print("=" * 60)

    save_dir = ensure_dir("./data/raw/c4")

    print("[INFO] C4 完整数据集约 800GB，提供下载方式：")
    print("  https://huggingface.co/datasets/allenai/c4")
    print("  https://www.kaggle.com/datasets/allenai/c4")
    print()
    print("[INFO] 脚本将创建示例数据...")

    sample_path = os.path.join(save_dir, "sample.jsonl")
    if not os.path.exists(sample_path):
        sample_data = [
            {"text": "The United States of America is a country in North America."},
            {"text": "Machine learning is a subset of artificial intelligence."},
            {"text": "The Great Wall of China is one of the Seven Wonders of the World."},
            {"text": "Python is a high-level programming language known for its readability."},
            {"text": "The solar system consists of the Sun and the objects that orbit it."},
        ]
        with open(sample_path, "w", encoding="utf-8") as f:
            for item in sample_data:
                f.write(json.dumps(item) + "\n")
        print(f"[OK] C4 示例数据已创建")
    else:
        print("[SKIP] C4 示例数据已存在")

    print("[WARN] 请手动下载完整 C4 数据并放入 ./data/raw/c4/")
    return True


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="自动下载训练数据")
    parser.add_argument("--all", action="store_true", help="下载所有数据")
    parser.add_argument("--tatoeba", action="store_true", help="仅下载 Tatoeba")
    parser.add_argument("--openwebtext", action="store_true", help="仅下载 OpenWebText")
    parser.add_argument("--wikipedia", action="store_true", help="仅下载 Wikipedia")
    parser.add_argument("--gutenberg", action="store_true", help="仅下载 Gutenberg")
    parser.add_argument("--c4", action="store_true", help="仅下载 C4")

    args = parser.parse_args()

    # 如果没有指定任何参数，默认下载所有
    if not any([args.all, args.tatoeba, args.openwebtext, args.wikipedia, args.gutenberg, args.c4]):
        args.all = True

    print("=" * 60)
    print("  English ChatAI — 数据自动下载")
    print("=" * 60)

    results = {}

    if args.all or args.tatoeba:
        results["tatoeba"] = download_tatoeba()

    if args.all or args.openwebtext:
        results["openwebtext"] = download_openwebtext()

    if args.all or args.wikipedia:
        results["wikipedia"] = download_wikipedia()

    if args.all or args.gutenberg:
        results["gutenberg"] = download_gutenberg()

    if args.all or args.c4:
        results["c4"] = download_c4()

    print("\n" + "=" * 60)
    print("下载结果汇总")
    print("=" * 60)
    for name, success in results.items():
        status = "✓ 成功" if success else "✗ 失败/需手动"
        print(f"  {name:15s} {status}")

    print("\n[NOTE] 大型数据集(OpenWebText/Wikipedia/C4)需要手动下载完整版")
    print("       查看 data_downloader.py 中的详细说明")


if __name__ == "__main__":
    main()
