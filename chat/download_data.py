#!/usr/bin/env python3
"""
小型英语数据集自动下载脚本
============================
自动下载约 2-5MB 的高质量英语句子数据，适合训练语法语感。

用法:
    python download_data.py        # 下载所有数据
    python download_data.py --list   # 查看可用数据集列表
"""

import os
import sys
import argparse
import urllib.request
import urllib.error
import zipfile
import io
import time
import random
from pathlib import Path

# ManyThings.org 英语句子数据集（多语言对，我们提取英语部分）
# 这些是 Tatoeba 项目的精选子集，语法规范、句子简短
DATASETS = {
    "eng_deu": {
        "name": "English-German",
        "url": "https://www.manythings.org/anki/deu-eng.zip",
        "filename": "deu.txt",
        "size_mb": 7.4,
        "english_lines": 200000,
    },
    "eng_fra": {
        "name": "English-French", 
        "url": "https://www.manythings.org/anki/fra-eng.zip",
        "filename": "fra.txt",
        "size_mb": 5.2,
        "english_lines": 190000,
    },
    "eng_spa": {
        "name": "English-Spanish",
        "url": "https://www.manythings.org/anki/spa-eng.zip",
        "filename": "spa.txt",
        "size_mb": 3.8,
        "english_lines": 140000,
    },
    "eng_jpn": {
        "name": "English-Japanese",
        "url": "https://www.manythings.org/anki/jpn-eng.zip",
        "filename": "jpn.txt",
        "size_mb": 2.2,
        "english_lines": 50000,
    },
    "eng_rus": {
        "name": "English-Russian",
        "url": "https://www.manythings.org/anki/rus-eng.zip",
        "filename": "rus.txt",
        "size_mb": 2.8,
        "english_lines": 500000,
    },
    "eng_por": {
        "name": "English-Portuguese",
        "url": "https://www.manythings.org/anki/por-eng.zip",
        "filename": "por.txt",
        "size_mb": 2.0,
        "english_lines": 40000,
    },
    "eng_ita": {
        "name": "English-Italian",
        "url": "https://www.manythings.org/anki/ita-eng.zip",
        "filename": "ita.txt",
        "size_mb": 1.8,
        "english_lines": 35000,
    },
    "eng_kor": {
        "name": "English-Korean",
        "url": "https://www.manythings.org/anki/kor-eng.zip",
        "filename": "kor.txt",
        "size_mb": 1.5,
        "english_lines": 25000,
    },
    "eng_nld": {
        "name": "English-Dutch",
        "url": "https://www.manythings.org/anki/nld-eng.zip",
        "filename": "nld.txt",
        "size_mb": 1.2,
        "english_lines": 20000,
    },
    "eng_cmn": {
        "name": "English-Chinese",
        "url": "https://www.manythings.org/anki/cmn-eng.zip",
        "filename": "cmn.txt",
        "size_mb": 1.0,
        "english_lines": 15000,
    },
}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path


def download_with_progress(url: str, timeout: int = 120) -> bytes:
    """下载文件并返回字节内容"""
    print(f"  下载: {url}")

    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.0"
    })

    with urllib.request.urlopen(req, timeout=timeout) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunks = []
        chunk_size = 8192

        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            downloaded += len(chunk)

            if total_size > 0:
                pct = downloaded / total_size * 100
                print(f"    {downloaded/1024/1024:.1f}MB / {total_size/1024/1024:.1f}MB ({pct:.0f}%)", end="\r")
            else:
                print(f"    {downloaded/1024/1024:.1f}MB downloaded", end="\r")

        print()  # 换行
        return b"".join(chunks)


def download_and_extract_dataset(key: str, save_dir: str = "./data/raw/manythings") -> str:
    """
    下载并解压数据集，返回解压后的 txt 文件路径

    Args:
        key: 数据集标识 (如 "eng_deu")
        save_dir: 保存目录

    Returns:
        解压后的 txt 文件路径
    """
    dataset = DATASETS[key]
    dataset_dir = ensure_dir(os.path.join(save_dir, key))
    txt_path = os.path.join(dataset_dir, dataset["filename"])

    # 如果已存在，跳过
    if os.path.exists(txt_path):
        print(f"  [SKIP] {dataset['name']} 已存在")
        return txt_path

    print(f"\n[下载] {dataset['name']} (~{dataset['size_mb']}MB)")

    try:
        # 下载 zip
        zip_bytes = download_with_progress(dataset["url"])

        # 解压
        print(f"  解压中...")
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(dataset_dir)

        print(f"  [OK] {dataset['name']} 完成 -> {txt_path}")
        return txt_path

    except Exception as e:
        print(f"  [ERROR] {dataset['name']} 下载失败: {e}")
        return None


def download_all_datasets(save_dir: str = "./data/raw/manythings"):
    """下载所有 ManyThings 数据集"""
    print("=" * 60)
    print("  下载 ManyThings 英语句子数据集")
    print("=" * 60)
    print(f"共 {len(DATASETS)} 个数据集，预计总大小 ~25MB")
    print("-" * 60)

    downloaded = []
    failed = []

    for key in DATASETS:
        path = download_and_extract_dataset(key, save_dir)
        if path:
            downloaded.append(key)
        else:
            failed.append(key)
        time.sleep(0.5)  # 礼貌延迟

    print("\n" + "=" * 60)
    print("下载结果")
    print("=" * 60)
    print(f"成功: {len(downloaded)}/{len(DATASETS)}")
    if failed:
        print(f"失败: {', '.join(failed)}")

    return downloaded


def list_datasets():
    """列出所有可用数据集"""
    print("=" * 60)
    print("  可用数据集列表")
    print("=" * 60)
    print(f"{'ID':<12} {'名称':<20} {'大小':<10} {'英语句子数':<12}")
    print("-" * 60)

    total_size = 0
    total_lines = 0
    for key, info in DATASETS.items():
        print(f"{key:<12} {info['name']:<20} ~{info['size_mb']:<9.1f}MB {info['english_lines']:<12,}")
        total_size += info['size_mb']
        total_lines += info['english_lines']

    print("-" * 60)
    print(f"{'总计':<12} {'':<20} ~{total_size:<9.1f}MB {total_lines:<12,}")


def main():
    parser = argparse.ArgumentParser(description="下载英语句子数据集")
    parser.add_argument("--list", action="store_true", help="列出可用数据集")
    parser.add_argument("--dataset", type=str, default=None, help="指定下载某个数据集 (如 eng_deu)")
    parser.add_argument("--save-dir", type=str, default="./data/raw/manythings", help="保存目录")

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset:
        if args.dataset not in DATASETS:
            print(f"[ERROR] 未知数据集: {args.dataset}")
            print("可用数据集:")
            list_datasets()
            return
        download_and_extract_dataset(args.dataset, args.save_dir)
    else:
        download_all_datasets(args.save_dir)

    print("\n[OK] 全部完成！下一步: python process_data.py")


if __name__ == "__main__":
    main()
