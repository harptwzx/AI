#!/usr/bin/env python3
"""
全自动数据处理脚本
==================
将下载的原始数据统一处理成 train.txt 和 val.txt

用法:
    python process_data.py          # 处理所有数据
    python process_data.py --sample # 仅用示例数据快速测试
"""

import os
import re
import csv
import json
import bz2
import random
import argparse
from pathlib import Path
from typing import List, Set

# 设置随机种子保证可复现
random.seed(42)


# ============================================================
# 工具函数
# ============================================================
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

    # 移除过长重复字符
    text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)

    return text.strip()

def is_valid_sentence(text: str, min_len: int = 15, max_len: int = 300) -> bool:
    """判断是否为有效英语句子"""
    if not text:
        return False
    if len(text) < min_len or len(text) > max_len:
        return False
    # 至少包含一个字母
    if not re.search(r'[a-zA-Z]', text):
        return False
    # 英语内容比例
    non_ascii = len(re.findall(r'[^\x00-\x7F]', text))
    if non_ascii / len(text) > 0.2:
        return False
    # 不能全是数字或符号
    alpha_count = len(re.findall(r'[a-zA-Z]', text))
    if alpha_count < len(text) * 0.3:
        return False
    return True

def split_to_sentences(text: str) -> List[str]:
    """将文本拆分为句子"""
    # 基于句号、问号、感叹号拆分，但保留缩写
    text = re.sub(r'([\w])\.([\w])', r'\1<DOT>\2', text)  # 保护缩写中的点
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.replace("<DOT>", ".").strip() for s in sentences]
    return [s for s in sentences if s]


# ============================================================
# 1. 处理 Tatoeba
# ============================================================
def process_tatoeba() -> List[str]:
    """处理 Tatoeba 数据"""
    print("\n[1/5] 处理 Tatoeba 数据...")

    filepath = "./data/raw/tatoeba/sentences.csv"
    if not os.path.exists(filepath):
        print(f"  [SKIP] 未找到 {filepath}")
        return []

    sentences = []
    count = 0

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 3 and row[1] == "eng":
                text = clean_text(row[2])
                if is_valid_sentence(text):
                    sentences.append(text)
                count += 1
                if count % 100000 == 0:
                    print(f"  已处理 {count:,} 行，收集 {len(sentences):,} 句")

    print(f"  [OK] Tatoeba: {len(sentences):,} 句")
    return sentences


# ============================================================
# 2. 处理 OpenWebText
# ============================================================
def process_openwebtext(max_docs: int = 100000) -> List[str]:
    """处理 OpenWebText 数据"""
    print("\n[2/5] 处理 OpenWebText 数据...")

    owt_dir = "./data/raw/openwebtext"
    if not os.path.exists(owt_dir):
        print(f"  [SKIP] 未找到 {owt_dir}")
        return []

    sentences = []

    # 查找所有 .txt 文件
    txt_files = list(Path(owt_dir).rglob("*.txt"))

    if not txt_files:
        print(f"  [WARN] 未找到 .txt 文件")
        return []

    print(f"  找到 {len(txt_files)} 个 .txt 文件")

    for i, txt_path in enumerate(txt_files[:max_docs]):
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            text = clean_text(text)
            doc_sentences = split_to_sentences(text)
            for s in doc_sentences:
                if is_valid_sentence(s):
                    sentences.append(s)

            if (i + 1) % 1000 == 0:
                print(f"  已处理 {i+1}/{min(len(txt_files), max_docs)} 文件，收集 {len(sentences):,} 句")
        except Exception as e:
            continue

    print(f"  [OK] OpenWebText: {len(sentences):,} 句")
    return sentences


# ============================================================
# 3. 处理 Wikipedia
# ============================================================
def process_wikipedia() -> List[str]:
    """处理 Wikipedia 数据"""
    print("\n[3/5] 处理 Wikipedia 数据...")

    wiki_dir = "./data/raw/wikipedia"
    if not os.path.exists(wiki_dir):
        print(f"  [SKIP] 未找到 {wiki_dir}")
        return []

    sentences = []

    # 查找 .xml.bz2 文件
    xml_files = list(Path(wiki_dir).glob("*.xml*"))

    if not xml_files:
        print(f"  [WARN] 未找到 .xml 文件")
        return []

    for xml_path in xml_files:
        print(f"  处理 {xml_path.name}...")

        if str(xml_path).endswith(".bz2"):
            opener = lambda: bz2.open(str(xml_path), "rt", encoding="utf-8", errors="replace")
        else:
            opener = lambda: open(str(xml_path), "r", encoding="utf-8", errors="replace")

        buffer = ""
        in_text = False
        article_count = 0

        with opener() as f:
            for line in f:
                if "<text" in line:
                    in_text = True
                    buffer = ""
                elif "</text>" in line:
                    in_text = False
                    article_count += 1

                    # 提取纯文本
                    text = extract_wiki_text(buffer)
                    text = clean_text(text)
                    doc_sentences = split_to_sentences(text)
                    for s in doc_sentences:
                        if is_valid_sentence(s):
                            sentences.append(s)

                    if article_count % 1000 == 0:
                        print(f"    已处理 {article_count:,} 篇文章，收集 {len(sentences):,} 句")

                    buffer = ""
                elif in_text:
                    buffer += line

    print(f"  [OK] Wikipedia: {len(sentences):,} 句")
    return sentences

def extract_wiki_text(markup: str) -> str:
    """从 wiki markup 提取纯文本"""
    # 移除模板
    text = re.sub(r'\{\{.*?\}\}', '', markup, flags=re.DOTALL)
    # 移除引用
    text = re.sub(r'<ref.*?</ref>', '', text, flags=re.DOTALL)
    # 移除 HTML 标签
    text = re.sub(r'<.*?>', '', text)
    # 移除 wiki 链接 [[...]]
    text = re.sub(r'\[\[([^|]*\|)?(.*?)(\|.*?)?\]\]', r'\2', text)
    # 移除文件/图片链接
    text = re.sub(r'\[\[(File|Image):.*?\]\]', '', text)
    # 移除表格
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # 移除标题标记
    text = re.sub(r'=+\s*(.*?)\s*=+', r'\1', text)
    # 移除特殊字符
    text = re.sub(r'&[a-zA-Z]+;', ' ', text)
    return text


# ============================================================
# 4. 处理 Gutenberg
# ============================================================
def process_gutenberg() -> List[str]:
    """处理 Project Gutenberg 书籍"""
    print("\n[4/5] 处理 Gutenberg 数据...")

    gutenberg_dir = "./data/raw/gutenberg"
    if not os.path.exists(gutenberg_dir):
        print(f"  [SKIP] 未找到 {gutenberg_dir}")
        return []

    sentences = []
    txt_files = list(Path(gutenberg_dir).glob("*.txt"))

    print(f"  找到 {len(txt_files)} 本书")

    for txt_path in txt_files:
        try:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()

            # 移除 Gutenberg 头尾声明
            text = remove_gutenberg_boilerplate(text)
            text = clean_text(text)
            doc_sentences = split_to_sentences(text)
            for s in doc_sentences:
                if is_valid_sentence(s):
                    sentences.append(s)
        except Exception as e:
            print(f"  [ERROR] 处理 {txt_path.name}: {e}")

    print(f"  [OK] Gutenberg: {len(sentences):,} 句")
    return sentences

def remove_gutenberg_boilerplate(text: str) -> str:
    """移除 Gutenberg 标准头尾声明"""
    start_markers = ["*** START OF", "***START OF", "This eBook is for the use"]
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg", "End of the Project Gutenberg"]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            nl_idx = text.find("\n", idx)
            if nl_idx != -1:
                start_idx = nl_idx + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx]


# ============================================================
# 5. 处理 C4
# ============================================================
def process_c4(max_lines: int = 500000) -> List[str]:
    """处理 C4 数据"""
    print("\n[5/5] 处理 C4 数据...")

    c4_dir = "./data/raw/c4"
    if not os.path.exists(c4_dir):
        print(f"  [SKIP] 未找到 {c4_dir}")
        return []

    sentences = []
    json_files = list(Path(c4_dir).glob("*.json*"))

    if not json_files:
        print(f"  [WARN] 未找到 .json 文件")
        return []

    print(f"  找到 {len(json_files)} 个文件")

    for json_path in json_files:
        print(f"  处理 {json_path.name}...")

        if str(json_path).endswith(".gz"):
            import gzip
            opener = lambda: gzip.open(str(json_path), "rt", encoding="utf-8", errors="replace")
        else:
            opener = lambda: open(str(json_path), "r", encoding="utf-8", errors="replace")

        with opener() as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    text = clean_text(text)
                    doc_sentences = split_to_sentences(text)
                    for s in doc_sentences:
                        if is_valid_sentence(s):
                            sentences.append(s)
                except json.JSONDecodeError:
                    continue

                if (i + 1) % 10000 == 0:
                    print(f"    已处理 {i+1:,} 行，收集 {len(sentences):,} 句")

    print(f"  [OK] C4: {len(sentences):,} 句")
    return sentences


# ============================================================
# 保存数据
# ============================================================
def save_data(all_sentences: List[str], train_ratio: float = 0.95):
    """保存训练集和验证集"""
    processed_dir = ensure_dir("./data/processed")

    # 去重
    print(f"\n[Data] 去重前: {len(all_sentences):,} 句")
    seen: Set[str] = set()
    unique_sentences = []
    for s in all_sentences:
        if s not in seen:
            seen.add(s)
            unique_sentences.append(s)
    print(f"[Data] 去重后: {len(unique_sentences):,} 句")

    # 随机打乱
    random.shuffle(unique_sentences)

    # 划分
    split_idx = int(len(unique_sentences) * train_ratio)
    train_sentences = unique_sentences[:split_idx]
    val_sentences = unique_sentences[split_idx:]

    # 保存
    train_path = os.path.join(processed_dir, "train.txt")
    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_sentences:
            f.write(s + "\n")
    print(f"[Data] 训练集已保存: {train_path} ({len(train_sentences):,} 句)")

    val_path = os.path.join(processed_dir, "val.txt")
    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_sentences:
            f.write(s + "\n")
    print(f"[Data] 验证集已保存: {val_path} ({len(val_sentences):,} 句)")

    # 统计信息
    total_chars = sum(len(s) for s in unique_sentences)
    print(f"\n[Data] 统计:")
    print(f"  总句子数: {len(unique_sentences):,}")
    print(f"  总字符数: {total_chars:,}")
    print(f"  平均句长: {total_chars / len(unique_sentences):.1f} 字符")
    print(f"  训练集: {len(train_sentences):,} ({len(train_sentences)/len(unique_sentences)*100:.1f}%)")
    print(f"  验证集: {len(val_sentences):,} ({len(val_sentences)/len(unique_sentences)*100:.1f}%)")


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="数据处理")
    parser.add_argument("--sample", action="store_true", help="仅使用示例数据快速测试")
    args = parser.parse_args()

    print("=" * 60)
    print("  English ChatAI — 数据处理")
    print("=" * 60)

    all_sentences = []

    # 处理各数据源
    tatoeba = process_tatoeba()
    all_sentences.extend(tatoeba)

    owt = process_openwebtext()
    all_sentences.extend(owt)

    wiki = process_wikipedia()
    all_sentences.extend(wiki)

    gutenberg = process_gutenberg()
    all_sentences.extend(gutenberg)

    c4 = process_c4()
    all_sentences.extend(c4)

    if not all_sentences:
        print("\n[ERROR] 没有收集到任何数据！")
        print("[HINT] 请先运行: python download_data.py")
        return

    # 保存
    save_data(all_sentences)

    print("\n" + "=" * 60)
    print("[OK] 数据处理完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
