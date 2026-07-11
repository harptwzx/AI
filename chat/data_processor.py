"""
数据预处理脚本
==============
处理从各来源下载的原始数据，生成统一的训练文本文件。
支持: Tatoeba, OpenWebText, Wikipedia, Gutenberg, C4
"""

import os
import json
import csv
import re
import glob
import tarfile
import bz2
import gzip
from pathlib import Path
from typing import List, Iterator
import xml.etree.ElementTree as ET

class DataProcessor:
    """统一数据预处理器"""

    def __init__(self, raw_dir: str = "./data/raw", processed_dir: str = "./data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(processed_dir, exist_ok=True)

    # ─────────────────────────────────────────────────────────
    # 通用文本清洗
    # ─────────────────────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """清洗文本"""
        if not text:
            return ""

        # 移除多余空白
        text = re.sub(r'\s+', ' ', text)

        # 移除控制字符（保留换行）
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f]', '', text)

        # 移除 URL
        text = re.sub(r'https?://\S+', '', text)

        # 移除多余标点重复
        text = re.sub(r'([!?.]){3,}', r'\1\1\1', text)

        # 移除过长无意义序列
        text = re.sub(r'(.)\1{10,}', r'\1\1\1', text)

        return text.strip()

    def is_valid_sentence(self, text: str, min_len: int = 10, max_len: int = 500) -> bool:
        """判断是否为有效句子"""
        if not text:
            return False
        if len(text) < min_len or len(text) > max_len:
            return False
        # 至少包含一个字母
        if not re.search(r'[a-zA-Z]', text):
            return False
        # 非英语内容比例不能太高
        non_ascii = len(re.findall(r'[^\x00-\x7F]', text))
        if non_ascii / len(text) > 0.3:
            return False
        return True

    # ─────────────────────────────────────────────────────────
    # Tatoeba 处理
    # ─────────────────────────────────────────────────────────
    def process_tatoeba(self) -> List[str]:
        """处理 Tatoeba 数据"""
        print("[DataProcessor] 处理 Tatoeba 数据...")

        sentences_path = os.path.join(self.raw_dir, "tatoeba", "sentences.csv")
        if not os.path.exists(sentences_path):
            print(f"[DataProcessor] 警告: 未找到 {sentences_path}")
            print("[DataProcessor] 请从 https://tatoeba.org/en/downloads 下载 sentences.csv")
            return []

        english_sentences = []

        with open(sentences_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) >= 3 and row[1] == "eng":
                    sentence = self.clean_text(row[2])
                    if self.is_valid_sentence(sentence):
                        english_sentences.append(sentence)

        print(f"[DataProcessor] Tatoeba 英语句子: {len(english_sentences):,}")
        return english_sentences

    # ─────────────────────────────────────────────────────────
    # OpenWebText 处理
    # ─────────────────────────────────────────────────────────
    def process_openwebtext(self, max_docs: int = 500000) -> List[str]:
        """处理 OpenWebText 数据"""
        print("[DataProcessor] 处理 OpenWebText 数据...")

        owt_dir = os.path.join(self.raw_dir, "openwebtext")

        # 查找 tar.xz 或解压后的文件
        tar_files = glob.glob(os.path.join(owt_dir, "*.tar.xz"))
        txt_files = glob.glob(os.path.join(owt_dir, "**", "*.txt"), recursive=True)

        sentences = []

        if tar_files and not txt_files:
            print(f"[DataProcessor] 解压 tar 文件...")
            for tar_path in tar_files:
                with tarfile.open(tar_path, "r:xz") as tar:
                    for member in tar.getmembers():
                        if member.isfile() and member.name.endswith(".txt"):
                            f = tar.extractfile(member)
                            if f:
                                text = f.read().decode("utf-8", errors="replace")
                                text = self.clean_text(text)
                                # 将文档拆分为句子
                                doc_sentences = self.split_to_sentences(text)
                                sentences.extend([s for s in doc_sentences if self.is_valid_sentence(s)])

                                if len(sentences) >= max_docs * 3:
                                    break
        elif txt_files:
            for txt_path in txt_files[:max_docs]:
                with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                    text = self.clean_text(f.read())
                    doc_sentences = self.split_to_sentences(text)
                    sentences.extend([s for s in doc_sentences if self.is_valid_sentence(s)])
        else:
            print(f"[DataProcessor] 警告: 未在 {owt_dir} 找到数据")
            print("[DataProcessor] 请从 https://skylion007.github.io/OpenWebTextCorpus/ 下载")

        print(f"[DataProcessor] OpenWebText 句子: {len(sentences):,}")
        return sentences

    # ─────────────────────────────────────────────────────────
    # Wikipedia 处理
    # ─────────────────────────────────────────────────────────
    def process_wikipedia(self) -> List[str]:
        """处理 Wikipedia dump"""
        print("[DataProcessor] 处理 Wikipedia 数据...")

        wiki_dir = os.path.join(self.raw_dir, "wikipedia")
        xml_files = glob.glob(os.path.join(wiki_dir, "*.xml*"))

        sentences = []

        for xml_path in xml_files:
            print(f"[DataProcessor] 处理 {os.path.basename(xml_path)}...")

            # 处理 .bz2 压缩
            if xml_path.endswith(".bz2"):
                opener = lambda: bz2.open(xml_path, "rt", encoding="utf-8", errors="replace")
            else:
                opener = lambda: open(xml_path, "r", encoding="utf-8", errors="replace")

            with opener() as f:
                # 逐行解析，避免内存爆炸
                buffer = ""
                in_text = False

                for line in f:
                    if "<text" in line:
                        in_text = True
                        buffer = ""
                    elif "</text>" in line:
                        in_text = False
                        # 提取纯文本（移除 wiki markup）
                        text = self.extract_wiki_text(buffer)
                        text = self.clean_text(text)
                        doc_sentences = self.split_to_sentences(text)
                        sentences.extend([s for s in doc_sentences if self.is_valid_sentence(s)])
                    elif in_text:
                        buffer += line

        print(f"[DataProcessor] Wikipedia 句子: {len(sentences):,}")
        return sentences

    def extract_wiki_text(self, wiki_markup: str) -> str:
        """从 wiki markup 中提取纯文本"""
        # 移除模板
        text = re.sub(r'\{\{.*?\}\}', '', wiki_markup, flags=re.DOTALL)
        # 移除引用
        text = re.sub(r'<ref.*?</ref>', '', text, flags=re.DOTALL)
        # 移除 HTML 标签
        text = re.sub(r'<.*?>', '', text)
        # 移除 wiki 链接 [[...]]
        text = re.sub(r'\[\[([^|]*\|)?(.*?)(\|.*?)?\]\]', r'\2', text)
        # 移除文件链接
        text = re.sub(r'\[\[File:.*?\]\]', '', text)
        text = re.sub(r'\[\[Image:.*?\]\]', '', text)
        # 移除表格标记
        text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
        # 移除标题标记
        text = re.sub(r'=+\s*(.*?)\s*=+', r'\1', text)
        # 移除特殊标记
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)

        return text

    # ─────────────────────────────────────────────────────────
    # Gutenberg 处理
    # ─────────────────────────────────────────────────────────
    def process_gutenberg(self) -> List[str]:
        """处理 Project Gutenberg 书籍"""
        print("[DataProcessor] 处理 Gutenberg 数据...")

        gutenberg_dir = os.path.join(self.raw_dir, "gutenberg")
        txt_files = glob.glob(os.path.join(gutenberg_dir, "**", "*.txt"), recursive=True)

        sentences = []

        for txt_path in txt_files:
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
                # 移除 Gutenberg 头尾声明
                text = self.remove_gutenberg_boilerplate(text)
                text = self.clean_text(text)
                doc_sentences = self.split_to_sentences(text)
                sentences.extend([s for s in doc_sentences if self.is_valid_sentence(s)])

        print(f"[DataProcessor] Gutenberg 句子: {len(sentences):,}")
        return sentences

    def remove_gutenberg_boilerplate(self, text: str) -> str:
        """移除 Project Gutenberg 的标准头尾声明"""
        # 常见的起始标记
        start_markers = [
            "*** START OF",
            "***START OF",
            "This eBook is for the use",
            "Project Gutenberg",
        ]
        # 常见的结束标记
        end_markers = [
            "*** END OF",
            "***END OF",
            "End of Project Gutenberg",
            "End of the Project Gutenberg",
        ]

        # 找起始位置
        start_idx = 0
        for marker in start_markers:
            idx = text.find(marker)
            if idx != -1:
                # 找到该标记后的换行位置
                nl_idx = text.find("\n", idx)
                if nl_idx != -1:
                    start_idx = nl_idx + 1
                break

        # 找结束位置
        end_idx = len(text)
        for marker in end_markers:
            idx = text.find(marker)
            if idx != -1:
                end_idx = idx
                break

        return text[start_idx:end_idx]

    # ─────────────────────────────────────────────────────────
    # C4 处理
    # ─────────────────────────────────────────────────────────
    def process_c4(self, max_lines: int = 2000000) -> List[str]:
        """处理 C4 数据集"""
        print("[DataProcessor] 处理 C4 数据...")

        c4_dir = os.path.join(self.raw_dir, "c4")
        json_files = glob.glob(os.path.join(c4_dir, "*.json*"))

        sentences = []

        for json_path in json_files:
            print(f"[DataProcessor] 处理 {os.path.basename(json_path)}...")

            if json_path.endswith(".gz"):
                opener = lambda: gzip.open(json_path, "rt", encoding="utf-8", errors="replace")
            else:
                opener = lambda: open(json_path, "r", encoding="utf-8", errors="replace")

            with opener() as f:
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    try:
                        data = json.loads(line)
                        text = data.get("text", "")
                        text = self.clean_text(text)
                        doc_sentences = self.split_to_sentences(text)
                        sentences.extend([s for s in doc_sentences if self.is_valid_sentence(s)])
                    except json.JSONDecodeError:
                        continue

        print(f"[DataProcessor] C4 句子: {len(sentences):,}")
        return sentences

    # ─────────────────────────────────────────────────────────
    # 通用工具
    # ─────────────────────────────────────────────────────────
    def split_to_sentences(self, text: str) -> List[str]:
        """将文本拆分为句子列表"""
        # 简单的句子拆分（基于句号、问号、感叹号）
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def save_sentences(self, sentences: List[str], filename: str):
        """保存句子到文件"""
        filepath = os.path.join(self.processed_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for sentence in sentences:
                f.write(sentence + "\n")
        print(f"[DataProcessor] 已保存 {len(sentences):,} 行到 {filepath}")

    def process_all(self):
        """处理所有可用数据"""
        all_sentences = []

        # 处理各数据源
        tatoeba = self.process_tatoeba()
        all_sentences.extend(tatoeba)

        owt = self.process_openwebtext()
        all_sentences.extend(owt)

        wiki = self.process_wikipedia()
        all_sentences.extend(wiki)

        gutenberg = self.process_gutenberg()
        all_sentences.extend(gutenberg)

        c4 = self.process_c4()
        all_sentences.extend(c4)

        # 去重
        print(f"[DataProcessor] 去重前: {len(all_sentences):,}")
        all_sentences = list(dict.fromkeys(all_sentences))
        print(f"[DataProcessor] 去重后: {len(all_sentences):,}")

        # 保存
        self.save_sentences(all_sentences, "all_sentences.txt")

        # 划分训练/验证集 (90/10)
        split_idx = int(len(all_sentences) * 0.9)
        train_sentences = all_sentences[:split_idx]
        val_sentences = all_sentences[split_idx:]

        self.save_sentences(train_sentences, "train.txt")
        self.save_sentences(val_sentences, "val.txt")

        print(f"[DataProcessor] 训练集: {len(train_sentences):,}")
        print(f"[DataProcessor] 验证集: {len(val_sentences):,}")

        return train_sentences, val_sentences


if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all()
