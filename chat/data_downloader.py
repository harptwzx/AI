"""
数据下载指南 — 手动下载并放置到指定目录
============================================

请按以下步骤手动下载数据，并放入对应目录：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集 1】Tatoeba 英语例句库（推荐首选，质量高、句子规范）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
网址: https://tatoeba.org/en/downloads
下载: sentences.csv + links.csv
放置目录: ./data/raw/tatoeba/
文件名: sentences.csv, links.csv
说明: 约 1200 万条多语言例句，英语部分约 400 万条
      句子短、语法规范、适合训练语法语感
      每周日更新，CC-BY 2.0 FR 许可

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集 2】OpenWebText（网页文本，语料丰富）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
网址: https://skylion007.github.io/OpenWebTextCorpus/
下载: openwebtext.tar.xz (约 12.8 GB)
放置目录: ./data/raw/openwebtext/
文件名: openwebtext.tar.xz
说明: 约 800 万篇网页文档，38GB 解压后文本
      GPT-2 训练数据的开源复刻版
      CC0 许可，可自由使用

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集 3】Wikipedia 英文 dump（知识密集、语法规范）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
网址: https://dumps.wikimedia.org/enwiki/latest/
下载: enwiki-latest-pages-articles.xml.bz2 (约 20 GB)
放置目录: ./data/raw/wikipedia/
文件名: enwiki-latest-pages-articles.xml.bz2
说明: 英文维基百科全量数据
      需要先用 WikiExtractor 提取纯文本

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集 4】Project Gutenberg（经典文学，语言优美）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
网址: https://www.gutenberg.org/
或批量下载: https://www.gutenberg.org/files/
放置目录: ./data/raw/gutenberg/
文件名: 多个 .txt 文件
说明: 7 万+ 本公版书，文学性强
      适合训练优美语感和长文本连贯性

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集 5】C4 / Colossal Clean Crawled Corpus（已清洗网页）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
网址: https://huggingface.co/datasets/allenai/c4
或: https://www.kaggle.com/datasets/allenai/c4
放置目录: ./data/raw/c4/
文件名: c4-en*.json.gz
说明: Google T5 训练用的清洗版 Common Crawl
      约 365M 网页，已过滤低质量内容

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【数据集 6】English Grammar Exercises（语法练习，可选）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
网址: https://www.kaggle.com/datasets/ziya07/chatbot-based-english-learning-dataset
下载: chatbot_language_tutor_dataset.csv
放置目录: ./data/raw/grammar_exercises/
文件名: chatbot_language_tutor_dataset.csv
说明: 仅 200 行，含正确/错误句子对照
      可用于后续语法纠错微调阶段

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
推荐组合（训练语法语感阶段）:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
主数据: Tatoeba (40%) + OpenWebText (30%) + Wikipedia (20%) + Gutenberg (10%)
"""

import os

def create_data_dirs():
    """创建数据目录结构"""
    dirs = [
        "./data/raw/tatoeba",
        "./data/raw/openwebtext",
        "./data/raw/wikipedia",
        "./data/raw/gutenberg",
        "./data/raw/c4",
        "./data/raw/grammar_exercises",
        "./data/processed",
        "./data/tokenizer",
        "./models",
        "./logs",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"[OK] {d}")

if __name__ == "__main__":
    create_data_dirs()
    print("\n" + "="*60)
    print("目录创建完成！请按上方说明手动下载数据到对应目录。")
    print("="*60)
