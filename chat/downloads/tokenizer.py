"""
自实现 BPE Tokenizer
==================
从零训练 BPE，不依赖任何外部 tokenizer 库。
生成 token 对照文件（vocab.json 和 merges.txt）。
"""

import json
import collections
import os
import pickle
import time
from typing import List, Dict, Tuple

class BPETokenizer:
    """
    字节对编码 (Byte Pair Encoding) Tokenizer

    输出文件:
    - vocab.json: {token_id: token_str} 对照表
    - merges.txt: 合并规则列表
    - tokenizer_config.json: 配置信息
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256  # 保留 256 个基础字节

        # 基础 vocab: 0-255 对应 UTF-8 bytes
        self.vocab: Dict[int, str] = {i: bytes([i]).decode("utf-8", errors="replace") for i in range(256)}
        self.merges: List[Tuple[int, int]] = []  # 合并规则
        self.special_tokens = {
            "<|pad|>": vocab_size,      # 填充
            "<|endoftext|>": vocab_size + 1,  # 文本结束
            "<|user|>": vocab_size + 2,      # 用户消息标记
            "<|assistant|>": vocab_size + 3,  # AI 回复标记
        }
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """统计相邻 token 对出现频率"""
        counts = collections.Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i+1])] += 1
        return counts

    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """将指定 pair 合并为新 token id"""
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i+1]) == pair:
                newids.append(new_id)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def train(self, texts: List[str], save_dir: str = "./data/tokenizer"):
        """
        在文本上训练 BPE

        Args:
            texts: 训练文本列表
            save_dir: 保存 tokenizer 文件的目录
        """
        print(f"[Tokenizer] 开始训练 BPE，目标 vocab_size={self.vocab_size}")
        print(f"[Tokenizer] 文本总数: {len(texts)}")

        # 合并所有文本
        print("[Tokenizer] 编码文本为 UTF-8 bytes...")
        all_text = "\n".join(texts)
        ids = list(all_text.encode("utf-8"))
        print(f"[Tokenizer] 总 bytes: {len(ids):,}")

        # BPE 合并
        print(f"[Tokenizer] 开始 {self.num_merges} 次合并...")
        start_time = time.time()

        for i in range(self.num_merges):
            stats = self._get_stats(ids)
            if not stats:
                print(f"[Tokenizer] 提前终止: 无更多可合并对 (step {i})")
                break

            best = max(stats, key=stats.get)
            new_id = 256 + i
            ids = self._merge(ids, best, new_id)
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if (i + 1) % 1000 == 0 or i < 5:
                elapsed = time.time() - start_time
                print(f"  Merge {i+1}/{self.num_merges} | "
                      f"pair={best}, freq={stats[best]}, "
                      f"seq_len={len(ids):,}, elapsed={elapsed:.1f}s")

        # 添加 special tokens 到 vocab
        for token_str, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_str

        total_time = time.time() - start_time
        print(f"[Tokenizer] 训练完成！耗时 {total_time:.1f}s")
        print(f"[Tokenizer] 最终 vocab_size: {len(self.vocab)}")
        print(f"[Tokenizer] 最终序列长度: {len(ids):,} (压缩率: {len(all_text.encode('utf-8'))/len(ids):.2f}x)")

        # 保存
        os.makedirs(save_dir, exist_ok=True)
        self.save(save_dir)

        return ids

    def save(self, save_dir: str):
        """保存 tokenizer 到文件"""
        # vocab.json: {id: token_str}
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            # 确保所有值都是字符串
            vocab_str = {str(k): v for k, v in self.vocab.items()}
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)
        print(f"[Tokenizer] vocab.json 已保存 ({len(self.vocab)} tokens)")

        # merges.txt: 每行一个合并规则
        merges_path = os.path.join(save_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("# BPE merges\n")
            f.write("# format: token_a token_b\n")
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"[Tokenizer] merges.txt 已保存 ({len(self.merges)} rules)")

        # tokenizer_config.json
        config = {
            "vocab_size": self.vocab_size,
            "actual_vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
            "version": "1.0"
        }
        config_path = os.path.join(save_dir, "tokenizer_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"[Tokenizer] tokenizer_config.json 已保存")

    @classmethod
    def load(cls, load_dir: str):
        """从文件加载 tokenizer"""
        instance = cls(vocab_size=32000)  # 先创建默认实例

        # 加载 vocab
        vocab_path = os.path.join(load_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_str = json.load(f)
            instance.vocab = {int(k): v for k, v in vocab_str.items()}

        # 加载 merges
        merges_path = os.path.join(load_dir, "merges.txt")
        instance.merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                a, b = line.split()
                instance.merges.append((int(a), int(b)))

        # 加载 config
        config_path = os.path.join(load_dir, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            instance.vocab_size = config["vocab_size"]
            instance.special_tokens = config["special_tokens"]
            instance.inverse_special = {v: k for k, v in instance.special_tokens.items()}

        print(f"[Tokenizer] 已从 {load_dir} 加载")
        print(f"[Tokenizer] vocab_size={instance.vocab_size}, merges={len(instance.merges)}")
        return instance

    def encode(self, text: str) -> List[int]:
        """文本 → token ids"""
        # 先处理 special tokens
        for token_str, token_id in self.special_tokens.items():
            text = text.replace(token_str, f"\x00{token_id}\x00")

        parts = text.split("\x00")
        result = []

        for part in parts:
            if part.isdigit():
                result.append(int(part))
            else:
                ids = list(part.encode("utf-8"))
                for pair in self.merges:
                    new_id = 256 + self.merges.index(pair)
                    ids = self._merge(ids, pair, new_id)
                result.extend(ids)

        return result

    def decode(self, ids: List[int]) -> str:
        """token ids → 文本"""
        bytes_list = []
        for i in ids:
            if i in self.inverse_special:
                bytes_list.append(self.inverse_special[i].encode("utf-8"))
            elif i in self.vocab:
                bytes_list.append(self.vocab[i].encode("utf-8", errors="replace"))
            else:
                bytes_list.append(b"<unk>")

        return b"".join(bytes_list).decode("utf-8", errors="replace")

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        """批量编码"""
        return [self.encode(t) for t in texts]

    def decode_batch(self, id_lists: List[List[int]]) -> List[str]:
        """批量解码"""
        return [self.decode(ids) for ids in id_lists]


# ============================================================
# 简易字符级 Tokenizer（BPE 训练前的备选方案）
# ============================================================
class CharTokenizer:
    """字符级 Tokenizer，作为 BPE 的轻量替代"""

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.special_tokens = {
            "<pad>": 0,
            "<unk>": 1,
            "<endoftext>": 2,
            "<user>": 3,
            "<assistant>": 4,
        }
        # 预留 0-99 给 special tokens
        self.next_id = 100

    def train(self, texts: List[str], save_dir: str = "./data/tokenizer_char"):
        """构建字符表"""
        print("[CharTokenizer] 构建字符表...")
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        for char in sorted(all_chars):
            self.char_to_id[char] = self.next_id
            self.id_to_char[self.next_id] = char
            self.next_id += 1

        # 合并 special tokens
        for k, v in self.special_tokens.items():
            self.char_to_id[k] = v
            self.id_to_char[v] = k

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "char_vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"char_to_id": self.char_to_id, "id_to_char": self.id_to_char}, 
                     f, ensure_ascii=False, indent=2)

        print(f"[CharTokenizer] 字符表大小: {len(self.char_to_id)}")
        return self

    def encode(self, text: str) -> List[int]:
        return [self.char_to_id.get(c, self.special_tokens["<unk>"]) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, "<unk>") for i in ids)


if __name__ == "__main__":
    # 测试
    sample_texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
    ]

    print("=" * 60)
    print("测试 CharTokenizer")
    print("=" * 60)
    ct = CharTokenizer()
    ct.train(sample_texts)
    encoded = ct.encode("Hello world")
    print(f"编码: {encoded}")
    print(f"解码: {ct.decode(encoded)}")

    print("\n" + "=" * 60)
    print("测试 BPETokenizer")
    print("=" * 60)
    bt = BPETokenizer(vocab_size=300)
    bt.train(sample_texts)
    encoded = bt.encode("Hello world")
    print(f"编码: {encoded}")
    print(f"解码: {bt.decode(encoded)}")
