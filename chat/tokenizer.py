"""
自实现 Tokenizer
================
支持两种模式:
1. CharTokenizer: 字符级，零训练时间，适合快速开始
2. BPETokenizer: 字节对编码，训练慢但压缩率高
"""

import json
import collections
import os
import time
from typing import List, Dict, Tuple


# ============================================================
# 1. CharTokenizer — 字符级 Tokenizer（推荐快速开始）
# ============================================================
class CharTokenizer:
    """
    字符级 Tokenizer

    每个可见字符一个 token，零训练时间。
    对于 ~18MB 的英语数据，vocab 大小约 100-150。
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<endoftext>": 2,
        "<user>": 3,
        "<assistant>": 4,
    }

    def __init__(self):
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0

    def build_vocab(self, texts: List[str], save_dir: str = "./data/tokenizer"):
        """从文本构建字符表（瞬间完成）"""
        print("[CharTokenizer] 构建字符表...")

        # 收集所有字符
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        # 排序（让常用字符在前面，便于查看）
        sorted_chars = sorted(all_chars)

        # 构建映射
        # 预留 0-99 给 special tokens
        self.char_to_id = dict(self.SPECIAL_TOKENS)
        next_id = 100

        for char in sorted_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                next_id += 1

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = next_id

        print(f"[CharTokenizer] 字符表大小: {self.vocab_size}")
        print(f"[CharTokenizer] 特殊 token: {list(self.SPECIAL_TOKENS.keys())}")

        # 保存
        self.save(save_dir)
        return self

    def save(self, save_dir: str):
        """保存 tokenizer"""
        os.makedirs(save_dir, exist_ok=True)

        data = {
            "type": "char",
            "vocab_size": self.vocab_size,
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char,
            "special_tokens": self.SPECIAL_TOKENS,
        }

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 保存可读版本
        with open(os.path.join(save_dir, "vocab_readable.txt"), "w", encoding="utf-8") as f:
            f.write(f"# CharTokenizer Vocab\n")
            f.write(f"# vocab_size: {self.vocab_size}\n")
            f.write(f"# type: char\n")
            f.write("-" * 50 + "\n")
            f.write(f"{'ID':<8} {'Char':<20} {'Repr'}\n")
            f.write("-" * 50 + "\n")

            for token_id in sorted(self.id_to_char.keys()):
                char = self.id_to_char[token_id]
                if token_id in self.SPECIAL_TOKENS.values():
                    char_type = "SPECIAL"
                else:
                    char_type = "CHAR"
                repr_str = repr(char)
                f.write(f"{token_id:<8} {char:<20} {repr_str}\n")

        print(f"[CharTokenizer] 已保存到 {save_dir}")

    @classmethod
    def load(cls, load_dir: str):
        """加载 tokenizer"""
        config_path = os.path.join(load_dir, "tokenizer_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls()
        instance.char_to_id = {k: int(v) for k, v in data["char_to_id"].items()}
        instance.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        instance.vocab_size = data["vocab_size"]

        print(f"[CharTokenizer] 已从 {load_dir} 加载")
        print(f"[CharTokenizer] vocab_size={instance.vocab_size}")
        return instance

    def encode(self, text: str) -> List[int]:
        """文本 → token ids"""
        return [self.char_to_id.get(c, self.SPECIAL_TOKENS["<unk>"]) for c in text]

    def decode(self, ids: List[int]) -> str:
        """token ids → 文本"""
        return "".join(self.id_to_char.get(i, "<unk>") for i in ids)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, id_lists: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in id_lists]


# ============================================================
# 2. BPETokenizer — 字节对编码（原版，保留但慢）
# ============================================================
class BPETokenizer:
    """
    字节对编码 (Byte Pair Encoding) Tokenizer

    输出文件:
    - vocab.json: {token_id: token_str} 对照表
    - merges.txt: 合并规则列表
    - tokenizer_config.json: 配置信息

    注意: 纯 Python 实现，大数据集训练很慢。
    建议先用 CharTokenizer 快速开始。
    """

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.num_merges = vocab_size - 256

        self.vocab: Dict[int, str] = {i: bytes([i]).decode("utf-8", errors="replace") for i in range(256)}
        self.merges: List[Tuple[int, int]] = []
        self.special_tokens = {
            "<|pad|>": vocab_size,
            "<|endoftext|>": vocab_size + 1,
            "<|user|>": vocab_size + 2,
            "<|assistant|>": vocab_size + 3,
        }
        self.inverse_special = {v: k for k, v in self.special_tokens.items()}

    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts = collections.Counter()
        for i in range(len(ids) - 1):
            counts[(ids[i], ids[i+1])] += 1
        return counts

    def _merge(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
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

    def train(self, texts: List[str], save_dir: str = "./data/tokenizer", max_merges: int = None):
        """训练 BPE"""
        print(f"[BPETokenizer] 开始训练 BPE，目标 vocab_size={self.vocab_size}")
        print(f"[BPETokenizer] 文本总数: {len(texts)}")

        all_text = "\n".join(texts)
        ids = list(all_text.encode("utf-8"))
        print(f"[BPETokenizer] 总 bytes: {len(ids):,}")

        num_merges = max_merges or self.num_merges
        print(f"[BPETokenizer] 开始 {num_merges} 次合并...")
        start_time = time.time()

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                print(f"[BPETokenizer] 提前终止: 无更多可合并对 (step {i})")
                break

            best = max(stats, key=stats.get)
            new_id = 256 + i
            ids = self._merge(ids, best, new_id)
            self.merges.append(best)
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]

            if (i + 1) % 1000 == 0 or i < 5:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * (num_merges - i - 1)
                print(f"  Merge {i+1}/{num_merges} | "
                      f"pair={best}, freq={stats[best]}, "
                      f"seq_len={len(ids):,}, elapsed={elapsed:.0f}s, ETA={eta/60:.0f}min")

        for token_str, token_id in self.special_tokens.items():
            self.vocab[token_id] = token_str

        total_time = time.time() - start_time
        print(f"[BPETokenizer] 训练完成！耗时 {total_time:.0f}s")
        print(f"[BPETokenizer] 最终 vocab_size: {len(self.vocab)}")

        os.makedirs(save_dir, exist_ok=True)
        self.save(save_dir)
        return ids

    def save(self, save_dir: str):
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            vocab_str = {str(k): v for k, v in self.vocab.items()}
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)
        print(f"[BPETokenizer] vocab.json 已保存 ({len(self.vocab)} tokens)")

        merges_path = os.path.join(save_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("# BPE merges\n")
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")
        print(f"[BPETokenizer] merges.txt 已保存 ({len(self.merges)} rules)")

        config = {
            "vocab_size": self.vocab_size,
            "actual_vocab_size": len(self.vocab),
            "num_merges": len(self.merges),
            "special_tokens": self.special_tokens,
            "type": "bpe",
            "version": "1.0"
        }
        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, load_dir: str):
        instance = cls(vocab_size=32000)

        vocab_path = os.path.join(load_dir, "vocab.json")
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_str = json.load(f)
            instance.vocab = {int(k): v for k, v in vocab_str.items()}

        merges_path = os.path.join(load_dir, "merges.txt")
        instance.merges = []
        with open(merges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                a, b = line.split()
                instance.merges.append((int(a), int(b)))

        config_path = os.path.join(load_dir, "tokenizer_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            instance.vocab_size = config["vocab_size"]
            instance.special_tokens = config["special_tokens"]
            instance.inverse_special = {v: k for k, v in instance.special_tokens.items()}

        print(f"[BPETokenizer] 已从 {load_dir} 加载")
        print(f"[BPETokenizer] vocab_size={instance.vocab_size}, merges={len(instance.merges)}")
        return instance

    def encode(self, text: str) -> List[int]:
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
        bytes_list = []
        for i in ids:
            if i in self.inverse_special:
                bytes_list.append(self.inverse_special[i].encode("utf-8"))
            elif i in self.vocab:
                bytes_list.append(self.vocab[i].encode("utf-8", errors="replace"))
            else:
                bytes_list.append(b"<unk>")

        return b"".join(bytes_list).decode("utf-8", errors="replace")


# ============================================================
# 3. 统一加载接口
# ============================================================
def load_tokenizer(load_dir: str = "./data/tokenizer"):
    """自动检测并加载 tokenizer（支持 char 和 bpe）"""
    config_path = os.path.join(load_dir, "tokenizer_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 tokenizer 配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_type = config.get("type", "bpe")

    if tokenizer_type == "char":
        return CharTokenizer.load(load_dir)
    else:
        return BPETokenizer.load(load_dir)


if __name__ == "__main__":
    # 测试 CharTokenizer
    print("=" * 60)
    print("测试 CharTokenizer")
    print("=" * 60)

    sample_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
    ]

    ct = CharTokenizer()
    ct.build_vocab(sample_texts, save_dir="./data/tokenizer_test")

    encoded = ct.encode("Hello world!")
    decoded = ct.decode(encoded)
    print(f"\n编码: {encoded}")
    print(f"解码: {decoded}")
