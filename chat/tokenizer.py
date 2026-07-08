"""
自实现 Tokenizer
================
支持三种模式:
1. WordTokenizer: 常见单词映射为 token，罕见词拆为字符（推荐）
2. CharTokenizer: 字符级，零训练时间
3. BPETokenizer: 字节对编码，训练慢
"""

import json
import collections
import os
import re
import time
from typing import List, Dict, Tuple, Set


# ============================================================
# 1. WordTokenizer — 常见单词 token + 罕见词字符拆分
# ============================================================
class WordTokenizer:
    """
    混合 Word-Char Tokenizer

    策略:
    - 高频单词 → 独立 token
    - 中频单词 → 拆为 subword（简单规则：前缀/后缀/词根）
    - 低频/罕见词 → 拆为字符

    自动统计训练集，生成 token 映射文件。
    """

    SPECIAL_TOKENS = {
        "<pad>": 0,
        "<unk>": 1,
        "<endoftext>": 2,
        "<user>": 3,
        "<assistant>": 4,
        "<word>": 5,      # 标记单词开始
        "<char>": 6,      # 标记字符开始
    }

    # 常见前缀/后缀，用于 subword 拆分
    COMMON_PREFIXES = [
        "un", "re", "in", "im", "dis", "en", "em", "non", "over", "mis",
        "sub", "pre", "inter", "fore", "de", "trans", "super", "semi",
        "anti", "mid", "under", "out", "up", "down", "off", "bi", "tri",
        "co", "auto", "micro", "macro", "multi", "poly", "mono",
    ]

    COMMON_SUFFIXES = [
        "ing", "ed", "er", "est", "ly", "tion", "sion", "ness", "ment",
        "ity", "ty", "ful", "less", "able", "ible", "al", "ial", "ic",
        "ical", "ous", "ious", "ive", "ative", "ize", "ise", "ify",
        "en", "ly", "ward", "wise", "dom", "ship", "hood", "ism",
        "ist", "er", "or", "ar", "ee", "ess", "let", "ling", "kin",
        "s", "es", "ies", "ied", "ied", "ied",
    ]

    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self.vocab_size = 0

        # 统计信息
        self.word_freq: Dict[str, int] = {}
        self.min_word_freq = 5  # 单词成为独立 token 的最小频率
        self.max_word_vocab = 15000  # 最大单词 token 数
        self.max_char_vocab = 150  # 最大字符 token 数

    def _tokenize_word(self, word: str) -> List[str]:
        """
        将单词拆分为 tokens

        策略:
        1. 如果单词在词表中 → 返回 [word]
        2. 尝试前缀拆分 → [prefix, rest]
        3. 尝试后缀拆分 → [stem, suffix]
        4. 拆为字符 → [c1, c2, ...]
        """
        word_lower = word.lower()

        # 1. 完整单词匹配
        if word_lower in self.word_to_id:
            return [word_lower]

        # 2. 尝试前缀拆分
        for prefix in sorted(self.COMMON_PREFIXES, key=len, reverse=True):
            if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 1:
                rest = word_lower[len(prefix):]
                if rest in self.word_to_id:
                    return [prefix, rest]
                # 递归拆分 rest
                rest_tokens = self._tokenize_word_simple(rest)
                if len(rest_tokens) < len(word_lower):
                    return [prefix] + rest_tokens

        # 3. 尝试后缀拆分
        for suffix in sorted(self.COMMON_SUFFIXES, key=len, reverse=True):
            if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 1:
                stem = word_lower[:-len(suffix)]
                if stem in self.word_to_id:
                    return [stem, suffix]
                stem_tokens = self._tokenize_word_simple(stem)
                if len(stem_tokens) < len(word_lower):
                    return stem_tokens + [suffix]

        # 4. 拆为字符
        return list(word_lower)

    def _tokenize_word_simple(self, word: str) -> List[str]:
        """简化版：只检查完整匹配，不匹配则拆字符"""
        if word in self.word_to_id:
            return [word]
        return list(word)

    def _extract_words(self, texts: List[str]) -> List[str]:
        """从文本中提取所有单词"""
        all_words = []
        for text in texts:
            # 提取单词（包括连字符单词）
            words = re.findall(r"[a-zA-Z]+(?:[-'][a-zA-Z]+)*", text)
            all_words.extend([w.lower() for w in words])
        return all_words

    def build_vocab(self, texts: List[str], save_dir: str = "./data/tokenizer"):
        """
        构建词表

        步骤:
        1. 统计所有单词频率
        2. 选择高频词作为独立 token
        3. 选择常见 subword（前缀/后缀）
        4. 收集所有字符
        5. 构建映射表
        """
        print("=" * 60)
        print("WordTokenizer 训练")
        print("=" * 60)

        start_time = time.time()

        # 1. 统计单词频率
        print("[Tokenizer] 统计单词频率...")
        all_words = self._extract_words(texts)
        self.word_freq = collections.Counter(all_words)
        print(f"[Tokenizer] 总单词数: {len(all_words):,}")
        print(f"[Tokenizer] 唯一单词数: {len(self.word_freq):,}")

        # 2. 选择高频词作为独立 token
        print(f"[Tokenizer] 选择频率 >= {self.min_word_freq} 的单词...")
        common_words = [
            word for word, freq in self.word_freq.most_common(self.max_word_vocab)
            if freq >= self.min_word_freq and len(word) >= 2
        ]
        print(f"[Tokenizer] 高频单词数: {len(common_words)}")

        # 3. 收集所有字符
        print("[Tokenizer] 收集字符...")
        all_chars = set()
        for text in texts:
            all_chars.update(text)

        # 过滤：只保留常见字符（ASCII + 部分标点）
        common_chars = []
        for char in sorted(all_chars):
            if ord(char) < 128 or char in '—–''""':
                common_chars.append(char)

        print(f"[Tokenizer] 字符数: {len(common_chars)}")

        # 4. 构建映射表
        # ID 分配:
        # 0-99: special tokens
        # 100-999: 字符
        # 1000+: 单词

        self.word_to_id = dict(self.SPECIAL_TOKENS)
        self.char_to_id = dict(self.SPECIAL_TOKENS)
        next_id = 100

        # 先分配字符
        for char in sorted(common_chars):
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                self.word_to_id[char] = next_id  # 字符也在 word 表中
                next_id += 1

        char_end = next_id
        print(f"[Tokenizer] 字符 token 范围: 100-{char_end-1}")

        # 再分配单词
        for word in common_words:
            if word not in self.word_to_id:
                self.word_to_id[word] = next_id
                next_id += 1

        self.vocab_size = next_id
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}

        elapsed = time.time() - start_time
        print(f"[Tokenizer] 词表构建完成！耗时: {elapsed:.2f}s")
        print(f"[Tokenizer] vocab_size: {self.vocab_size}")
        print(f"[Tokenizer]   - 特殊 token: {len(self.SPECIAL_TOKENS)}")
        print(f"[Tokenizer]   - 字符 token: {char_end - 100}")
        print(f"[Tokenizer]   - 单词 token: {self.vocab_size - char_end}")

        # 保存
        self.save(save_dir, texts)
        return self

    def save(self, save_dir: str, sample_texts: List[str] = None):
        """保存 tokenizer 和 token 映射文件"""
        os.makedirs(save_dir, exist_ok=True)

        # 保存配置
        data = {
            "type": "word",
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "id_to_word": self.id_to_word,
            "char_to_id": self.char_to_id,
            "id_to_char": self.id_to_char,
            "special_tokens": self.SPECIAL_TOKENS,
            "min_word_freq": self.min_word_freq,
            "word_freq_sample": dict(list(self.word_freq.most_common(100))),
        }

        with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # 保存可读 vocab 表
        with open(os.path.join(save_dir, "vocab_readable.txt"), "w", encoding="utf-8") as f:
            f.write(f"# WordTokenizer Vocab\n")
            f.write(f"# vocab_size: {self.vocab_size}\n")
            f.write(f"# type: word\n")
            f.write(f"# min_word_freq: {self.min_word_freq}\n")
            f.write("=" * 60 + "\n")

            # Special tokens
            f.write(f"\n## Special Tokens ({len(self.SPECIAL_TOKENS)})\n")
            f.write("-" * 40 + "\n")
            for name, tid in self.SPECIAL_TOKENS.items():
                f.write(f"{tid:<8} {name:<20} SPECIAL\n")

            # Characters
            char_ids = sorted([k for k in self.id_to_word.keys() if 100 <= k < 100 + len(self.char_to_id) - len(self.SPECIAL_TOKENS)])
            f.write(f"\n## Character Tokens ({len(char_ids)})\n")
            f.write("-" * 40 + "\n")
            for tid in char_ids[:50]:
                word = self.id_to_word[tid]
                f.write(f"{tid:<8} {repr(word):<20} CHAR\n")
            if len(char_ids) > 50:
                f.write(f"... 还有 {len(char_ids) - 50} 个字符\n")

            # Words
            word_ids = sorted([k for k in self.id_to_word.keys() if k >= max(char_ids) + 1 if char_ids else 1000])
            f.write(f"\n## Word Tokens (top 200, total {len(word_ids)})\n")
            f.write("-" * 40 + "\n")

            # 按频率排序显示
            sorted_words = sorted(
                [(self.id_to_word[tid], tid) for tid in word_ids],
                key=lambda x: self.word_freq.get(x[0], 0),
                reverse=True
            )
            for word, tid in sorted_words[:200]:
                freq = self.word_freq.get(word, 0)
                f.write(f"{tid:<8} {word:<20} WORD (freq={freq})\n")
            if len(sorted_words) > 200:
                f.write(f"... 还有 {len(sorted_words) - 200} 个单词\n")

        # 保存纯单词列表（方便查看）
        with open(os.path.join(save_dir, "word_list.txt"), "w", encoding="utf-8") as f:
            f.write(f"# Word Token List\n")
            f.write(f"# Total words: {len(word_ids)}\n")
            f.write("-" * 40 + "\n")
            for word, tid in sorted_words:
                freq = self.word_freq.get(word, 0)
                f.write(f"{word:<20} id={tid:<8} freq={freq}\n")

        # 测试并保存测试结果
        if sample_texts:
            with open(os.path.join(save_dir, "test_results.txt"), "w", encoding="utf-8") as f:
                f.write("# Tokenizer Test Results\n")
                f.write("=" * 60 + "\n\n")

                test_sentences = sample_texts[:10]
                for text in test_sentences:
                    encoded = self.encode(text)
                    decoded = self.decode(encoded)
                    tokens = [self.id_to_word.get(i, "<unk>") for i in encoded]

                    f.write(f"原文: {text}\n")
                    f.write(f"编码: {encoded}\n")
                    f.write(f"Tokens: {tokens}\n")
                    f.write(f"解码: {decoded}\n")
                    f.write(f"压缩率: {len(text)}/{len(encoded)} = {len(text)/len(encoded):.2f}x\n")
                    f.write("-" * 40 + "\n\n")

        print(f"[Tokenizer] 已保存到 {save_dir}:")
        print(f"  - tokenizer_config.json")
        print(f"  - vocab_readable.txt")
        print(f"  - word_list.txt")
        print(f"  - test_results.txt")

    @classmethod
    def load(cls, load_dir: str):
        """加载 tokenizer"""
        config_path = os.path.join(load_dir, "tokenizer_config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        instance = cls()
        instance.word_to_id = data["word_to_id"]
        instance.id_to_word = {int(k): v for k, v in data["id_to_word"].items()}
        instance.char_to_id = data["char_to_id"]
        instance.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        instance.vocab_size = data["vocab_size"]
        instance.min_word_freq = data.get("min_word_freq", 5)
        instance.word_freq = data.get("word_freq_sample", {})

        print(f"[WordTokenizer] 已从 {load_dir} 加载")
        print(f"[WordTokenizer] vocab_size={instance.vocab_size}")
        return instance

    def encode(self, text: str) -> List[int]:
        """文本 → token ids"""
        tokens = self._tokenize_text(text)
        return [self.word_to_id.get(t, self.SPECIAL_TOKENS["<unk>"]) for t in tokens]

    def _tokenize_text(self, text: str) -> List[str]:
        """将文本拆分为 token 列表"""
        tokens = []
        i = 0
        while i < len(text):
            char = text[i]

            # 如果是字母，尝试匹配最长的单词
            if char.isalpha():
                # 尝试匹配最长单词
                matched = False
                for j in range(min(i + 20, len(text)), i, -1):
                    word = text[i:j].lower()
                    if word in self.word_to_id and len(word) >= 2:
                        tokens.append(word)
                        i = j
                        matched = True
                        break

                if not matched:
                    # 尝试前缀/后缀拆分
                    word = text[i:].lower()
                    sub_tokens = self._tokenize_word(word)
                    # 只取匹配上的部分
                    consumed = 0
                    for st in sub_tokens:
                        if consumed >= len(word):
                            break
                        tokens.append(st)
                        consumed += len(st)
                    i += consumed if consumed > 0 else 1
            else:
                # 非字母字符，单独处理
                char_lower = char.lower()
                if char_lower in self.word_to_id:
                    tokens.append(char_lower)
                else:
                    tokens.append("<unk>")
                i += 1

        return tokens

    def decode(self, ids: List[int]) -> str:
        """token ids → 文本"""
        result = []
        for i in ids:
            token = self.id_to_word.get(i, "<unk>")
            if token in self.SPECIAL_TOKENS:
                continue
            result.append(token)
        return "".join(result)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [self.encode(t) for t in texts]

    def decode_batch(self, id_lists: List[List[int]]) -> List[str]:
        return [self.decode(ids) for ids in id_lists]


# ============================================================
# 2. CharTokenizer — 字符级（保留）
# ============================================================
class CharTokenizer:
    """字符级 Tokenizer"""

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
        print("[CharTokenizer] 构建字符表...")

        all_chars = set()
        for text in texts:
            all_chars.update(text)

        sorted_chars = sorted(all_chars)

        self.char_to_id = dict(self.SPECIAL_TOKENS)
        next_id = 100

        for char in sorted_chars:
            if char not in self.char_to_id:
                self.char_to_id[char] = next_id
                next_id += 1

        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = next_id

        print(f"[CharTokenizer] 字符表大小: {self.vocab_size}")

        self.save(save_dir)
        return self

    def save(self, save_dir: str):
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

        with open(os.path.join(save_dir, "vocab_readable.txt"), "w", encoding="utf-8") as f:
            f.write(f"# CharTokenizer Vocab\n")
            f.write(f"# vocab_size: {self.vocab_size}\n")
            f.write("-" * 50 + "\n")
            for token_id in sorted(self.id_to_char.keys()):
                char = self.id_to_char[token_id]
                token_type = "SPECIAL" if token_id in self.SPECIAL_TOKENS.values() else "CHAR"
                f.write(f"{token_id:<8} {repr(char):<20} {token_type}\n")

        print(f"[CharTokenizer] 已保存到 {save_dir}")

    @classmethod
    def load(cls, load_dir: str):
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
        return [self.char_to_id.get(c, self.SPECIAL_TOKENS["<unk>"]) for c in text]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.id_to_char.get(i, "<unk>") for i in ids)


# ============================================================
# 3. BPETokenizer — 字节对编码（保留）
# ============================================================
class BPETokenizer:
    """字节对编码 Tokenizer（训练慢，保留供参考）"""

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
        print(f"[BPETokenizer] 开始训练 BPE，目标 vocab_size={self.vocab_size}")

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

        os.makedirs(save_dir, exist_ok=True)
        self.save(save_dir)
        return ids

    def save(self, save_dir: str):
        vocab_path = os.path.join(save_dir, "vocab.json")
        with open(vocab_path, "w", encoding="utf-8") as f:
            vocab_str = {str(k): v for k, v in self.vocab.items()}
            json.dump(vocab_str, f, ensure_ascii=False, indent=2)

        merges_path = os.path.join(save_dir, "merges.txt")
        with open(merges_path, "w", encoding="utf-8") as f:
            f.write("# BPE merges\n")
            for pair in self.merges:
                f.write(f"{pair[0]} {pair[1]}\n")

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
# 4. 统一加载接口
# ============================================================
def load_tokenizer(load_dir: str = "./data/tokenizer"):
    """自动检测并加载 tokenizer（支持 word/char/bpe）"""
    config_path = os.path.join(load_dir, "tokenizer_config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到 tokenizer 配置文件: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    tokenizer_type = config.get("type", "bpe")

    if tokenizer_type == "word":
        return WordTokenizer.load(load_dir)
    elif tokenizer_type == "char":
        return CharTokenizer.load(load_dir)
    else:
        return BPETokenizer.load(load_dir)


if __name__ == "__main__":
    print("=" * 60)
    print("测试 WordTokenizer")
    print("=" * 60)

    sample_texts = [
        "Hello, world! This is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is fascinating.",
        "I love learning English every day.",
        "Unhappiness is not a good feeling.",
        "Running quickly through the forest.",
    ]

    wt = WordTokenizer()
    wt.build_vocab(sample_texts, save_dir="./data/tokenizer_test")

    for text in sample_texts[:3]:
        encoded = wt.encode(text)
        decoded = wt.decode(encoded)
        tokens = [wt.id_to_word.get(i, "<unk>") for i in encoded]
        print(f"\n原文: {text}")
        print(f"编码: {encoded}")
        print(f"Tokens: {tokens}")
        print(f"解码: {decoded}")
        print(f"压缩率: {len(text)} -> {len(encoded)} = {len(text)/len(encoded):.2f}x")
