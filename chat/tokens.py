import json
import re

# ============ 对已生成的 processed.txt 做后处理 ============
with open("processed.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

punctuation = set('.,!?;:\'\"()-')

def merge_spaced_sequence(tokens):
    """合并 token 列表中的连续单字母序列"""
    result = []
    i = 0
    while i < len(tokens):
        # 找连续单字母（至少3个）
        if len(tokens[i]) == 1 and tokens[i].isalpha() and tokens[i] not in punctuation:
            j = i
            while j < len(tokens) and len(tokens[j]) == 1 and tokens[j].isalpha() and tokens[j] not in punctuation:
                j += 1
            if j - i >= 3:
                merged = ''.join(tokens[i:j])
                result.append(merged)
                i = j
            else:
                result.append(tokens[i])
                i += 1
        else:
            result.append(tokens[i])
            i += 1
    return result

output_lines = []
for line in lines:
    tokens = line.strip().split()
    merged = merge_spaced_sequence(tokens)
    output_lines.append(" ".join(merged))

with open("processed.txt", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

# 验证
print("前15行预览：")
for line in output_lines[:15]:
    print(line[:120])

# 检查是否有残留的单字母序列
spaced_count = sum(1 for line in output_lines for t in line.split() if len(t) == 1 and t.isalpha())
print(f"\n残留单字母 token 数: {spaced_count}")

# ============ 重建 vocab.json ============
token_counter = {}
for line in output_lines:
    for token in line.split():
        token_counter[token] = token_counter.get(token, 0) + 1

special_tokens = ['<|pad|>', '<|bos|>', '<|eos|>', '<|unk|>', '<|user|>', '<|bot|>']

frequent_words = [word for word, count in token_counter.items() 
                  if count >= 3 and word not in punctuation and len(word) > 1]
frequent_words = sorted(set(frequent_words))

used_punct = sorted([p for p in punctuation if token_counter.get(p, 0) > 0])

all_chars = set()
for word in token_counter:
    if word not in frequent_words and word not in punctuation:
        all_chars.update(list(word))
all_chars = sorted(all_chars)

vocab = {}
idx = 0
for t in special_tokens:
    vocab[t] = idx
    idx += 1
for p in used_punct:
    vocab[p] = idx
    idx += 1
for w in frequent_words:
    vocab[w] = idx
    idx += 1
for c in all_chars:
    vocab[c] = idx
    idx += 1

print(f"\n词表大小: {len(vocab)}")
print(f"高频词数量: {len(frequent_words)}")
print("前30个高频词:", frequent_words[:30])

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("\n完成！")
