import json
import re

with open("processed.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

punctuation = set('.,!?;:\'\"()-')

def fix_punctuation(tokens):
    """修复标点粘连：didnt -> didn ' t, youd -> you ' d"""
    result = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        
        # 处理 n't 类缩写
        if t.endswith("n") and i + 1 < len(tokens) and tokens[i + 1] == "t":
            # 可能是 "didn" + "t" -> "didn" + "'" + "t"
            # 但这里已经拆开了，直接加 '
            result.append(t)
            result.append("'")
            result.append("t")
            i += 2
            continue
        
        # 处理 's, 'd, 'll, 're, 've, 'm
        if i + 1 < len(tokens) and tokens[i + 1] in ('s', 'd', 't', 'm', 're', 'll', 've'):
            if "'" not in t and len(t) > 1:
                # 如 "harry" + "s" -> 不是缩写，正常处理
                pass
            elif len(t) == 1 and t in ('i', 'you', 'he', 'she', 'it', 'we', 'they', 'who', 'what', 'that', 'there', 'here'):
                # 代词 + s/d/t -> 加 '
                result.append(t)
                result.append("'")
                result.append(tokens[i + 1])
                i += 2
                continue
        
        # 处理 a + firm -> a firm（简单空格修复）
        if t == "a" and i + 1 < len(tokens) and tokens[i + 1] in ('firm', 'few', 'little', 'lot', 'bit', 'couple', 'dozen', 'hundred', 'thousand', 'million'):
            result.append(t)
            result.append(tokens[i + 1])
            i += 2
            continue
        
        result.append(t)
        i += 1
    
    return result

# 更简单的方案：直接字符串替换修复常见粘连
def fix_line(line):
    # 常见缩写修复
    replacements = [
        (r'\bdon t\b', "don ' t"),
        (r'\bdidn t\b', "didn ' t"),
        (r'\bcouldn t\b', "couldn ' t"),
        (r'\bwouldn t\b', "wouldn ' t"),
        (r'\bshouldn t\b', "shouldn ' t"),
        (r'\bwasn t\b', "wasn ' t"),
        (r'\bweren t\b', "weren ' t"),
        (r'\bhaven t\b', "haven ' t"),
        (r'\bhasn t\b', "hasn ' t"),
        (r'\bhadn t\b', "hadn ' t"),
        (r'\bisn t\b', "isn ' t"),
        (r'\baren t\b', "aren ' t"),
        (r'\bwon t\b', "won ' t"),
        (r'\bcan t\b', "can ' t"),
        (r'\bi m\b', "i ' m"),
        (r'\bi ll\b', "i ' ll"),
        (r'\bi d\b', "i ' d"),
        (r'\bi ve\b', "i ' ve"),
        (r'\byou re\b', "you ' re"),
        (r'\byou ll\b', "you ' ll"),
        (r'\byou d\b', "you ' d"),
        (r'\byou ve\b', "you ' ve"),
        (r'\bhe s\b', "he ' s"),
        (r'\bhe ll\b', "he ' ll"),
        (r'\bhe d\b', "he ' d"),
        (r'\bshe s\b', "she ' s"),
        (r'\bshe ll\b', "she ' ll"),
        (r'\bshe d\b', "she ' d"),
        (r'\bit s\b', "it ' s"),
        (r'\bit ll\b', "it ' ll"),
        (r'\bit d\b', "it ' d"),
        (r'\bwe re\b', "we ' re"),
        (r'\bwe ll\b', "we ' ll"),
        (r'\bwe d\b', "we ' d"),
        (r'\bwe ve\b', "we ' ve"),
        (r'\bthey re\b', "they ' re"),
        (r'\bthey ll\b', "they ' ll"),
        (r'\bthey d\b', "they ' d"),
        (r'\bthey ve\b', "they ' ve"),
        (r'\bthat s\b', "that ' s"),
        (r'\bthat ll\b', "that ' ll"),
        (r'\bthat d\b', "that ' d"),
        (r'\bthere s\b', "there ' s"),
        (r'\bthere ll\b', "there ' ll"),
        (r'\bthere d\b', "there ' d"),
        (r'\bhere s\b', "here ' s"),
        (r'\bwho s\b', "who ' s"),
        (r'\bwho ll\b', "who ' ll"),
        (r'\bwho d\b', "who ' d"),
        (r'\bwhat s\b', "what ' s"),
        (r'\bwhat ll\b', "what ' ll"),
        (r'\bwhat d\b', "what ' d"),
        (r'\bhow s\b', "how ' s"),
        (r'\bhow ll\b', "how ' ll"),
        (r'\bhow d\b', "how ' d"),
        (r'\bwhere s\b', "where ' s"),
        (r'\bwhere ll\b', "where ' ll"),
        (r'\bwhere d\b', "where ' d"),
        (r'\bwhen s\b', "when ' s"),
        (r'\bwhen ll\b', "when ' ll"),
        (r'\bwhen d\b', "when ' d"),
        (r'\bwhy s\b', "why ' s"),
        (r'\bwhy ll\b', "why ' ll"),
        (r'\bwhy d\b', "why ' d"),
        # 所有格
        (r'\bsorcerer s\b', "sorcerer ' s"),
        (r'\bchamber s\b', "chamber ' s"),
        (r'\bprisoner s\b', "prisoner ' s"),
        (r'\bgoblet s\b', "goblet ' s"),
        (r'\border s\b', "order ' s"),
        (r'\bphoenix s\b', "phoenix ' s"),
        (r'\bharry s\b', "harry ' s"),
        (r'\bpotter s\b', "potter ' s"),
        (r'\bdursley s\b', "dursley ' s"),
        (r'\bmr s\b', "mr ' s"),
        (r'\bmrs s\b', "mrs ' s"),
        # 其他常见粘连
        (r'\bafirm\b', "a firm"),
        (r'\bafew\b', "a few"),
        (r'\balot\b', "a lot"),
        (r'\balittle\b', "a little"),
        (r'\babit\b', "a bit"),
    ]
    
    for pattern, replacement in replacements:
        line = re.sub(pattern, replacement, line)
    
    return line

output_lines = []
for line in lines:
    line = line.strip()
    if not line:
        continue
    line = fix_line(line)
    output_lines.append(line)

with open("processed.txt", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

# 重建 vocab.json
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

print(f"词表大小: {len(vocab)}")
print(f"高频词数量: {len(frequent_words)}")
print("前30个高频词:", frequent_words[:30])

with open("vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)

print("\n前15行预览：")
for line in output_lines[:15]:
    print(line[:120])
