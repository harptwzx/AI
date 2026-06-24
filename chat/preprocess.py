import json
import re

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_set = set(vocab.keys())

def tokenize_word(word):
    if word in vocab_set:
        return [word]
    else:
        return list(word)

with open("merged.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

output_lines = []

for raw_line in lines:
    line = raw_line.strip()
    
    # 跳过空行
    if not line:
        continue
    
    # 1. 去掉页码
    if re.match(r'-+\s*Page\s*\d+\s*-+', line):
        continue
    
    # 2. 跳过明显的非正文（封面、目录、页眉等）
    skip_keywords = [
        'book covers', 'complete collection', 'chapter', 'contents',
        'dedication', 'acknowledgements', 'copyright', 'isbn',
        'printed in', 'first edition', 'publisher', 'all rights reserved'
    ]
    if any(k in line.lower() for k in skip_keywords):
        continue
    
    # 3. 转小写
    line = line.lower()
    
    # 4. 分词处理
    words = line.split()
    new_tokens = []
    
    for w in words:
        # 分离尾部标点
        punct = ''
        while w and w[-1] in '.,!?;:"\'':
            punct = w[-1] + punct
            w = w[:-1]
        
        # 分离头部标点
        head_punct = ''
        while w and w[0] in '"\'':
            head_punct += w[0]
            w = w[1:]
        
        if not w:
            continue
        
        # 处理单词
        if head_punct:
            for p in head_punct:
                if p in vocab_set:
                    new_tokens.append(p)
        
        tokens = tokenize_word(w)
        new_tokens.extend(tokens)
        
        if punct:
            for p in punct:
                if p in vocab_set:
                    new_tokens.append(p)
    
    # 过滤太短的行
    if len(new_tokens) >= 5:
        output_lines.append(" ".join(new_tokens))

# 保存
with open("processed.txt", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"处理完成，共 {len(output_lines)} 行")
print("前10行预览：")
for line in output_lines[:10]:
    print(line[:100])
