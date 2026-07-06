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
in_book_content = False  # 标记是否进入正文

for raw_line in lines:
    line = raw_line.strip()
    
    if not line:
        continue
    
    # 去掉页码
    if re.match(r'-+\s*Page\s*\d+\s*-+', line):
        continue
    
    lower = line.lower()
    
    # 更强的跳过规则
    skip_patterns = [
        'complete', 'collection', 'book covers', 'compiled', 'ocr', 'scanned',
        'wetaskiwin', 'omnipage', 'dedication', 'acknowledgements', 'copyright',
        'isbn', 'printed in', 'first edition', 'publisher', 'all rights reserved',
        'contents', 'chapter one', 'chapter 1', 'prologue', 'epilogue',
        'j k rowling', 'j.k. rowling', 'scholastic', 'bloomsbury'
    ]
    if any(p in lower for p in skip_patterns):
        continue
    
    # 跳过纯大写的行（通常是标题）
    if line.isupper() and len(line) > 3:
        continue
    
    # 跳过数字占比过高的行（页码、编号）
    digits = sum(c.isdigit() for c in line)
    if digits / len(line) > 0.3:
        continue
    
    # 检测到第一章开始，标记进入正文
    if 'mr. and mrs.' in lower or 'mr and mrs' in lower or "the boy who lived" in lower:
        in_book_content = True
    
    # 如果没检测到开始标记，但行里有正常叙事内容，也接受
    # 主要过滤掉明显的非正文
    
    line = line.lower()
    words = line.split()
    new_tokens = []
    
    for w in words:
        punct = ''
        while w and w[-1] in '.,!?;:"\'':
            punct = w[-1] + punct
            w = w[:-1]
        
        head_punct = ''
        while w and w[0] in '"\'':
            head_punct += w[0]
            w = w[1:]
        
        if not w:
            continue
        
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
    
    if len(new_tokens) >= 5:
        output_lines.append(" ".join(new_tokens))

with open("processed.txt", "w", encoding="utf-8") as f:
    for line in output_lines:
        f.write(line + "\n")

print(f"处理完成，共 {len(output_lines)} 行")
print("前15行预览：")
for line in output_lines[:15]:
    print(line[:100])
