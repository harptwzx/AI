import re

# 读取原始文本
with open("merged_raw.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 1. 去掉页码标记：----------------------- Page 1-----------------------
text = re.sub(r'-+\s*Page\s*\d+\s*-+', '', text)

# 2. 去掉多余空行
text = re.sub(r'\n\s*\n', '\n', text)

# 3. 全部转小写
text = text.lower()

# 4. 按句子或段落分割成一行一行（模型训练用）
# 简单方案：按句号/问号/感叹号分割，每句一行
sentences = re.split(r'(?<=[.!?])\s+', text)
sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

# 保存
with open("merged.txt", "w", encoding="utf-8") as f:
    for sent in sentences:
        f.write(sent + "\n")

print(f"处理完成，共 {len(sentences)} 行")
print("前5行预览：")
for s in sentences[:5]:
    print(s[:100])
