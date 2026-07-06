import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

# ============ 加载词表 ============
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
PAD_ID = token_to_id["<|pad|>"]
EOS_ID = token_to_id["<|eos|>"]
UNK_ID = token_to_id["<|unk|>"]
SEQ_LEN = 128

# ============ 加载模型 ============
print("加载模型...")
model = keras.models.load_model("saved_model/phase1_lm_v2.keras")
print("模型加载完成！")

# ============ 生成函数 ============
def generate(seed_text, max_new=30, temperature=0.8):
    # 将输入文本转为 token
    seed_tokens = []
    for w in seed_text.lower().split():
        if w in vocab:
            seed_tokens.append(w)
        else:
            # 未登录词拆成字母
            seed_tokens.extend(list(w))

    current = [token_to_id.get(t, UNK_ID) for t in seed_tokens]

    # 逐个生成
    for _ in range(max_new):
        # 填充到模型输入长度
        padded = current[-(SEQ_LEN-1):]
        padded = [PAD_ID] * ((SEQ_LEN-1) - len(padded)) + padded

        # 预测下一个 token
        pred = model.predict(np.array([padded]), verbose=0)
        logits = pred[0, -1, :] / temperature
        logits = logits - np.max(logits)  # 数值稳定
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        next_id = np.random.choice(VOCAB_SIZE, p=probs)

        if next_id == EOS_ID:
            break
        current.append(next_id)

    # 将 ID 转回文本，合并字母
    tokens = [id_to_token[i] for i in current]
    result = ""
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            result += t  # 字母直接连
        else:
            result += " " + t + " "  # 其他 token 加空格

    return " ".join(result.split())

# ============ 交互式生成 ============
print("\n========== 文本生成 ==========")
print("输入提示词，模型会续写。输入 'quit' 退出。\n")

while True:
    user_input = input("提示词: ").strip()
    if user_input.lower() == "quit":
        break
    
    if not user_input:
        continue

    print("\n--- 生成结果 ---")
    for temp in [0.5, 0.8, 1.2]:
        print(f"\n[temperature={temp}]")
        result = generate(user_input, max_new=40, temperature=temp)
        print(result)
    print("-" * 50)
