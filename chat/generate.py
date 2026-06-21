import json
import tensorflow as tf

# 加载词汇表
with open("vocab.json", "r", encoding="utf-8") as f:
    word_to_id = json.load(f)
id_to_word = {int(v): k for k, v in word_to_id.items()}

# 加载SavedModel
model = tf.saved_model.load("saved_model/gpt_language_model")
infer = model.signatures["serving_default"]

def generate(prompt_text, max_tokens=30, temperature=0.8):
    words = prompt_text.lower().split()
    prompt_ids = [word_to_id.get(w, 3) for w in words]
    
    # 简单贪心生成（可扩展为采样）
    for _ in range(max_tokens):
        input_tensor = tf.constant([prompt_ids], dtype=tf.int32)
        output = infer(input_tensor)
        logits = output["logits"]
        
        next_token = int(tf.argmax(logits[0, -1, :]))
        prompt_ids.append(next_token)
    
    result = [id_to_word.get(i, "<|unk|>") for i in prompt_ids[len(words):]]
    return " ".join(result)

# 测试
print(generate("harry looked at", max_tokens=20))
