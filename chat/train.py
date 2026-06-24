import json
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ============ 1. 加载词表 ============
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
PAD_ID = token_to_id["<|pad|>"]
EOS_ID = token_to_id["<|eos|>"]
UNK_ID = token_to_id["<|unk|>"]

SEQ_LEN = 128

print(f"词表大小: {VOCAB_SIZE}")

# ============ 2. tf.data 流式（优化版） ============
def text_generator():
    with open("merged.txt", "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            
            ids = [token_to_id.get(t, UNK_ID) for t in tokens]
            
            if len(ids) > SEQ_LEN:
                ids = ids[:SEQ_LEN]
            else:
                ids += [PAD_ID] * (SEQ_LEN - len(ids))
            
            yield ids[:-1], ids[1:]

# 用 tf.data 优化：cache 到内存（如果数据能放下）或文件，prefetch
dataset = tf.data.Dataset.from_generator(
    text_generator,
    output_signature=(
        tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32),
        tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32)
    )
)

BATCH_SIZE = 32  # CPU 用小 batch

dataset = (dataset
    .shuffle(buffer_size=50000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

# ============ 3. 模型（加 input_shape！） ============
EMBED_DIM = 128   # CPU 用小维度
LSTM_UNITS = 256  # CPU 用小单元

model = keras.Sequential([
    # 关键：指定 input_shape，模型才能构建
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True, input_shape=(SEQ_LEN - 1,)),
    layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2),
    layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2),
    layers.TimeDistributed(layers.Dense(VOCAB_SIZE, activation="softmax"))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()  # 现在应该显示参数了

# ============ 4. 训练 ============
# 目标需要 expand_dims 匹配 sparse_categorical_crossentropy
def prepare_batch(x, y):
    return x, tf.expand_dims(y, -1)

dataset = dataset.map(prepare_batch)

# 加 steps_per_epoch 避免 Unknown
# 先数一下总行数（快速估算）
print("正在统计语料行数...")
with open("merged.txt", "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
steps = total_lines // BATCH_SIZE
print(f"语料行数: {total_lines}, 每轮步数: {steps}")

model.fit(
    dataset,
    epochs=10,
    steps_per_epoch=steps,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1)
    ]
)

# ============ 5. 保存 ============
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/my_chat_model")
model.save("saved_model/my_chat_model.h5")
model.save("saved_model/my_chat_model.keras")

with open("saved_model/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
print("保存完成！")

# ============ 6. 生成 ============
def generate(seed_text, max_new=30, temperature=0.8):
    tokens = seed_text.split() if isinstance(seed_text, str) else seed_text
    current = [token_to_id.get(t, UNK_ID) for t in tokens]
    
    for _ in range(max_new):
        padded = current[-(SEQ_LEN-1):]
        padded = [PAD_ID] * ((SEQ_LEN-1) - len(padded)) + padded
        
        pred = model.predict(np.array([padded]), verbose=0)
        logits = pred[0, -1, :] / temperature
        probs = np.exp(logits - np.max(logits))  # 数值稳定
        probs = probs / np.sum(probs)
        next_id = np.random.choice(VOCAB_SIZE, p=probs)
        
        if next_id == EOS_ID:
            break
        current.append(next_id)
    
    return " ".join(id_to_token[i] for i in current)

print("\n--- 生成测试 ---")
print(generate("the cat sat"))
print(generate("i think that"))
print(generate("hello how are you"))
