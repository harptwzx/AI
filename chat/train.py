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

SEQ_LEN = 128  # 总长度，输入127，预测第128个

print(f"词表大小: {VOCAB_SIZE}")

# ============ 2. tf.data 流式数据管道 ============
def text_generator():
    with open("merged.txt", "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            
            ids = [token_to_id.get(t, UNK_ID) for t in tokens]
            
            # 截断或填充到 SEQ_LEN
            if len(ids) > SEQ_LEN:
                ids = ids[:SEQ_LEN]
            else:
                ids += [PAD_ID] * (SEQ_LEN - len(ids))
            
            # 输入[:-1] (127个), 目标[1:] (127个，每个位置预测下一个)
            yield ids[:-1], ids[1:]

dataset = tf.data.Dataset.from_generator(
    text_generator,
    output_signature=(
        tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32),
        tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32)
    )
)

# 数据管道：打乱 → 分batch → 预取
BATCH_SIZE = 64
dataset = (dataset
    .shuffle(buffer_size=100000)  # 根据内存调，越大shuffle越均匀
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

# 查看一个batch
for x, y in dataset.take(1):
    print(f"Batch形状: x={x.shape}, y={y.shape}")
    break

# ============ 3. 模型 ============
EMBED_DIM = 256
LSTM_UNITS = 512

model = keras.Sequential([
    layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True),
    layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2),
    layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2),
    layers.TimeDistributed(layers.Dense(VOCAB_SIZE, activation="softmax"))
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============ 4. 训练 ============
# 计算步数（可选，用于显示进度）
# 先数行数，或直接用 epochs，让 tf.data 自己跑完
model.fit(dataset, epochs=10, callbacks=[
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
])

# ============ 5. 保存 ============
os.makedirs("saved_model", exist_ok=True)

# SavedModel（推荐，TensorFlow Serving 用）
model.save("saved_model/my_chat_model")
print("✓ SavedModel")

# HDF5
model.save("saved_model/my_chat_model.h5")
print("✓ HDF5 (.h5)")

# Keras 原生
model.save("saved_model/my_chat_model.keras")
print("✓ Keras (.keras)")

# 词表
with open("saved_model/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
print("✓ vocab.json")

# ============ 6. 生成测试 ============
def generate(seed_text, max_new=30, temperature=0.8):
    tokens = seed_text.split() if isinstance(seed_text, str) else seed_text
    current = [token_to_id.get(t, UNK_ID) for t in tokens]
    
    for _ in range(max_new):
        # 取最后127个
        padded = current[-(SEQ_LEN-1):]
        padded = [PAD_ID] * ((SEQ_LEN-1) - len(padded)) + padded
        
        pred = model.predict(np.array([padded]), verbose=0)
        logits = pred[0, -1, :] / temperature
        probs = np.exp(logits) / np.sum(np.exp(logits))
        next_id = np.random.choice(VOCAB_SIZE, p=probs)
        
        if next_id == EOS_ID:
            break
        current.append(next_id)
    
    return " ".join(id_to_token[i] for i in current)

print("\n--- 生成测试 ---")
print(generate("the cat sat"))
print(generate("i think that"))
print(generate("hello how are you"))

