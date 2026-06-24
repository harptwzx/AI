import json
import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# ============ 加载词表 ============
with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
PAD_ID = token_to_id["<|pad|>"]
BOS_ID = token_to_id["<|bos|>"]
EOS_ID = token_to_id["<|eos|>"]
UNK_ID = token_to_id["<|unk|>"]

SEQ_LEN = 128
BATCH_SIZE = 32

print(f"词表大小: {VOCAB_SIZE}")

# ============ 数据管道（小说文本：连续片段） ============
def text_generator():
    with open("merged.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # 把多行合并成长序列，再滑动窗口切分
    # 这样模型能学到跨句子的连贯性
    all_tokens = []
    for line in lines:
        tokens = line.strip().split()
        if len(tokens) > 0:
            all_tokens.extend(tokens)
            all_tokens.append("<|eos|>")  # 每句结束加标记
    
    # 滑动窗口：步长64，重叠一半，增加数据量
    stride = 64
    for i in range(0, len(all_tokens) - SEQ_LEN, stride):
        chunk = all_tokens[i:i + SEQ_LEN]
        ids = [token_to_id.get(t, UNK_ID) for t in chunk]
        
        # 填充
        if len(ids) < SEQ_LEN:
            ids += [PAD_ID] * (SEQ_LEN - len(ids))
        
        x = ids[:-1]
        y = ids[1:]
        yield x, y

dataset = tf.data.Dataset.from_generator(
    text_generator,
    output_signature=(
        tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32),
        tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32)
    )
)

dataset = (dataset
    .shuffle(buffer_size=200000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE))

# 估算步数
with open("merged.txt", "r", encoding="utf-8") as f:
    total_tokens = sum(len(line.split()) for line in f)
estimated_steps = (total_tokens // 64) // BATCH_SIZE
print(f"预估总token数: {total_tokens}, 预估步数: {estimated_steps}")

# ============ 模型 ============
EMBED_DIM = 256
LSTM_UNITS = 512

inputs = keras.Input(shape=(SEQ_LEN - 1,))
x = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inputs)

# 双向LSTM + 单向LSTM（捕捉前后文 + 生成方向）
x = layers.Bidirectional(
    layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)
)(x)
x = layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)(x)

outputs = layers.TimeDistributed(layers.Dense(VOCAB_SIZE, activation="softmax"))(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============ 训练：15 epoch ============
model.fit(
    dataset,
    epochs=15,
    steps_per_epoch=estimated_steps,
    callbacks=[
        keras.callbacks.EarlyStopping(
            patience=3,
            restore_best_weights=True,
            monitor="loss"
        ),
        keras.callbacks.ModelCheckpoint(
            "saved_model/best_hp.keras",
            save_best_only=True,
            monitor="loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=1,
            monitor="loss",
            min_lr=1e-5
        )
    ]
)

# ============ 保存 ============
os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/hp_phase1.keras")
model.save("saved_model/hp_phase1.h5")

with open("saved_model/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
print("第一阶段完成！")

# ============ 哈利波特风格生成测试 ============
def generate(seed_text, max_new=30, temperature=0.8):
    tokens = seed_text.lower().split() if isinstance(seed_text, str) else seed_text
    current = [token_to_id.get(t, UNK_ID) for t in tokens]
    
    for _ in range(max_new):
        padded = current[-(SEQ_LEN-1):]
        padded = [PAD_ID] * ((SEQ_LEN-1) - len(padded)) + padded
        
        pred = model.predict(np.array([padded]), verbose=0)
        logits = pred[0, -1, :] / temperature
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / np.sum(probs)
        next_id = np.random.choice(VOCAB_SIZE, p=probs)
        
        if next_id == EOS_ID:
            break
        current.append(next_id)
    
    result = " ".join(id_to_token[i] for i in current)
    # 清理特殊token显示
    result = result.replace("<|eos|>", "\n")
    result = result.replace("<|pad|>", "")
    return result.strip()

print("\n--- 哈利波特风格续写 ---")
print(generate("harry potter looked at"))
print(generate("the wand chose"))
print(generate("dumbledore said"))
print(generate("hogwarts castle stood"))
