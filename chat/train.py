import json
import os
import signal
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 中断保存
save_path = "saved_model/interrupted.keras"
def save_on_interrupt(signum, frame):
    print(f"\n\n训练被中断，保存到 {save_path} ...")
    model.save(save_path)
    print("保存完成！")
    sys.exit(0)
signal.signal(signal.SIGINT, save_on_interrupt)

with open("vocab.json", "r", encoding="utf-8") as f:
    vocab = json.load(f)

token_to_id = vocab
id_to_token = {v: k for k, v in vocab.items()}

VOCAB_SIZE = len(vocab)
PAD_ID = token_to_id["<|pad|>"]
EOS_ID = token_to_id["<|eos|>"]
UNK_ID = token_to_id["<|unk|>"]

SEQ_LEN = 128
BATCH_SIZE = 32

print(f"词表大小: {VOCAB_SIZE}")

def text_generator():
    with open("processed.txt", "r", encoding="utf-8") as f:
        all_tokens = []
        for line in f:
            tokens = line.strip().split()
            if len(tokens) > 0:
                all_tokens.extend(tokens)
                all_tokens.append("<|eos|>")
    
    stride = 64
    for i in range(0, len(all_tokens) - SEQ_LEN, stride):
        chunk = all_tokens[i:i + SEQ_LEN]
        ids = [token_to_id.get(t, UNK_ID) for t in chunk]
        
        if len(ids) < SEQ_LEN:
            ids += [PAD_ID] * (SEQ_LEN - len(ids))
        
        yield ids[:-1], ids[1:]

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
    .repeat()              # ← 关键：循环数据
    .prefetch(tf.data.AUTOTUNE))

# 步数 = 数据量 / batch_size（一轮的步数）
with open("processed.txt", "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
steps = total_lines // BATCH_SIZE
print(f"每轮步数: {steps}")

EMBED_DIM = 256
LSTM_UNITS = 512

inputs = keras.Input(shape=(SEQ_LEN - 1,))
x = layers.Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2))(x)
x = layers.LSTM(LSTM_UNITS, return_sequences=True, dropout=0.2)(x)
outputs = layers.TimeDistributed(layers.Dense(VOCAB_SIZE, activation="softmax"))(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    dataset,
    epochs=15,
    steps_per_epoch=steps,   # 每轮固定步数
    callbacks=[
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="loss"),
        keras.callbacks.ModelCheckpoint("saved_model/best_lm.keras", save_best_only=True, monitor="loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, monitor="loss", min_lr=1e-6)
    ]
)

os.makedirs("saved_model", exist_ok=True)
model.save("saved_model/phase1_lm.keras")
model.save("saved_model/phase1_lm.h5")

with open("saved_model/vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False)
print("完成！")

def generate(seed_text, max_new=30, temperature=0.8):
    seed_tokens = []
    for w in seed_text.lower().split():
        if w in vocab:
            seed_tokens.append(w)
        else:
            seed_tokens.extend(list(w))
    
    current = [token_to_id.get(t, UNK_ID) for t in seed_tokens]
    
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
    
    tokens = [id_to_token[i] for i in current]
    result = ""
    for t in tokens:
        if len(t) == 1 and t.isalpha():
            result += t
        else:
            result += " " + t + " "
    
    return " ".join(result.split())

print("\n--- 测试 ---")
print(generate("the cat sat on the"))
print(generate("in the morning i went to"))
print(generate("she opened the door and"))
