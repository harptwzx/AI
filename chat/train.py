import json
import os
import signal
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# 确保保存目录存在
os.makedirs("saved_model", exist_ok=True)

# 中断保存
save_path = "saved_model/interrupted_v2.keras"
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
    .repeat()
    .prefetch(tf.data.AUTOTUNE))

with open("processed.txt", "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)
steps = total_lines // BATCH_SIZE
print(f"每轮步数: {steps}")

# ============ 加载或创建模型 ============
MODEL_PATH = "saved_model/phase1_lm.keras"

if os.path.isfile(MODEL_PATH):
    print("\n========== 加载已有模型 ==========")
    model = keras.models.load_model(MODEL_PATH)
    model.summary()
    print(f"总参数量: {model.count_params():,}")
else:
    print("\n========== 创建新模型 ==========")
    
    # 构建 Transformer 语言模型
    def build_model(vocab_size, seq_len, embed_dim=256, num_heads=8, ff_dim=512, num_layers=4):
        inputs = layers.Input(shape=(seq_len - 1,))
        
        # 词嵌入 + 位置编码
        x = layers.Embedding(vocab_size, embed_dim)(inputs)
        pos_encoding = layers.Embedding(seq_len - 1, embed_dim)(tf.range(seq_len - 1))
        x = x + pos_encoding
        
        # Transformer 编码器层
        for _ in range(num_layers):
            # 多头自注意力
            attn_output = layers.MultiHeadAttention(
                num_heads=num_heads, 
                key_dim=embed_dim // num_heads
            )(x, x, use_causal_mask=True)
            x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
            
            # 前馈网络
            ff_output = layers.Dense(ff_dim, activation="relu")(x)
            ff_output = layers.Dense(embed_dim)(ff_output)
            x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        
        # 输出层
        outputs = layers.Dense(vocab_size, activation="softmax")(x)
        
        model = keras.Model(inputs, outputs)
        return model
    
    model = build_model(VOCAB_SIZE, SEQ_LEN)
    model.summary()
    print(f"总参数量: {model.count_params():,}")
    
    # 首次训练（phase 1）
    print("\n========== 开始首次训练 ==========")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    model.fit(
        dataset,
        epochs=10,
        steps_per_epoch=steps,
        callbacks=[
            keras.callbacks.ModelCheckpoint(
                "saved_model/phase1_lm.keras", 
                save_best_only=True, 
                monitor="loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=2, 
                monitor="loss", 
                min_lr=1e-6
            )
        ]
    )
    
    print("\n首次训练完成，模型已保存！")

# ============ 继续训练 ============
print("\n========== 开始继续训练 ==========")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    dataset,
    epochs=20,
    steps_per_epoch=steps,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="loss"),
        keras.callbacks.ModelCheckpoint("saved_model/best_continue.keras", save_best_only=True, monitor="loss"),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, monitor="loss", min_lr=1e-6)
    ]
)

model.save("saved_model/phase1_lm_v2.keras")
model.save("saved_model/phase1_lm_v2.h5")

print("\n继续训练完成！")

# ============ 生成测试 ============
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

print("\n--- 生成测试 ---")
print(generate("the cat sat on the"))
print(generate("in the morning i went to"))
print(generate("she opened the door and"))
