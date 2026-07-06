import json
import os
import signal
import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

os.makedirs("saved_model", exist_ok=True)

best_weights = None
best_val_loss = float('inf')
current_epoch = 0

save_path = "saved_model/interrupted_best.keras"

def save_on_interrupt(signum, frame):
    global best_weights, best_val_loss, current_epoch
    print(f"\n\n⚠️  训练被中断 (Epoch {current_epoch})")
    if best_weights is not None:
        print(f"保存历史最佳模型 (val_loss={best_val_loss:.4f})...")
        model.set_weights(best_weights)
        model.save(save_path)
        print("✅ 完成")
    else:
        model.save(save_path)
        print("✅ 已保存当前模型")
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

def text_generator(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
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

with open("processed.txt", "r", encoding="utf-8") as f:
    all_lines = f.readlines()

np.random.seed(42)
np.random.shuffle(all_lines)

split_idx = int(len(all_lines) * 0.9)
train_lines = all_lines[:split_idx]
val_lines = all_lines[split_idx:]

with open("processed_train.tmp", "w", encoding="utf-8") as f:
    f.writelines(train_lines)
with open("processed_val.tmp", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

def make_dataset(file_path, shuffle=False):
    ds = tf.data.Dataset.from_generator(
        lambda: text_generator(file_path),
        output_signature=(
            tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32),
            tf.TensorSpec(shape=(SEQ_LEN - 1,), dtype=tf.int32)
        )
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=200000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset("processed_train.tmp", shuffle=True)
val_ds = make_dataset("processed_val.tmp", shuffle=False)

train_steps = max(1, len(train_lines) // BATCH_SIZE)
val_steps = max(1, len(val_lines) // BATCH_SIZE)
print(f"训练步数/epoch: {train_steps}, 验证步数/epoch: {val_steps}")

# ============ 标签平滑损失 ============
def smoothed_loss(y_true, y_pred, smoothing=0.1):
    num_classes = tf.cast(tf.shape(y_pred)[-1], y_pred.dtype)
    y_true = tf.cast(y_true, tf.int32)
    one_hot = tf.one_hot(y_true, num_classes)
    smoothed = one_hot * (1.0 - smoothing) + smoothing / num_classes
    return tf.keras.losses.categorical_crossentropy(smoothed, y_pred)

# ============ 保持原结构的模型，只加 Dropout ============
def build_model(vocab_size, seq_len, embed_dim=256, num_heads=8, ff_dim=512, num_layers=4, dropout_rate=0.1):
    inputs = layers.Input(shape=(seq_len - 1,))
    
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    pos_encoding = layers.Embedding(seq_len - 1, embed_dim)(tf.range(seq_len - 1))
    x = x + pos_encoding
    x = layers.Dropout(dropout_rate)(x)
    
    for _ in range(num_layers):
        attn_output = layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=embed_dim // num_heads,
            dropout=dropout_rate
        )(x, x, use_causal_mask=True)
        x = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
        x = layers.Dropout(dropout_rate)(x)
        
        ff_output = layers.Dense(ff_dim, activation="relu")(x)
        ff_output = layers.Dropout(dropout_rate)(ff_output)
        ff_output = layers.Dense(embed_dim)(ff_output)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ff_output)
        x = layers.Dropout(dropout_rate)(x)
    
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(vocab_size, activation="softmax")(x)
    
    return keras.Model(inputs, outputs)

# ============ 加载或创建 ============
MODEL_PATH = "saved_model/phase1_lm.keras"

if os.path.isfile(MODEL_PATH):
    print("\n========== 加载已有模型 ==========")
    model = keras.models.load_model(MODEL_PATH)
    model.summary()
    print(f"总参数量: {model.count_params():,}")
    best_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in model.get_weights()]
else:
    print("\n========== 创建新模型 ==========")
    model = build_model(VOCAB_SIZE, SEQ_LEN)
    model.summary()
    print(f"总参数量: {model.count_params():,}")

# ============ 训练 ============
print("\n========== 开始训练 ==========")

model.compile(
    optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01),
    loss=smoothed_loss,
    metrics=["accuracy"]
)

class BestModelTracker(keras.callbacks.Callback):
    def __init__(self, save_path="saved_model/best_so_far.keras"):
        super().__init__()
        self.save_path = save_path
        self.best_val_loss = float('inf')
        
    def on_epoch_end(self, epoch, logs=None):
        global best_weights, best_val_loss, current_epoch
        current_epoch = epoch + 1
        val_loss = logs.get('val_loss', float('inf'))
        val_acc = logs.get('val_accuracy', 0)
        
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_weights = [w.numpy() if hasattr(w, 'numpy') else w for w in self.model.get_weights()]
            best_val_loss = val_loss
            print(f"  🏆 新最佳！val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
            self.model.save(self.save_path)
        else:
            print(f"  val_loss={val_loss:.4f} (最佳: {self.best_val_loss:.4f})")

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1
    ),
    BestModelTracker(save_path="saved_model/best_so_far.keras"),
    keras.callbacks.ModelCheckpoint(
        "saved_model/phase1_lm.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),
    keras.callbacks.TerminateOnNaN(),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1
)

print("\n========== 训练结束 ==========")
if best_weights is not None:
    model.set_weights(best_weights)
    print(f"已恢复最佳权重 (val_loss={best_val_loss:.4f})")

model.save("saved_model/phase1_lm.keras")
model.save("saved_model/phase1_lm_v2.keras")
model.save("saved_model/phase1_lm_v2.h5")
print("✅ 保存完成！")

for tmp in ["processed_train.tmp", "processed_val.tmp"]:
    if os.path.exists(tmp):
        os.remove(tmp)

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
