import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import json
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm  # 进度条库

# ========== 1. 加载词汇表 ==========
with open("vocab.json", "r", encoding="utf-8") as f:
    word_to_id = json.load(f)

id_to_word = {int(v): k for k, v in word_to_id.items()}
VOCAB_SIZE = len(word_to_id)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

print(f"词汇表大小: {VOCAB_SIZE}")


# ========== 2. 读取TXT并转Token IDs ==========
def text_to_token_ids(filepath, word_to_id):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    text = text.lower()
    words = text.split()
    
    token_ids = []
    for word in words:
        clean = word.strip(".,!?;:\"'()[]{}")
        token_ids.append(word_to_id.get(clean, UNK_ID))
    
    return token_ids


print("读取哈利波特TXT...")
token_ids = text_to_token_ids("harry_potter.txt", word_to_id)
print(f"总Token数: {len(token_ids)}")


# ========== 3. 滑动窗口生成训练数据 ==========
SEQ_LENGTH = 30

def create_training_data(token_ids, seq_length, stride=1):
    inputs, targets = [], []
    for i in range(0, len(token_ids) - seq_length, stride):
        inp = token_ids[i : i + seq_length]
        tgt = token_ids[i + 1 : i + seq_length + 1]
        if len(tgt) == seq_length:
            inputs.append(inp)
            targets.append(tgt)
    return np.array(inputs, dtype=np.int32), np.array(targets, dtype=np.int32)


X_train, y_train = create_training_data(token_ids, seq_length=SEQ_LENGTH)
print(f"训练样本数: {len(X_train)}")


# ========== 4. 纯手写神经层 ==========

def softmax(x, axis=-1):
    exp_x = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, epsilon=1e-6):
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / tf.sqrt(variance + epsilon)


def relu(x):
    return tf.maximum(x, 0)


def dropout(x, rate, training):
    if not training:
        return x
    mask = tf.random.uniform(tf.shape(x)) > rate
    return tf.cast(mask, tf.float32) * x / (1.0 - rate)


class Linear(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = tf.Variable(
            tf.random.uniform([in_features, out_features], -limit, limit),
            name="weight"
        )
        self.b = tf.Variable(tf.zeros([out_features]), name="bias")
    
    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b


class Embedding(tf.Module):
    def __init__(self, num_embeddings, embedding_dim, name=None):
        super().__init__(name=name)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = tf.Variable(
            tf.random.normal([num_embeddings, embedding_dim], stddev=0.02),
            name="embedding"
        )
    
    def __call__(self, indices):
        return tf.nn.embedding_lookup(self.weight, indices)


class MultiHeadAttention(tf.Module):
    def __init__(self, d_model, num_heads, name=None):
        super().__init__(name=name)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = d_model // num_heads
        
        self.Wq = Linear(d_model, d_model, name="Wq")
        self.Wk = Linear(d_model, d_model, name="Wk")
        self.Wv = Linear(d_model, d_model, name="Wv")
        self.Wo = Linear(d_model, d_model, name="Wo")
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def __call__(self, q, k, v, mask, training):
        batch_size = tf.shape(q)[0]
        
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk)
        
        if mask is not None:
            scores += (mask * -1e9)
        
        attn_weights = softmax(scores, axis=-1)
        output = tf.matmul(attn_weights, v)
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, -1, self.d_model])
        
        return self.Wo(output)


class FeedForward(tf.Module):
    def __init__(self, d_model, dff, name=None):
        super().__init__(name=name)
        self.linear1 = Linear(d_model, dff, name="linear1")
        self.linear2 = Linear(dff, d_model, name="linear2")
    
    def __call__(self, x):
        x = self.linear1(x)
        x = relu(x)
        x = self.linear2(x)
        return x


class DecoderLayer(tf.Module):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads, name="mha")
        self.ffn = FeedForward(d_model, dff, name="ffn")
        
        self.norm1_gamma = tf.Variable(tf.ones([d_model]), name="norm1_gamma")
        self.norm1_beta = tf.Variable(tf.zeros([d_model]), name="norm1_beta")
        self.norm2_gamma = tf.Variable(tf.ones([d_model]), name="norm2_gamma")
        self.norm2_beta = tf.Variable(tf.zeros([d_model]), name="norm2_beta")
        
        self.dropout_rate = dropout_rate
    
    def __call__(self, x, training, mask):
        attn_out = self.mha(x, x, x, mask, training)
        attn_out = dropout(attn_out, self.dropout_rate, training)
        x = layer_norm(x + attn_out)
        x = x * self.norm1_gamma + self.norm1_beta
        
        ffn_out = self.ffn(x)
        ffn_out = dropout(ffn_out, self.dropout_rate, training)
        x = layer_norm(x + ffn_out)
        x = x * self.norm2_gamma + self.norm2_beta
        
        return x


class GPTModel(tf.Module):
    def __init__(self, vocab_size, d_model=128, num_layers=4,
                 num_heads=4, dff=256, max_seq_length=64, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        self.token_embedding = Embedding(vocab_size, d_model, name="token_emb")
        self.position_embedding = Embedding(max_seq_length, d_model, name="pos_emb")
        
        self.dropout_rate = dropout_rate
        
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate, name=f"decoder_{i}")
            for i in range(num_layers)
        ]
        
        self.output_linear = Linear(d_model, vocab_size, name="output")
    
    def __call__(self, inputs, training=False):
        seq_length = tf.shape(inputs)[1]
        
        token_emb = self.token_embedding(inputs)
        positions = tf.range(seq_length)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        x = dropout(x, self.dropout_rate, training)
        
        mask = 1 - tf.linalg.band_part(tf.ones([seq_length, seq_length]), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        
        for layer in self.decoder_layers:
            x = layer(x, training, mask)
        
        logits = self.output_linear(x)
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=30, temperature=1.0, repetition_penalty=1.5):
        generated = prompt_ids
        
        for _ in range(max_new_tokens):
            curr_len = tf.shape(generated)[1]
            if curr_len > self.max_seq_length:
                generated = generated[:, -self.max_seq_length:]
            
            logits = self(generated, training=False)
            next_logits = logits[:, -1, :] / temperature
            
            next_logits_np = next_logits.numpy()
            for token_id in set(generated[0].numpy().tolist()):
                next_logits_np[0][token_id] /= repetition_penalty
            next_logits = tf.constant(next_logits_np, dtype=tf.float32)
            
            probs = softmax(next_logits)
            next_token = tf.random.categorical(tf.math.log(probs), num_samples=1)
            
            generated = tf.concat([generated, next_token], axis=1)
        
        return generated


# ========== 5. 训练（带进度条）==========

# CPU优化参数
D_MODEL = 64        # 改小
NUM_LAYERS = 2      # 改少
NUM_HEADS = 2       # 改少
DFF = 128           # 改小
DROPOUT = 0.1
BATCH_SIZE = 32     # 改小
EPOCHS = 20         # 先跑20轮
LEARNING_RATE = 0.001

model = GPTModel(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    dff=DFF,
    max_seq_length=SEQ_LENGTH,
    dropout_rate=DROPOUT
)

optimizer = tf.optimizers.Adam(learning_rate=LEARNING_RATE)


def compute_loss(y_true, y_pred):
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 计算总批次数
total_batches = len(X_train) // BATCH_SIZE
if len(X_train) % BATCH_SIZE != 0:
    total_batches += 1

print(f"\n=== 开始训练 ===")
print(f"总样本: {len(X_train)}, 批次大小: {BATCH_SIZE}, 每轮批次: {total_batches}")
print(f"模型参数: d_model={D_MODEL}, layers={NUM_LAYERS}, heads={NUM_HEADS}, dff={DFF}")
print("=" * 50)

start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    total_loss = 0.0
    num_batches = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(dataset, total=total_batches, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for batch_x, batch_y in pbar:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = compute_loss(batch_y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        loss_val = loss.numpy()
        total_loss += loss_val
        num_batches += 1
        
        # 更新进度条显示
        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'avg_loss': f'{total_loss/num_batches:.4f}'
        })
    
    pbar.close()
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1}/{EPOCHS} 完成 | 平均Loss: {avg_loss:.4f} | 耗时: {epoch_time:.1f}s")

total_time = time.time() - start_time
print(f"\n训练完成！总耗时: {total_time:.1f}s ({total_time/60:.1f}分钟)")


# ========== 6. 导出 SavedModel ==========
print("\n=== 导出 SavedModel ===")

@tf.function(input_signature=[tf.TensorSpec(shape=[None, None], dtype=tf.int32)])
def serving_fn(input_ids):
    return {"logits": model(input_ids, training=False)}

tf.saved_model.save(
    model,
    "saved_model/gpt_language_model",
    signatures={"serving_default": serving_fn}
)

print("SavedModel 已导出到: saved_model/gpt_language_model/")


# ========== 7. 测试生成（带进度）==========
print("\n=== 测试生成 ===")

def generate_text(prompt_text, max_tokens=30, temperature=0.8):
    words = prompt_text.lower().split()
    prompt_ids = [word_to_id.get(w, UNK_ID) for w in words]
    prompt_tensor = tf.constant([prompt_ids], dtype=tf.int32)
    
    generated = model.generate(
        prompt_tensor,
        max_new_tokens=max_tokens,
        temperature=temperature,
        repetition_penalty=1.5
    )
    
    new_ids = generated[0, len(prompt_ids):].numpy().tolist()
    new_words = [id_to_word.get(i, "<|unk|>") for i in new_ids]
    return " ".join(new_words)


test_prompts = ["harry looked at", "the magic", "ron and", "the dark lord"]

for prompt in tqdm(test_prompts, desc="生成测试"):
    result = generate_text(prompt, max_tokens=20, temperature=0.8)
    print(f"\nPrompt: '{prompt}'")
    print(f"Generated: '{prompt} {result}'")
