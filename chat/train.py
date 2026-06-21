import json
import numpy as np
import tensorflow as tf
print('loading successfully')


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
    """手写softmax"""
    exp_x = tf.exp(x - tf.reduce_max(x, axis=axis, keepdims=True))
    return exp_x / tf.reduce_sum(exp_x, axis=axis, keepdims=True)


def layer_norm(x, epsilon=1e-6):
    """手写字层归一化"""
    mean = tf.reduce_mean(x, axis=-1, keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / tf.sqrt(variance + epsilon)


def relu(x):
    """手写ReLU"""
    return tf.maximum(x, 0)


def dropout(x, rate, training):
    """手写dropout"""
    if not training:
        return x
    mask = tf.random.uniform(tf.shape(x)) > rate
    return tf.cast(mask, tf.float32) * x / (1.0 - rate)


class Linear(tf.Module):
    """全连接层: y = xW + b"""
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        # Xavier初始化
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.W = tf.Variable(
            tf.random.uniform([in_features, out_features], -limit, limit),
            name="weight"
        )
        self.b = tf.Variable(tf.zeros([out_features]), name="bias")
    
    def __call__(self, x):
        return tf.matmul(x, self.W) + self.b


class Embedding(tf.Module):
    """嵌入层"""
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
    """手写多头注意力"""
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
        """拆分多头: (batch, seq, d_model) -> (batch, heads, seq, depth)"""
        x = tf.reshape(x, [batch_size, -1, self.num_heads, self.depth])
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def __call__(self, q, k, v, mask, training):
        batch_size = tf.shape(q)[0]
        
        # 线性投影
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        
        # 拆多头
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # 缩放点积注意力
        dk = tf.cast(self.depth, tf.float32)
        scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(dk)
        
        # 应用掩码
        if mask is not None:
            scores += (mask * -1e9)
        
        # Softmax
        attn_weights = softmax(scores, axis=-1)
        
        # 加权求和
        output = tf.matmul(attn_weights, v)
        
        # 合并多头
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, [batch_size, -1, self.d_model])
        
        # 最终线性
        return self.Wo(output)


class FeedForward(tf.Module):
    """手写前馈网络"""
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
    """手写Decoder块"""
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        self.mha = MultiHeadAttention(d_model, num_heads, name="mha")
        self.ffn = FeedForward(d_model, dff, name="ffn")
        
        # 层归一化参数
        self.norm1_gamma = tf.Variable(tf.ones([d_model]), name="norm1_gamma")
        self.norm1_beta = tf.Variable(tf.zeros([d_model]), name="norm1_beta")
        self.norm2_gamma = tf.Variable(tf.ones([d_model]), name="norm2_gamma")
        self.norm2_beta = tf.Variable(tf.zeros([d_model]), name="norm2_beta")
        
        self.dropout_rate = dropout_rate
    
    def __call__(self, x, training, mask):
        # 自注意力
        attn_out = self.mha(x, x, x, mask, training)
        attn_out = dropout(attn_out, self.dropout_rate, training)
        x = layer_norm(x + attn_out)
        x = x * self.norm1_gamma + self.norm1_beta
        
        # 前馈
        ffn_out = self.ffn(x)
        ffn_out = dropout(ffn_out, self.dropout_rate, training)
        x = layer_norm(x + ffn_out)
        x = x * self.norm2_gamma + self.norm2_beta
        
        return x


class GPTModel(tf.Module):
    """手写GPT模型"""
    def __init__(self, vocab_size, d_model=128, num_layers=4,
                 num_heads=4, dff=256, max_seq_length=64, dropout_rate=0.1, name=None):
        super().__init__(name=name)
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # 嵌入
        self.token_embedding = Embedding(vocab_size, d_model, name="token_emb")
        self.position_embedding = Embedding(max_seq_length, d_model, name="pos_emb")
        
        self.dropout_rate = dropout_rate
        
        # Decoder层
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout_rate, name=f"decoder_{i}")
            for i in range(num_layers)
        ]
        
        # 输出层
        self.output_linear = Linear(d_model, vocab_size, name="output")
    
    def __call__(self, inputs, training=False):
        seq_length = tf.shape(inputs)[1]
        
        # Token嵌入 + 位置嵌入
        token_emb = self.token_embedding(inputs)
        positions = tf.range(seq_length)
        pos_emb = self.position_embedding(positions)
        x = token_emb + pos_emb
        x = dropout(x, self.dropout_rate, training)
        
        # 因果掩码
        mask = 1 - tf.linalg.band_part(tf.ones([seq_length, seq_length]), -1, 0)
        mask = mask[tf.newaxis, tf.newaxis, :, :]
        
        # Decoder层
        for layer in self.decoder_layers:
            x = layer(x, training, mask)
        
        # 输出logits
        logits = self.output_linear(x)
        
        return logits
    
    def generate(self, prompt_ids, max_new_tokens=30, temperature=1.0, repetition_penalty=1.5):
        """自回归生成"""
        generated = prompt_ids
        
        for _ in range(max_new_tokens):
            curr_len = tf.shape(generated)[1]
            if curr_len > self.max_seq_length:
                generated = generated[:, -self.max_seq_length:]
            
            # 前向传播
            logits = self(generated, training=False)
            next_logits = logits[:, -1, :] / temperature
            
            # 重复惩罚（numpy操作）
            next_logits_np = next_logits.numpy()
            for token_id in set(generated[0].numpy().tolist()):
                next_logits_np[0][token_id] /= repetition_penalty
            next_logits = tf.constant(next_logits_np, dtype=tf.float32)
            
            # 采样
            probs = softmax(next_logits)
            next_token = tf.random.categorical(tf.math.log(probs), num_samples=1)
            
            generated = tf.concat([generated, next_token], axis=1)
        
        return generated


# ========== 5. 训练 ==========

D_MODEL = 128
NUM_LAYERS = 4
NUM_HEADS = 4
DFF = 256
DROPOUT = 0.1
BATCH_SIZE = 64
EPOCHS = 100
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
    """计算损失"""
    mask = tf.cast(tf.not_equal(y_true, PAD_ID), tf.float32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
    loss *= mask
    return tf.reduce_sum(loss) / tf.reduce_sum(mask)


# 数据集
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
dataset = dataset.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 训练循环
print("\n=== 开始训练 ===")
for epoch in range(EPOCHS):
    total_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(batch_x, training=True)
            loss = compute_loss(batch_y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        total_loss += loss.numpy()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.4f}")


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


# ========== 7. 测试生成 ==========

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


print("\n=== 测试生成 ===")
test_prompts = ["harry looked at", "the magic", "ron and", "the dark lord"]

for prompt in test_prompts:
    print(f"\nPrompt: '{prompt}'")
    result = generate_text(prompt, max_tokens=20, temperature=0.8)
    print(f"Generated: '{prompt} {result}'")
