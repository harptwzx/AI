import tensorflow as tf
import numpy as np
import json
import os

vocab_path = 'vocab.json'
with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

id2token = {int(v): k for k, v in vocab.items()}
vocab_size = len(vocab)

saved_model_path = 'path/to/your/saved_model'
loaded = tf.saved_model.load(saved_model_path)

infer = loaded.signatures['serving_default']

all_vars = {v.name: v for v in loaded.variables}

weights = {}

weights['token_embedding'] = all_vars['token_embedding/embeddings:0']
weights['position_embedding'] = all_vars['position_embedding/embeddings:0']

for layer_idx in range(3):
    prefix = f'transformer_layer_{layer_idx}'
    weights[f'{prefix}_attn_wq'] = all_vars[f'{prefix}/multi_head_attention/query/kernel:0']
    weights[f'{prefix}_attn_wq_b'] = all_vars[f'{prefix}/multi_head_attention/query/bias:0']
    weights[f'{prefix}_attn_wk'] = all_vars[f'{prefix}/multi_head_attention/key/kernel:0']
    weights[f'{prefix}_attn_wk_b'] = all_vars[f'{prefix}/multi_head_attention/key/bias:0']
    weights[f'{prefix}_attn_wv'] = all_vars[f'{prefix}/multi_head_attention/value/kernel:0']
    weights[f'{prefix}_attn_wv_b'] = all_vars[f'{prefix}/multi_head_attention/value/bias:0']
    weights[f'{prefix}_attn_wo'] = all_vars[f'{prefix}/multi_head_attention/attention_output/kernel:0']
    weights[f'{prefix}_attn_wo_b'] = all_vars[f'{prefix}/multi_head_attention/attention_output/bias:0']
    weights[f'{prefix}_ffn_w1'] = all_vars[f'{prefix}/ffn/dense/kernel:0']
    weights[f'{prefix}_ffn_b1'] = all_vars[f'{prefix}/ffn/dense/bias:0']
    weights[f'{prefix}_ffn_w2'] = all_vars[f'{prefix}/ffn/dense_1/kernel:0']
    weights[f'{prefix}_ffn_b2'] = all_vars[f'{prefix}/ffn/dense_1/bias:0']
    weights[f'{prefix}_ln1_gamma'] = all_vars[f'{prefix}/layer_norm/gamma:0']
    weights[f'{prefix}_ln1_beta'] = all_vars[f'{prefix}/layer_norm/beta:0']
    weights[f'{prefix}_ln2_gamma'] = all_vars[f'{prefix}/layer_norm_1/gamma:0']
    weights[f'{prefix}_ln2_beta'] = all_vars[f'{prefix}/layer_norm_1/beta:0']

weights['final_ln_gamma'] = all_vars['final_layer_norm/gamma:0']
weights['final_ln_beta'] = all_vars['final_layer_norm/beta:0']

if 'output_projection/kernel:0' in all_vars:
    weights['output_proj_w'] = all_vars['output_projection/kernel:0']
    weights['output_proj_b'] = all_vars['output_projection/bias:0']
else:
    weights['output_proj_w'] = weights['token_embedding']
    weights['output_proj_b'] = None

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 3
MAX_SEQ_LEN = 256

def create_transformer_block(inputs, embed_dim, num_heads, ff_dim, weights_dict, layer_idx):
    prefix = f'transformer_layer_{layer_idx}'
    batch_size = tf.shape(inputs)[0]
    seq_len = tf.shape(inputs)[1]

    q = tf.matmul(inputs, weights_dict[f'{prefix}_attn_wq']) + weights_dict[f'{prefix}_attn_wq_b']
    k = tf.matmul(inputs, weights_dict[f'{prefix}_attn_wk']) + weights_dict[f'{prefix}_attn_wk_b']
    v = tf.matmul(inputs, weights_dict[f'{prefix}_attn_wv']) + weights_dict[f'{prefix}_attn_wv_b']

    head_dim = embed_dim // num_heads
    q = tf.reshape(q, [batch_size, seq_len, num_heads, head_dim])
    k = tf.reshape(k, [batch_size, seq_len, num_heads, head_dim])
    v = tf.reshape(v, [batch_size, seq_len, num_heads, head_dim])

    q = tf.transpose(q, [0, 2, 1, 3])
    k = tf.transpose(k, [0, 2, 1, 3])
    v = tf.transpose(v, [0, 2, 1, 3])

    scale = tf.cast(head_dim, tf.float32) ** -0.5
    attn_scores = tf.matmul(q, k, transpose_b=True) * scale

    mask = tf.linalg.band_part(tf.ones([seq_len, seq_len]), -1, 0)
    mask = tf.cast(mask, tf.float32)
    attn_scores = attn_scores * mask + (1.0 - mask) * -1e9

    attn_weights = tf.nn.softmax(attn_scores, axis=-1)
    attn_output = tf.matmul(attn_weights, v)

    attn_output = tf.transpose(attn_output, [0, 2, 1, 3])
    attn_output = tf.reshape(attn_output, [batch_size, seq_len, embed_dim])

    attn_output = tf.matmul(attn_output, weights_dict[f'{prefix}_attn_wo']) + weights_dict[f'{prefix}_attn_wo_b']

    ln1 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        gamma_initializer=tf.constant_initializer(weights_dict[f'{prefix}_ln1_gamma'].numpy()),
        beta_initializer=tf.constant_initializer(weights_dict[f'{prefix}_ln1_beta'].numpy()),
        name=f'{prefix}_ln1'
    )(inputs + attn_output)

    ffn_output = tf.matmul(ln1, weights_dict[f'{prefix}_ffn_w1']) + weights_dict[f'{prefix}_ffn_b1']
    ffn_output = tf.nn.gelu(ffn_output)
    ffn_output = tf.matmul(ffn_output, weights_dict[f'{prefix}_ffn_w2']) + weights_dict[f'{prefix}_ffn_b2']

    ln2 = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        gamma_initializer=tf.constant_initializer(weights_dict[f'{prefix}_ln2_gamma'].numpy()),
        beta_initializer=tf.constant_initializer(weights_dict[f'{prefix}_ln2_beta'].numpy()),
        name=f'{prefix}_ln2'
    )(ln1 + ffn_output)

    return ln2

def build_pretrained_model(weights_dict, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_seq_len):
    inputs = tf.keras.layers.Input(shape=(max_seq_len,), dtype=tf.int32, name='input_ids')

    token_embed = tf.keras.layers.Embedding(
        vocab_size, embed_dim,
        embeddings_initializer=tf.constant_initializer(weights_dict['token_embedding'].numpy()),
        name='token_embedding'
    )(inputs)

    positions = tf.range(start=0, limit=max_seq_len, delta=1)
    pos_embed = tf.keras.layers.Embedding(
        max_seq_len, embed_dim,
        embeddings_initializer=tf.constant_initializer(weights_dict['position_embedding'].numpy()),
        name='position_embedding'
    )(positions)

    x = token_embed + pos_embed

    for i in range(num_layers):
        x = create_transformer_block(x, embed_dim, num_heads, ff_dim, weights_dict, i)

    x = tf.keras.layers.LayerNormalization(
        epsilon=1e-6,
        gamma_initializer=tf.constant_initializer(weights_dict['final_ln_gamma'].numpy()),
        beta_initializer=tf.constant_initializer(weights_dict['final_ln_beta'].numpy()),
        name='final_ln'
    )(x)

    if weights_dict['output_proj_b'] is not None:
        logits = tf.matmul(x, weights_dict['output_proj_w'], transpose_b=True) + weights_dict['output_proj_b']
    else:
        logits = tf.matmul(x, weights_dict['output_proj_w'], transpose_b=True)

    model = tf.keras.Model(inputs=inputs, outputs=logits, name='PretrainedTransformer')
    return model

model = build_pretrained_model(weights, vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, MAX_SEQ_LEN)

def tokenize(text, vocab, max_len):
    tokens = text.lower().split()
    ids = []
    for t in tokens:
        if t in vocab:
            ids.append(int(vocab[t]))
        else:
            ids.append(int(vocab.get('<UNK>', 1)))
    if len(ids) < max_len:
        ids = ids + [int(vocab.get('<PAD>', 0))] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
    return ids

def load_dialogue_data(json_path, vocab, max_len):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    inputs = []
    labels = []
    bos_id = int(vocab.get('<BOS>', 2))
    sep_id = int(vocab.get('<SEP>', 3))
    eos_id = int(vocab.get('<EOS>', 4))
    pad_id = int(vocab.get('<PAD>', 0))

    for item in data:
        input_text = item.get('input', '')
        response_text = item.get('response', '')

        input_ids = tokenize(input_text, vocab, max_len)
        response_ids = tokenize(response_text, vocab, max_len)

        input_ids = [i for i in input_ids if i != pad_id]
        response_ids = [i for i in response_ids if i != pad_id]

        sequence = [bos_id] + input_ids + [sep_id] + response_ids + [eos_id]

        if len(sequence) > max_len:
            sequence = sequence[:max_len]
        else:
            sequence = sequence + [pad_id] * (max_len - len(sequence))

        inputs.append(sequence[:-1])
        labels.append(sequence[1:])

    return tf.constant(inputs, dtype=tf.int32), tf.constant(labels, dtype=tf.int32)

dialogue_json_path = 'dialogue_data.json'
X, Y = load_dialogue_data(dialogue_json_path, vocab, MAX_SEQ_LEN)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(X, Y, batch_size=16, epochs=10, validation_split=0.1)

model.save('dialogue_finetuned_model.keras')
