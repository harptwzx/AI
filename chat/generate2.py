#!/usr/bin/env python3
"""
生成脚本
========
给定上文，让模型自回归生成下文。

用法:
    python generate.py                          # 交互模式
    python generate.py --prompt "he looked at the window and"   # 一次性生成
    python generate.py --model ./models/best_model.keras --max_tokens 100
"""

import os
import re
import argparse
import numpy as np
import tensorflow as tf

from model import GPTModel, WarmupCosineDecay, count_parameters
from tokenizer import load_tokenizer


# ============================================================
# 配置
# ============================================================
class GenerateConfig:
    # 模型路径
    model_path = "./models/best_model.keras"

    # Tokenizer 路径
    tokenizer_dir = "./data/tokenizer"

    # 生成参数
    max_seq_len = 128          # 模型支持的最大序列长度
    max_new_tokens = 50        # 最多生成多少个新 token
    temperature = 0.8          # 采样温度，<1更保守，>1更随机
    top_k = 40                 # Top-K 采样，0表示不限制
    top_p = 0.9                # Nucleus 采样，1.0表示不限制


# ============================================================
# 加载
# ============================================================
def load_model_and_tokenizer(config: GenerateConfig):
    """加载模型和 tokenizer"""
    print(f"[Load] Tokenizer from {config.tokenizer_dir}")
    tokenizer = load_tokenizer(config.tokenizer_dir)
    print(f"[Load] vocab_size={tokenizer.vocab_size}")

    # 获取 endoftext token id
    if hasattr(tokenizer, 'SPECIAL_TOKENS'):
        eot_id = tokenizer.SPECIAL_TOKENS.get("<endoftext>", 2)
    else:
        eot_id = tokenizer.special_tokens.get("<|endoftext|>", 32001)

    print(f"[Load] Model from {config.model_path}")
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"模型文件不存在: {config.model_path}")

    model = tf.keras.models.load_model(
        config.model_path,
        custom_objects={
            "GPTModel": GPTModel,
            "WarmupCosineDecay": WarmupCosineDecay,
        },
        compile=False,  # 生成不需要编译，避免 masked_loss 找不到
    )
    print(f"[Load] 参数量: {count_parameters(model):,}")

    return model, tokenizer, eot_id


# ============================================================
# 智能解码：修复 WordTokenizer decode 后单词粘在一起的问题
# ============================================================
def smart_decode(tokenizer, ids: list) -> str:
    """
    WordTokenizer.decode() 直接拼接 token，没有空格。
    例如 ["he", "looked", "at"] -> "helookedat"

    我们通过检查每个 token 是否是完整单词，在适当位置加空格。
    """
    tokens = []
    for i in ids:
        token = tokenizer.id_to_word.get(i, "<unk>")
        if token in tokenizer.SPECIAL_TOKENS:
            continue
        tokens.append(token)

    if not tokens:
        return ""

    # 拼接策略：
    # - 第一个 token 直接放
    # - 如果当前 token 是长度>=2的纯字母单词，且前一个也是字母，加空格
    # - 否则直接拼接（处理前缀/后缀、标点等）
    result_parts = [tokens[0]]

    for i in range(1, len(tokens)):
        token = tokens[i]
        prev = tokens[i - 1]

        # 当前 token 是完整单词（长度>=2，纯字母）
        is_word = len(token) >= 2 and token.isalpha()
        # 前一个 token 是字母类
        prev_is_alpha = prev.isalpha()

        if is_word and prev_is_alpha:
            # 检查是否是常见前缀+词根组合（如 un+happy），不加空格
            # 简单判断：如果当前 token 在词表中且频率较高，视为独立单词
            if token in tokenizer.word_to_id and len(token) > 2:
                result_parts.append(" " + token)
            else:
                result_parts.append(token)
        else:
            result_parts.append(token)

    raw = "".join(result_parts)

    # 清理多余空格
    raw = re.sub(r'\s+', ' ', raw).strip()

    # 标点前不要空格
    raw = re.sub(r' \.', '.', raw)
    raw = re.sub(r' ,', ',', raw)
    raw = re.sub(r' !', '!', raw)
    raw = re.sub(r' \?', '?', raw)
    raw = re.sub(r' \)', ')', raw)
    raw = re.sub(r'\( ', '(', raw)

    return raw


# ============================================================
# 生成核心
# ============================================================
def generate(model, tokenizer, eot_id: int, prompt: str, config: GenerateConfig) -> str:
    """
    自回归生成文本。

    prompt: 上文，如 "he looked at the window and"
    返回: (生成部分, 完整文本)
    """
    # 编码 prompt
    input_ids = tokenizer.encode(prompt)
    print(f"[Gen] Prompt tokens ({len(input_ids)}): {input_ids[:20]}{'...' if len(input_ids) > 20 else ''}")

    generated_ids = list(input_ids)

    for step in range(config.max_new_tokens):
        # 截断到 max_seq_len
        context = generated_ids[-config.max_seq_len:]
        context_tensor = tf.constant([context], dtype=tf.int32)

        # 模型推理
        logits = model(context_tensor, training=False)  # (1, seq_len, vocab_size)

        # 取最后一个位置的 logits
        next_token_logits = logits[0, -1, :]  # (vocab_size,)

        # 温度缩放
        next_token_logits = next_token_logits / config.temperature

        # Top-K 过滤
        if config.top_k > 0:
            top_k_values, top_k_indices = tf.nn.top_k(next_token_logits, k=config.top_k)
            filtered_logits = tf.ones_like(next_token_logits) * float('-inf')
            filtered_logits = tf.tensor_scatter_nd_update(
                filtered_logits,
                tf.expand_dims(top_k_indices, 1),
                top_k_values
            )
            next_token_logits = filtered_logits

        # Top-P (Nucleus) 过滤
        if config.top_p < 1.0:
            sorted_logits = tf.sort(next_token_logits, direction='DESCENDING')
            sorted_probs = tf.nn.softmax(sorted_logits)
            cumsum_probs = tf.cumsum(sorted_probs)

            # 找到累积概率超过 top_p 的位置
            sorted_indices_to_remove = cumsum_probs > config.top_p
            # 保留第一个超过的
            sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, shift=1, axis=0)
            sorted_indices_to_remove = tf.tensor_scatter_nd_update(
                sorted_indices_to_remove,
                [[0]],
                [False]
            )

            # 映射回原始索引
            sorted_indices = tf.argsort(next_token_logits, direction='DESCENDING')
            indices_to_remove = tf.zeros_like(next_token_logits, dtype=tf.bool)
            indices_to_remove = tf.tensor_scatter_nd_update(
                indices_to_remove,
                tf.expand_dims(sorted_indices, 1),
                sorted_indices_to_remove
            )

            next_token_logits = tf.where(indices_to_remove, float('-inf'), next_token_logits)

        # 采样下一个 token
        probs = tf.nn.softmax(next_token_logits)

        # 使用 tf.random.categorical 采样
        log_probs = tf.math.log(tf.expand_dims(probs, 0))
        next_token_id = int(tf.random.categorical(log_probs, num_samples=1)[0, 0])

        generated_ids.append(next_token_id)

        # 遇到结束符停止
        if next_token_id == eot_id:
            print(f"[Gen] 遇到 <endoftext>，在第 {step + 1} 个 token 停止")
            break

    # 解码
    full_text = smart_decode(tokenizer, generated_ids)

    # 分离 prompt 和生成部分
    prompt_decoded = smart_decode(tokenizer, input_ids)
    if full_text.startswith(prompt_decoded):
        generated_only = full_text[len(prompt_decoded):].strip()
    else:
        # 如果 decode 后前缀对不上，直接返回后半部分
        generated_only = smart_decode(tokenizer, generated_ids[len(input_ids):])

    return generated_only, full_text


# ============================================================
# 交互模式
# ============================================================
def interactive_generate(model, tokenizer, eot_id: int, config: GenerateConfig):
    """交互式生成"""
    print("\n" + "=" * 60)
    print("  English ChatAI — 文本生成器")
    print("=" * 60)
    print("输入提示词，AI 会补全下文。")
    print("命令: /quit 退出, /temp <值> 调温度, /tokens <值> 调长度")
    print("=" * 60)

    while True:
        print()
        try:
            prompt = input("Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not prompt:
            continue

        # 命令处理
        if prompt.lower() in ("/quit", "/exit", "/q"):
            print("再见！")
            break

        if prompt.startswith("/temp "):
            try:
                config.temperature = float(prompt.split()[1])
                print(f"[Config] temperature = {config.temperature}")
            except:
                print("[Error] 用法: /temp 0.8")
            continue

        if prompt.startswith("/tokens "):
            try:
                config.max_new_tokens = int(prompt.split()[1])
                print(f"[Config] max_new_tokens = {config.max_new_tokens}")
            except:
                print("[Error] 用法: /tokens 50")
            continue

        # 生成
        try:
            generated, full = generate(model, tokenizer, eot_id, prompt, config)
            print(f"\n[AI 生成]")
            print(f"  补全: {generated}")
            print(f"  全文: {full}")
        except Exception as e:
            print(f"[Error] 生成失败: {e}")
            import traceback
            traceback.print_exc()


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPT 文本生成")
    parser.add_argument("--model", type=str, default=None, help="模型路径 (.keras)")
    parser.add_argument("--tokenizer", type=str, default=None, help="tokenizer 目录")
    parser.add_argument("--prompt", type=str, default=None, help="一次性生成，不进入交互模式")
    parser.add_argument("--max-tokens", type=int, default=50, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度 (0.1~2.0)")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K 采样 (0=不限制)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus 采样 (1.0=不限制)")

    args = parser.parse_args()

    config = GenerateConfig()
    if args.model:
        config.model_path = args.model
    if args.tokenizer:
        config.tokenizer_dir = args.tokenizer
    if args.max_tokens:
        config.max_new_tokens = args.max_tokens
    if args.temperature:
        config.temperature = args.temperature
    if args.top_k is not None:
        config.top_k = args.top_k
    if args.top_p:
        config.top_p = args.top_p

    print("=" * 60)
    print("  English ChatAI — 文本生成")
    print("=" * 60)
    print(f"  模型: {config.model_path}")
    print(f"  Tokenizer: {config.tokenizer_dir}")
    print(f"  温度: {config.temperature}")
    print(f"  Top-K: {config.top_k}")
    print(f"  Top-P: {config.top_p}")
    print(f"  最大生成: {config.max_new_tokens} tokens")
    print("=" * 60)

    model, tokenizer, eot_id = load_model_and_tokenizer(config)

    if args.prompt:
        generated, full = generate(model, tokenizer, eot_id, args.prompt, config)
        print(f"\nPrompt: {args.prompt}")
        print(f"Generated: {generated}")
        print(f"Full: {full}")
    else:
        interactive_generate(model, tokenizer, eot_id, config)