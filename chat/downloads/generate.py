"""
文本生成脚本
============
使用训练好的模型进行自回归文本生成。
支持: greedy / temperature sampling / top-k / top-p
"""

import os
import json
import tensorflow as tf
import numpy as np

from model import GPTModel
from tokenizer import BPETokenizer


class TextGenerator:
    """文本生成器"""

    def __init__(
        self,
        model_path: str,
        tokenizer_dir: str = "./data/tokenizer",
        max_seq_len: int = 256,
    ):
        self.max_seq_len = max_seq_len

        # 加载 tokenizer
        print(f"[Generator] 加载 tokenizer: {tokenizer_dir}")
        self.tokenizer = BPETokenizer.load(tokenizer_dir)

        # 加载模型
        print(f"[Generator] 加载模型: {model_path}")
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "GPTModel": GPTModel,
            }
        )
        print("[Generator] 模型加载完成")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.0,
        stop_token: str = "<|endoftext|>",
    ) -> str:
        """
        自回归文本生成

        Args:
            prompt: 输入提示文本
            max_new_tokens: 最大生成 token 数
            temperature: 温度参数（>1 更随机，<1 更确定）
            top_k: Top-K 采样（0 表示禁用）
            top_p: Top-P (nucleus) 采样（1.0 表示禁用）
            repetition_penalty: 重复惩罚（1.0 表示无惩罚）
            stop_token: 遇到此 token 停止生成

        Returns:
            生成的完整文本
        """
        # 编码 prompt
        input_ids = self.tokenizer.encode(prompt)
        input_ids = tf.constant([input_ids], dtype=tf.int32)

        generated_ids = []
        stop_id = self.tokenizer.special_tokens.get(stop_token, None)

        for _ in range(max_new_tokens):
            # 截断到最大长度
            if input_ids.shape[1] > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:]

            # 前向传播
            logits = self.model(input_ids, training=False)
            next_token_logits = logits[0, -1, :]  # 最后一个位置的 logits

            # 应用温度
            next_token_logits = next_token_logits / temperature

            # 应用重复惩罚
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids):
                    next_token_logits = tf.tensor_scatter_nd_update(
                        next_token_logits,
                        [[token_id]],
                        [next_token_logits[token_id] / repetition_penalty]
                    )

            # Top-K 采样
            if top_k > 0:
                top_k_logits, top_k_indices = tf.nn.top_k(next_token_logits, k=top_k)
            else:
                top_k_logits = next_token_logits
                top_k_indices = tf.range(len(next_token_logits))

            # Top-P (nucleus) 采样
            if top_p < 1.0:
                probs = tf.nn.softmax(top_k_logits)
                sorted_probs = tf.sort(probs, direction="DESCENDING")
                cumsum_probs = tf.cumsum(sorted_probs)

                # 找到 cumsum > top_p 的 cutoff
                cutoff_idx = tf.searchsorted(cumsum_probs, top_p, side="right")
                cutoff_idx = tf.minimum(cutoff_idx + 1, len(top_k_logits))

                top_k_logits = top_k_logits[:cutoff_idx]
                top_k_indices = top_k_indices[:cutoff_idx]

            # 采样
            probs = tf.nn.softmax(top_k_logits)
            sampled_idx = tf.random.categorical(
                tf.math.log(tf.expand_dims(probs, 0)),
                num_samples=1
            )[0, 0]

            next_token = int(top_k_indices[sampled_idx])

            # 检查停止条件
            if stop_id is not None and next_token == stop_id:
                break

            generated_ids.append(next_token)

            # 拼接到输入
            input_ids = tf.concat([
                input_ids,
                tf.constant([[next_token]], dtype=tf.int32)
            ], axis=1)

        # 解码
        full_ids = input_ids[0].numpy().tolist()
        generated_text = self.tokenizer.decode(full_ids)

        return generated_text

    def chat(
        self,
        user_message: str,
        history: list = None,
        max_new_tokens: int = 150,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        对话模式

        Args:
            user_message: 用户输入
            history: 历史对话列表 [("user", "..."), ("assistant", "...")]
        """
        # 构建 prompt
        prompt = ""
        if history:
            for role, msg in history:
                if role == "user":
                    prompt += f"<|user|>{msg}"
                else:
                    prompt += f"<|assistant|>{msg}"

        prompt += f"<|user|>{user_message}<|assistant|>"

        # 生成回复
        response = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            stop_token="<|user|>",
        )

        # 提取 assistant 的回复部分
        assistant_start = response.find("<|assistant|>")
        if assistant_start != -1:
            response = response[assistant_start + len("<|assistant|>"):]

        return response.strip()


# ============================================================
# 命令行接口
# ============================================================
def interactive_chat(model_path: str, tokenizer_dir: str):
    """交互式对话"""
    print("=" * 60)
    print("  英语 AI 助手 - 交互式对话")
    print("=" * 60)
    print("命令: /quit 退出, /clear 清空历史, /temp <值> 设置温度")
    print("-" * 60)

    generator = TextGenerator(model_path, tokenizer_dir)
    history = []
    temperature = 0.8

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input == "/quit":
            print("再见！")
            break
        elif user_input == "/clear":
            history = []
            print("[历史已清空]")
            continue
        elif user_input.startswith("/temp "):
            try:
                temperature = float(user_input.split()[1])
                print(f"[温度已设置为 {temperature}]")
            except:
                print("[格式错误，使用: /temp 0.8]")
            continue

        # 生成回复
        print("AI: ", end="", flush=True)
        response = generator.chat(
            user_input,
            history=history,
            temperature=temperature,
        )
        print(response)

        # 更新历史
        history.append(("user", user_input))
        history.append(("assistant", response))

        # 限制历史长度
        if len(history) > 20:
            history = history[-20:]


def generate_once(model_path: str, tokenizer_dir: str, prompt: str, **kwargs):
    """单次生成"""
    generator = TextGenerator(model_path, tokenizer_dir)
    result = generator.generate(prompt, **kwargs)
    print(result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="文本生成")
    parser.add_argument("--model", type=str, required=True, help="模型路径 (SaveModel)")
    parser.add_argument("--tokenizer", type=str, default="./data/tokenizer", help="Tokenizer 目录")
    parser.add_argument("--prompt", type=str, default=None, help="生成提示（不设置则进入交互模式）")
    parser.add_argument("--max-tokens", type=int, default=100, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=1.0, help="温度")
    parser.add_argument("--top-k", type=int, default=50, help="Top-K")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-P")

    args = parser.parse_args()

    if args.prompt:
        generate_once(
            args.model,
            args.tokenizer,
            args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    else:
        interactive_chat(args.model, args.tokenizer)
