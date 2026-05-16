#!/usr/bin/env python3
"""
AI 图片生成主程序
支持多种 AI 工具，通过命令行参数选择
"""
import argparse
import sys
import os
import json
from typing import Dict, Any

# 添加当前目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_generator import GeneratorFactory
from config import (
    DEFAULT_GENERATOR, 
    DEFAULT_STEPS, 
    DEFAULT_FORMAT,
    SUPPORTED_FORMATS
)
import stable_diffusion  # 这会自动注册生成器


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='AI 图片生成工具')
    
    parser.add_argument(
        '--prompt', '-p',
        type=str,
        required=True,
        help='图片描述提示词'
    )
    
    parser.add_argument(
        '--generator', '-g',
        type=str,
        default=DEFAULT_GENERATOR,
        choices=GeneratorFactory.list_generators(),
        help=f'选择 AI 生成器 (可用: {GeneratorFactory.list_generators()})'
    )
    
    parser.add_argument(
        '--steps', '-s',
        type=int,
        default=DEFAULT_STEPS,
        help=f'推理步数 (默认: {DEFAULT_STEPS})'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        default=DEFAULT_FORMAT,
        choices=SUPPORTED_FORMATS,
        help=f'图片格式 (默认: {DEFAULT_FORMAT})'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='输出目录 (默认: generated_images)'
    )
    
    parser.add_argument(
        '--negative-prompt', '-n',
        type=str,
        help='负面提示词（不希望出现的元素）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        help='随机种子（用于复现结果）'
    )
    
    parser.add_argument(
        '--output-json',
        action='store_true',
        help='输出 JSON 格式结果（用于 GitHub Actions）'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 获取生成器
    generator_kwargs = {}
    if args.output_dir:
        generator_kwargs['output_dir'] = args.output_dir
    
    generator = GeneratorFactory.get_generator(args.generator, **generator_kwargs)
    
    if not generator:
        print(f"❌ 未找到生成器: {args.generator}")
        print(f"可用生成器: {GeneratorFactory.list_generators()}")
        sys.exit(1)
    
    # 显示生成器信息
    if not args.output_json:
        info = generator.get_info()
        print(f"🤖 使用生成器: {info['name']}")
        print(f"   模型: {info['model']}")
        print(f"   平台: {info['platform']}")
    
    # 生成参数
    generate_kwargs = {
        'prompt': args.prompt,
        'steps': args.steps,
        'format': args.format,
    }
    
    if args.negative_prompt:
        generate_kwargs['negative_prompt'] = args.negative_prompt
    
    if args.seed:
        generate_kwargs['seed'] = args.seed
    
    # 生成图片
    result = generator.generate(**generate_kwargs)
    
    # 输出结果
    if args.output_json:
        # JSON 格式输出（用于 GitHub Actions）
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 人类可读格式
        if result['success']:
            print(f"\n✨ 成功！图片已保存到: {result['filepath']}")
            sys.exit(0)
        else:
            print(f"\n💥 失败: {result['error']}")
            sys.exit(1)


if __name__ == "__main__":
    main()