"""
配置文件，可配置默认参数
"""
import os
from pathlib import Path

# 基础配置
BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "generated_images"

# 生成器配置
DEFAULT_GENERATOR = os.getenv('AI_GENERATOR', 'stable_diffusion')

# Stable Diffusion 配置
STABLE_DIFFUSION_CONFIG = {
    'model_id': 'CompVis/stable-diffusion-v1-4',
    'output_dir': str(OUTPUT_DIR),
    'use_safety_checker': False
}

# 支持的图片格式
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg']

# 默认参数
DEFAULT_STEPS = 20
DEFAULT_FORMAT = 'png'

# 所有可用的生成器
AVAILABLE_GENERATORS = ['stable_diffusion']  # 后续可以添加更多