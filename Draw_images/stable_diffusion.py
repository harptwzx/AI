"""
Stable Diffusion 图片生成器
"""
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image
from typing import Dict, Any
import os
from .base_generator import BaseImageGenerator, GeneratorFactory


class StableDiffusionGenerator(BaseImageGenerator):
    """Stable Diffusion 生成器 (CPU 版本)"""
    
    def __init__(
        self, 
        model_id: str = "CompVis/stable-diffusion-v1-4",
        output_dir: str = "generated_images",
        use_safety_checker: bool = False
    ):
        super().__init__(output_dir)
        self.model_id = model_id
        self.use_safety_checker = use_safety_checker
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """加载模型（延迟加载，首次调用时加载）"""
        print(f"📦 加载模型: {self.model_id}")
        print("⏳ 这可能需要几分钟时间（首次下载约4GB）...")
        
        # 加载模型到 CPU
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            safety_checker=None if not self.use_safety_checker else None
        )
        self.pipe = self.pipe.to("cpu")
        
        # 使用 DDIM 调度器加速
        self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
        
        print("✅ 模型加载完成")
    
    def generate(
        self,
        prompt: str,
        steps: int = 20,
        format: str = "png",
        negative_prompt: str = None,
        seed: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成图片
        
        参数:
            prompt: 提示词
            steps: 推理步数 (10-50)
            format: 图片格式 (png, jpg, jpeg)
            negative_prompt: 负面提示词
            seed: 随机种子（用于复现）
        
        返回:
            包含生成结果的字典
        """
        try:
            print(f"\n🎨 开始生成图片")
            print(f"   提示词: {prompt}")
            print(f"   步数: {steps}")
            print(f"   格式: {format}")
            
            # 设置随机种子
            if seed is not None:
                torch.manual_seed(seed)
            
            # 生成图片
            print("⏳ 生成中... (CPU 模式较慢，请耐心等待)")
            with torch.no_grad():
                result = self.pipe(
                    prompt, 
                    num_inference_steps=steps,
                    negative_prompt=negative_prompt
                )
                image = result.images[0]
            
            # 生成文件名并保存
            ext = 'jpg' if format.lower() in ['jpg', 'jpeg'] else 'png'
            save_format = 'JPEG' if ext == 'jpg' else 'PNG'
            filepath = self._generate_filename(prompt, ext)
            self.save_image(image, filepath, save_format)
            
            # 获取文件大小
            file_size = os.path.getsize(filepath)
            
            # 保存元数据
            metadata = {
                'prompt': prompt,
                'steps': steps,
                'format': format,
                'model': self.model_id,
                'negative_prompt': negative_prompt,
                'seed': seed,
                'filepath': filepath,
                'file_size': file_size,
                'image_size': image.size
            }
            
            # 保存信息到文本文件
            info_file = f"{self.output_dir}/latest_info.txt"
            with open(info_file, 'w', encoding='utf-8') as f:
                for key, value in metadata.items():
                    f.write(f"{key}: {value}\n")
            
            print(f"✅ 图片生成完成!")
            print(f"📁 保存位置: {filepath}")
            print(f"📊 文件大小: {file_size} bytes")
            print(f"🎨 图片尺寸: {image.size}")
            
            return {
                'success': True,
                'filepath': filepath,
                'image_info': metadata,
                'error': None
            }
            
        except Exception as e:
            print(f"❌ 生成失败: {str(e)}")
            return {
                'success': False,
                'filepath': None,
                'image_info': None,
                'error': str(e)
            }
    
    def get_info(self) -> Dict[str, Any]:
        """获取生成器信息"""
        return {
            'name': 'Stable Diffusion',
            'model': self.model_id,
            'supported_formats': ['png', 'jpg', 'jpeg'],
            'min_steps': 10,
            'max_steps': 50,
            'default_steps': 20,
            'requires_gpu': False,
            'platform': 'CPU'
        }


# 注册生成器到工厂
GeneratorFactory.register('stable_diffusion', StableDiffusionGenerator)