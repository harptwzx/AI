"""
AI 图片生成器基础类
所有具体的生成器都需要继承此类
"""
from abc import ABC, abstractmethod
from PIL import Image
from datetime import datetime
import os
import re
from typing import Dict, Any, Optional, Tuple


class BaseImageGenerator(ABC):
    """图片生成器基类"""
    
    def __init__(self, output_dir: str = "generated_images"):
        self.output_dir = output_dir
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """确保输出目录存在"""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _sanitize_prompt(self, prompt: str, max_length: int = 50) -> str:
        """将提示词转换为安全的文件名"""
        # 移除特殊字符，保留中英文、数字和下划线
        safe = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', prompt)
        # 限制长度
        return safe[:max_length]
    
    def _generate_filename(self, prompt: str, extension: str = "png") -> str:
        """生成带时间戳的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = self._sanitize_prompt(prompt)
        return f"{self.output_dir}/{timestamp}_{safe_prompt}.{extension}"
    
    def save_image(self, image: Image.Image, filepath: str, format: str = "PNG", quality: int = 95) -> str:
        """保存图片到文件"""
        if format.upper() == "JPEG" or format.upper() == "JPG":
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image.save(filepath, format='JPEG', quality=quality)
        else:
            image.save(filepath, format='PNG')
        
        return filepath
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        steps: int = 20,
        format: str = "png",
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成图片的抽象方法
        子类必须实现
        
        返回:
            Dict 包含:
                - success: bool
                - filepath: str (保存路径)
                - image_info: dict (图片信息)
                - error: str (错误信息，如果有)
        """
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """获取生成器信息"""
        pass


class GeneratorFactory:
    """生成器工厂类，用于管理不同的 AI 工具"""
    
    _generators = {}
    
    @classmethod
    def register(cls, name: str, generator_class):
        """注册生成器"""
        cls._generators[name] = generator_class
    
    @classmethod
    def get_generator(cls, name: str, **kwargs) -> Optional[BaseImageGenerator]:
        """获取生成器实例"""
        if name in cls._generators:
            return cls._generators[name](**kwargs)
        return None
    
    @classmethod
    def list_generators(cls) -> list:
        """列出所有可用的生成器"""
        return list(cls._generators.keys())