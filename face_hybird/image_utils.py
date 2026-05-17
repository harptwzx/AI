"""
图像处理工具函数
"""
import os
from PIL import Image
import numpy as np


def save_image(image, filepath, format='PNG'):
    """保存图像"""
    if isinstance(image, np.ndarray):
        # 转换 numpy array 到 PIL Image (BGR to RGB)
        if len(image.shape) == 3:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(image)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    image.save(filepath, format=format)


def create_collage(image_paths, output_path, cols=5, thumb_size=(128, 128)):
    """创建图片拼接图"""
    if not image_paths:
        return
    
    rows = (len(image_paths) + cols - 1) // cols
    
    # 创建画布
    collage = Image.new('RGB', 
                       (thumb_size[0] * cols, thumb_size[1] * rows),
                       (255, 255, 255))
    
    for i, img_path in enumerate(image_paths):
        if i >= cols * rows:
            break
        
        row = i // cols
        col = i % cols
        
        try:
            img = Image.open(img_path)
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            
            # 计算位置
            x = col * thumb_size[0] + (thumb_size[0] - img.width) // 2
            y = row * thumb_size[1] + (thumb_size[1] - img.height) // 2
            
            collage.paste(img, (x, y))
        except Exception as e:
            print(f"无法加载图片 {img_path}: {e}")
    
    collage.save(output_path)
    print(f"✅ 拼接图已保存: {output_path}")


# 导入 cv2 用于颜色转换
import cv2