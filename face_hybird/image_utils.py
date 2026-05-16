"""
图像处理工具函数
"""
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def save_image(image, filepath, format='PNG'):
    """保存图像"""
    if isinstance(image, np.ndarray):
        # 转换 numpy array 到 PIL Image
        if len(image.shape) == 3:
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image = Image.fromarray(image)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    image.save(filepath, format=format)


def create_collage(image_paths, output_path, cols=5, thumb_size=(128, 128)):
    """
    创建图片拼接图
    
    Args:
        image_paths: 图片路径列表
        output_path: 输出路径
        cols: 列数
        thumb_size: 缩略图尺寸
    """
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


def add_text_to_image(image, text, position='bottom'):
    """给图片添加文字标注"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(image)
    
    # 使用默认字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    if position == 'bottom':
        y = image.height - 30
        x = 10
    else:
        y = 10
        x = 10
    
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    
    return image