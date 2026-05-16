"""
人脸数据采集模块
从网络自动获取包含人脸的图片
"""
import os
import requests
import random
from PIL import Image
from io import BytesIO
import hashlib
from tqdm import tqdm


class FaceDataCollector:
    """人脸数据采集器"""
    
    def __init__(self, cache_dir="../source_images"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 不同来源的搜索关键词
        self.search_queries = {
            'celebrities': [
                'celebrity face portrait',
                'actor headshot',
                'actress closeup',
                'famous person face',
                'movie star portrait'
            ],
            'random_people': [
                'person face portrait',
                'human face closeup',
                'random person headshot',
                'people portrait photography',
                'face expression'
            ],
            'anime': [
                'anime face portrait',
                'manga character face',
                'anime girl face',
                'anime boy face',
                'japanese anime face'
            ],
            'diverse': [
                'diverse ethnicities faces',
                'multicultural faces',
                'global faces portrait',
                'different age faces',
                'various expressions faces'
            ]
        }
    
    def collect(self, num_images=50, source_type='celebrities'):
        """
        采集人脸图片
        
        Args:
            num_images: 需要采集的数量
            source_type: 来源类型
        
        Returns:
            图片路径列表
        """
        print(f"开始采集 {num_images} 张 {source_type} 类型的人脸图片...")
        
        # 注意：实际实现需要使用图片API
        # 这里提供模拟版本和实际API版本
        
        # 方法1: 使用 Unsplash API (需要API key)
        # images = self._collect_from_unsplash(num_images, source_type)
        
        # 方法2: 使用 Pexels API (推荐)
        # images = self._collect_from_pexels(num_images, source_type)
        
        # 方法3: 模拟版本（用于演示）
        images = self._collect_demo(num_images)
        
        return images
    
    def _collect_from_pexels(self, num_images, source_type):
        """从 Pexels API 采集（需要API key）"""
        api_key = os.getenv('PEXELS_API_KEY')
        if not api_key:
            print("⚠️ 未设置 PEXELS_API_KEY，使用模拟数据")
            return self._collect_demo(num_images)
        
        queries = self.search_queries.get(source_type, self.search_queries['random_people'])
        images = []
        
        headers = {'Authorization': api_key}
        
        for query in queries:
            if len(images) >= num_images:
                break
                
            url = f"https://api.pexels.com/v1/search?query={query}&per_page={min(80, num_images - len(images))}&orientation=portrait"
            
            try:
                response = requests.get(url, headers=headers)
                data = response.json()
                
                for photo in data.get('photos', []):
                    img_url = photo['src']['medium']
                    img_data = requests.get(img_url).content
                    
                    # 保存图片
                    img_hash = hashlib.md5(img_data).hexdigest()
                    img_path = f"{self.cache_dir}/{img_hash}.jpg"
                    
                    with open(img_path, 'wb') as f:
                        f.write(img_data)
                    
                    images.append(img_path)
                    
                    if len(images) >= num_images:
                        break
                        
            except Exception as e:
                print(f"采集失败: {e}")
                continue
        
        return images
    
    def _collect_demo(self, num_images):
        """
        演示版本：生成简单的测试图片
        实际使用时替换为真实的API调用
        """
        from PIL import Image, ImageDraw
        import numpy as np
        
        print("⚠️ 使用演示模式，生成简单测试图片")
        print("💡 提示：配置 PEXELS_API_KEY 可获取真实人脸图片")
        
        images = []
        
        for i in range(min(num_images, 100)):
            # 创建一个简单的测试图片
            img = Image.new('RGB', (200, 200), color=(random.randint(100, 200), 
                                                       random.randint(100, 200),
                                                       random.randint(100, 200)))
            draw = ImageDraw.Draw(img)
            
            # 画一个简单的人脸轮廓
            # 脸型
            draw.ellipse([50, 50, 150, 180], outline='black', width=2)
            # 眼睛
            draw.ellipse([70, 90, 85, 105], fill='white', outline='black')
            draw.ellipse([115, 90, 130, 105], fill='white', outline='black')
            draw.ellipse([73, 95, 82, 100], fill='black')
            draw.ellipse([118, 95, 127, 100], fill='black')
            # 嘴巴
            draw.arc([80, 130, 120, 150], 0, 180, fill='black', width=2)
            
            img_path = f"{self.cache_dir}/demo_face_{i:03d}.png"
            img.save(img_path)
            images.append(img_path)
        
        return images