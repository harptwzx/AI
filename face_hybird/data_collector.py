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
import urllib.parse


class FaceDataCollector:
    """人脸数据采集器"""
    
    def __init__(self, cache_dir="../source_images"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 使用免费的人脸图片API
        self.apis = [
            "https://randomuser.me/api/portraits/women/{}.jpg",
            "https://randomuser.me/api/portraits/men/{}.jpg",
            "https://thispersondoesnotexist.com/",
        ]
    
    def collect(self, num_images=50, source_type='random'):
        """
        采集人脸图片
        """
        print(f"开始采集 {num_images} 张真实人脸图片...")
        images = []
        
        # 方法1: 使用 RandomUser 头像API (免费，无需key)
        for i in range(num_images):
            try:
                # RandomUser 提供真实人物头像
                gender = random.choice(['women', 'men'])
                img_id = random.randint(0, 99)
                url = f"https://randomuser.me/api/portraits/{gender}/{img_id}.jpg"
                
                print(f"  下载图片 {i+1}/{num_images}: {url}")
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    # 保存图片
                    img_hash = hashlib.md5(response.content).hexdigest()
                    img_path = f"{self.cache_dir}/face_{img_hash}.jpg"
                    
                    with open(img_path, 'wb') as f:
                        f.write(response.content)
                    
                    images.append(img_path)
                    print(f"    ✅ 成功")
                else:
                    print(f"    ❌ 失败，状态码: {response.status_code}")
                    
            except Exception as e:
                print(f"    ❌ 错误: {e}")
                continue
        
        print(f"✅ 成功采集 {len(images)} 张人脸图片")
        return images