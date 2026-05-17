"""
人脸特征编码模块
"""
import numpy as np
import cv2
import hashlib


class FaceEncoder:
    """人脸特征编码器"""
    
    def __init__(self, feature_dim=128):
        self.feature_dim = feature_dim
    
    def encode(self, face_image):
        """编码人脸为特征向量"""
        if face_image is None:
            return None
        
        features = []
        
        # 颜色特征
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [16], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # 纹理特征
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        hist_mag = np.histogram(magnitude, bins=16)[0]
        features.extend(hist_mag)
        
        # 确保固定长度
        features = np.array(features)
        if len(features) > self.feature_dim:
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # 归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        # 生成哈希
        hash_val = hashlib.md5(features.tobytes()).hexdigest()
        
        return {
            'vector': features,
            'dimension': len(features),
            'hash': hash_val
        }