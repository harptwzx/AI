"""
人脸特征编码模块
将人脸图像转换为特征向量（基因）
"""
import numpy as np
from PIL import Image
import cv2
from scipy.fft import dct
import hashlib


class FaceEncoder:
    """人脸特征编码器"""
    
    def __init__(self, feature_dim=256):
        self.feature_dim = feature_dim
    
    def encode(self, face_image):
        """
        将人脸图像编码为特征向量
        
        使用多种特征提取方法：
        1. 颜色直方图
        2. 纹理特征 (LBP)
        3. DCT 系数
        4. 边缘特征
        
        Returns:
            特征向量 (numpy array)
        """
        if face_image is None:
            return None
        
        features = []
        
        # 1. 颜色特征 (RGB直方图)
        color_features = self._extract_color_features(face_image)
        features.extend(color_features)
        
        # 2. 纹理特征 (简化版LBP)
        texture_features = self._extract_texture_features(face_image)
        features.extend(texture_features)
        
        # 3. DCT 频率特征
        dct_features = self._extract_dct_features(face_image)
        features.extend(dct_features)
        
        # 4. 几何特征（人脸关键点简化版）
        geo_features = self._extract_geometric_features(face_image)
        features.extend(geo_features)
        
        # 归一化到 [-1, 1]
        features = np.array(features)
        if len(features) > self.feature_dim:
            # 降维
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            # 填充
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # 归一化
        features = features / (np.linalg.norm(features) + 1e-8)
        
        return {
            'vector': features,
            'dimension': len(features),
            'hash': self._vector_to_hash(features)
        }
    
    def _extract_color_features(self, image):
        """提取颜色特征"""
        # 转换到 HSV 色彩空间
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        features = []
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        return features
    
    def _extract_texture_features(self, image):
        """提取纹理特征（简化的LBP）"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算梯度
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # 统计特征
        hist_mag = np.histogram(magnitude, bins=16)[0]
        hist_dir = np.histogram(direction, bins=16)[0]
        
        return list(hist_mag) + list(hist_dir)
    
    def _extract_dct_features(self, image):
        """提取 DCT 频率特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = gray.astype(np.float32) / 255.0
        
        # 应用 DCT
        dct_coeffs = dct(dct(gray.T, norm='ortho').T, norm='ortho')
        
        # 提取低频系数（前100个）
        dct_flat = dct_coeffs.flatten()[:100]
        
        return list(dct_flat)
    
    def _extract_geometric_features(self, image):
        """提取几何特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 计算图像矩
        moments = cv2.moments(gray)
        
        features = [
            moments['m00'],  # 面积
            moments['m10'] / (moments['m00'] + 1e-8),  # 重心x
            moments['m01'] / (moments['m00'] + 1e-8),  # 重心y
            moments['mu20'] / (moments['m00'] + 1e-8),  # 二阶矩
            moments['mu02'] / (moments['m00'] + 1e-8),
            moments['mu11'] / (moments['m00'] + 1e-8),
        ]
        
        return features
    
    def _vector_to_hash(self, vector):
        """将特征向量转换为哈希值"""
        vector_bytes = vector.tobytes()
        return hashlib.md5(vector_bytes).hexdigest()