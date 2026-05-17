"""
人脸特征编码模块 - 使用真实人脸特征
"""
import numpy as np
import cv2
import hashlib


class FaceEncoder:
    """人脸特征编码器 - 提取真实人脸特征"""
    
    def __init__(self, feature_dim=256):
        self.feature_dim = feature_dim
    
    def encode(self, face_image):
        """从真实人脸图像提取特征向量"""
        if face_image is None:
            return None
        
        features = []
        
        # 1. 颜色直方图特征 (RGB和HSV)
        for channel in range(3):
            hist = cv2.calcHist([face_image], [channel], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # HSV色彩空间
        hsv = cv2.cvtColor(face_image, cv2.COLOR_BGR2HSV)
        for channel in range(3):
            hist = cv2.calcHist([hsv], [channel], None, [32], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            features.extend(hist)
        
        # 2. 纹理特征 (简化的LBP，不依赖skimage)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lbp = self._compute_lbp(gray)
        features.extend(lbp)
        
        # 3. 梯度方向直方图 (简化版HOG)
        grad_features = self._compute_gradient_features(gray)
        features.extend(grad_features)
        
        # 4. 面部区域特征
        region_features = self._compute_region_features(gray)
        features.extend(region_features)
        
        # 5. 统计特征
        stat_features = self._compute_statistical_features(gray)
        features.extend(stat_features)
        
        # 转换为numpy数组并归一化
        features = np.array(features, dtype=np.float32)
        
        # 降维到目标维度
        if len(features) > self.feature_dim:
            # 均匀采样
            indices = np.linspace(0, len(features)-1, self.feature_dim, dtype=int)
            features = features[indices]
        elif len(features) < self.feature_dim:
            # 填充
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # 归一化到[-1, 1]
        if np.std(features) > 1e-6:
            features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        features = np.clip(features, -1, 1)
        
        # 生成哈希
        hash_val = hashlib.md5(features.tobytes()).hexdigest()
        
        return {
            'vector': features,
            'dimension': len(features),
            'hash': hash_val
        }
    
    def _compute_lbp(self, gray):
        """计算LBP纹理特征"""
        height, width = gray.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray[i, j]
                code = 0
                code |= (gray[i-1, j-1] >= center) << 7
                code |= (gray[i-1, j] >= center) << 6
                code |= (gray[i-1, j+1] >= center) << 5
                code |= (gray[i, j+1] >= center) << 4
                code |= (gray[i+1, j+1] >= center) << 3
                code |= (gray[i+1, j] >= center) << 2
                code |= (gray[i+1, j-1] >= center) << 1
                code |= (gray[i, j-1] >= center) << 0
                lbp[i-1, j-1] = code
        
        hist = cv2.calcHist([lbp], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def _compute_gradient_features(self, gray):
        """计算梯度特征"""
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        
        # 梯度直方图
        mag_hist = np.histogram(magnitude.flatten(), bins=16)[0]
        angle_hist = np.histogram(angle.flatten(), bins=16)[0]
        
        return list(mag_hist) + list(angle_hist)
    
    def _compute_region_features(self, gray):
        """计算区域特征"""
        h, w = gray.shape
        features = []
        
        # 将图像分成4x4区域
        rows, cols = 4, 4
        for i in range(rows):
            for j in range(cols):
                region = gray[i*h//rows:(i+1)*h//rows, j*w//cols:(j+1)*w//cols]
                features.append(np.mean(region))
                features.append(np.std(region))
        
        return features
    
    def _compute_statistical_features(self, gray):
        """计算统计特征"""
        features = []
        
        # 基本统计量
        features.append(np.mean(gray))
        features.append(np.std(gray))
        features.append(np.median(gray))
        
        # 百分位数
        for p in [10, 25, 50, 75, 90]:
            features.append(np.percentile(gray, p))
        
        # 偏度和峰度
        mean = np.mean(gray)
        std = np.std(gray)
        if std > 0:
            skew = np.mean(((gray - mean) / std) ** 3)
            kurt = np.mean(((gray - mean) / std) ** 4) - 3
            features.append(skew)
            features.append(kurt)
        
        return features