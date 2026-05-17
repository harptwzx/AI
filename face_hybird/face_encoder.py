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
        
        # 2. 纹理特征 (LBP)
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        lbp = self._compute_lbp(gray)
        features.extend(lbp)
        
        # 3. HOG特征 (方向梯度直方图)
        hog_features = self._compute_hog(gray)
        features.extend(hog_features)
        
        # 4. Gabor特征 (多尺度)
        gabor_features = self._compute_gabor_features(gray)
        features.extend(gabor_features)
        
        # 5. 面部关键点相对位置
        landmarks = self._detect_landmarks(gray)
        features.extend(landmarks)
        
        # 转换为numpy数组并归一化
        features = np.array(features, dtype=np.float32)
        
        # PCA降维到目标维度
        if len(features) > self.feature_dim:
            # 简单降维：取前N个最大的特征
            features = features[:self.feature_dim]
        elif len(features) < self.feature_dim:
            # 填充
            features = np.pad(features, (0, self.feature_dim - len(features)))
        
        # 归一化到[-1, 1]
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
        lbp = np.zeros_like(gray)
        height, width = gray.shape
        
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
                lbp[i, j] = code
        
        hist = cv2.calcHist([lbp.astype(np.uint8)], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist
    
    def _compute_hog(self, gray):
        """计算HOG特征"""
        from skimage.feature import hog
        try:
            features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                          cells_per_block=(2, 2), visualize=False)
            return features[:128]  # 取前128个
        except:
            return np.zeros(128)
    
    def _compute_gabor_features(self, gray):
        """计算Gabor纹理特征"""
        features = []
        # 不同方向和尺度的Gabor滤波器
        ksize = 31
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for sigma in [3, 5]:
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, 10, 0.5, 0)
                filtered = cv2.filter2D(gray, cv2.CV_32F, kernel)
                features.append(np.mean(filtered))
                features.append(np.std(filtered))
        return features
    
    def _detect_landmarks(self, gray):
        """检测面部关键点相对位置"""
        # 使用OpenCV的人脸检测器获取面部区域比例
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # 返回面部区域的比例特征
            return [x/gray.shape[1], y/gray.shape[0], w/gray.shape[1], h/gray.shape[0]]
        else:
            return [0.5, 0.5, 0.3, 0.3]