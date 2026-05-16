"""
人脸检测模块
使用 OpenCV 检测并提取人脸
"""
import cv2
import numpy as np
from PIL import Image
import os


class FaceDetector:
    """人脸检测器"""
    
    def __init__(self, min_face_size=100):
        self.min_face_size = min_face_size
        
        # 加载预训练的人脸检测模型
        # 使用 OpenCV 的 Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 也可以使用更准确的 DNN 模型
        # self.use_dnn = True
        # self._load_dnn_model()
    
    def detect_and_extract(self, image_path, target_size=(128, 128)):
        """
        检测并提取人脸
        
        Args:
            image_path: 图片路径
            target_size: 输出尺寸
            
        Returns:
            提取的人脸图像 (numpy array) 或 None
        """
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            if len(faces) == 0:
                return None
            
            # 选择最大的人脸
            if len(faces) > 1:
                # 按面积排序，取最大的
                faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            
            x, y, w, h = faces[0]
            
            # 稍微扩大一点区域
            margin = int(min(w, h) * 0.1)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            # 提取人脸区域
            face = img[y:y+h, x:x+w]
            
            # 调整大小
            face = cv2.resize(face, target_size)
            
            return face
            
        except Exception as e:
            print(f"人脸检测失败 {image_path}: {e}")
            return None
    
    def detect_multiple(self, image_path):
        """检测图片中的多个人脸"""
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        return faces