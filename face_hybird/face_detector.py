"""
人脸检测模块
"""
import cv2
import numpy as np
import os


class FaceDetector:
    """人脸检测器"""
    
    def __init__(self, min_face_size=100):
        self.min_face_size = min_face_size
        
        # 使用多个级联分类器提高检测率
        cascade_paths = [
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
            cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml'
        ]
        
        self.face_cascades = []
        for path in cascade_paths:
            if os.path.exists(path):
                self.face_cascades.append(cv2.CascadeClassifier(path))
    
    def detect_and_extract(self, image_path, target_size=(128, 128)):
        """检测并提取人脸"""
        try:
            # 读取图片
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用多个级联分类器检测
            all_faces = []
            for cascade in self.face_cascades:
                faces = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,  # 更敏感的检测
                    minNeighbors=3,     # 降低要求
                    minSize=(self.min_face_size, self.min_face_size)
                )
                if len(faces) > 0:
                    all_faces.extend(list(faces))
            
            if len(all_faces) == 0:
                return None
            
            # 选择最大的人脸
            all_faces = sorted(all_faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = all_faces[0]
            
            # 稍微扩大区域
            margin = int(min(w, h) * 0.1)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            # 提取人脸区域
            face = img[y:y+h, x:x+w]
            
            # 调整大小
            face = cv2.resize(face, target_size)
            
            # 保存提取的人脸
            face_path = image_path.replace('source_images', 'extracted_faces')
            face_path = face_path.replace('.jpg', '_face.jpg')
            os.makedirs(os.path.dirname(face_path), exist_ok=True)
            cv2.imwrite(face_path, face)
            
            return face
            
        except Exception as e:
            print(f"人脸检测失败: {e}")
            return None