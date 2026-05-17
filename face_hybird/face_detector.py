"""
人脸检测模块
"""
import cv2
import os


class FaceDetector:
    """人脸检测器"""
    
    def __init__(self, min_face_size=80):
        self.min_face_size = min_face_size
        
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # 备用人脸检测器
        alt_path = cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        self.alt_cascade = cv2.CascadeClassifier(alt_path)
    
    def detect_and_extract(self, image_path, target_size=(128, 128)):
        """检测并提取人脸"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 使用主检测器
            faces = self.face_cascade.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=5,
                minSize=(self.min_face_size, self.min_face_size)
            )
            
            # 如果没检测到，使用备选检测器
            if len(faces) == 0:
                faces = self.alt_cascade.detectMultiScale(
                    gray, scaleFactor=1.05, minNeighbors=5,
                    minSize=(self.min_face_size, self.min_face_size)
                )
            
            if len(faces) == 0:
                return None
            
            # 选择最大的人脸
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            x, y, w, h = faces[0]
            
            # 扩大区域
            margin = int(min(w, h) * 0.15)
            x = max(0, x - margin)
            y = max(0, y - margin)
            w = min(img.shape[1] - x, w + 2 * margin)
            h = min(img.shape[0] - y, h + 2 * margin)
            
            # 提取并调整大小
            face = img[y:y+h, x:x+w]
            face = cv2.resize(face, target_size)
            
            return face
            
        except Exception as e:
            return None