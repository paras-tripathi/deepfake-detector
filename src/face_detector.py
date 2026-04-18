import cv2
import numpy as np
from facenet_pytorch import MTCNN
from src.config_reader import Config

class FaceDetector:
    def __init__(self):
        self.cfg = Config()
        self.detector = MTCNN()
    
    def detect(self, image):
        results = self.detector.detect_faces(image)
        if not results:
           return None
        return results[0]  # Return the first detected face
    
    def align(self, image, keypoints):
        left_eye = keypoints['left_eye']
        right_eye = keypoints['right_eye']  # Implementation for face alignment goes here
        
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        center = (image.shape[1] // 2, image.shape[0] // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        return aligned
        

    def extract(self, image):
        face_data = self.detect(image)
    
        if face_data is None:
            return None
    
        x, y, w, h = face_data['box']
        keypoints = face_data['keypoints']
    
        aligned = self.align(image, keypoints)
    
        pad = 20
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)
    
        cropped = aligned[y1:y2, x1:x2]
    
        resized = cv2.resize(cropped, (224, 224))
    
        return resized
        