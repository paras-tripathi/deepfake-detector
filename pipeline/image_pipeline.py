import cv2
import torch
import numpy as np
from pipeline.base_pipeline import BasePipeline
from src.face_detector import FaceDetector
from src.preprocessor import Preprocessor
from src.model import DeepfakeDetector
from src.gradcam import GradCAM
from src.config_reader import Config

class ImagePipeline(BasePipeline):
    def __init__(self):
        self.cfg = Config()
        self.face_detector = FaceDetector()
        self.preprocessor = Preprocessor()
        self.model = DeepfakeDetector()
        self.model.eval()
        self.gradcam = GradCAM(self.model)
    
    def load_input(self, file):
        # Read image from file path or numpy array
        if isinstance(file, str):
            image = cv2.imread(file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = file
        return image
    
    def preprocess(self, image):
        # Detect and extract face
        face = self.face_detector.extract(image)
        if face is None:
            return None
        # Convert to tensor
        tensor = self.preprocessor.process(face)
        return tensor, face
    
    def predict(self, data):
        if data is None:
            return None
        tensor, face = data
        # Get prediction
        with torch.no_grad():
            output = self.model(tensor)
        prob = output.item()
        label = "FAKE" if prob > self.cfg.get('inference', 'threshold') else "REAL"
        return {
            'label': label,
            'confidence': round(prob * 100, 2),
            'face': face
        }
    
    def explain(self, data):
        if data is None:
            return None
        tensor, face = data
        # Generate GradCAM heatmap
        heatmap = self.gradcam.generate_heatmap(tensor)
        overlayed = self.gradcam.overlay_heatmap(heatmap, face)
        return overlayed
    
    def run(self, file):
        # Load
        image = self.load_input(file)
        # Preprocess
        data = self.preprocess(image)
        if data is None:
            return {'error': 'No face detected'}
        # Predict
        result = self.predict(data)
        # Explain
        explanation = self.explain(data)
        result['heatmap'] = explanation
        return result