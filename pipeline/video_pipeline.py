import cv2
import torch
import numpy as np
from pipeline.base_pipeline import BasePipeline
from src.face_detector import FaceDetector
from src.preprocessor import Preprocessor
from src.model import DeepfakeDetector
from src.gradcam import GradCAM
from src.config_reader import Config

class VideoPipeline(BasePipeline):
    def __init__(self):
        self.cfg = Config()
        self.face_detector = FaceDetector()
        self.preprocessor = Preprocessor()
        self.model = DeepfakeDetector()
        self.model.eval()
        self.gradcam = GradCAM(self.model)
    
    def load_input(self, file):
        # Load video file
        cap = cv2.VideoCapture(file)
        frames = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Sample every 10th frame
            if frame_count % 10 == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    def preprocess(self, frames):
        # Process each frame
        tensors = []
        faces = []
        for frame in frames:
            face = self.face_detector.extract(frame)
            if face is None:
                continue
            tensor = self.preprocessor.process(face)
            tensors.append(tensor)
            faces.append(face)
        return tensors, faces
    
    def predict(self, data):
        tensors, faces = data
        if not tensors:
            return None
        
        predictions = []
        with torch.no_grad():
            for tensor in tensors:
                output = self.model(tensor)
                predictions.append(output.item())
        
        # Average prediction across all frames
        avg_prob = np.mean(predictions)
        label = "FAKE" if avg_prob > self.cfg.get('inference', 'threshold') else "REAL"
        
        return {
            'label': label,
            'confidence': round(avg_prob * 100, 2),
            'frames_analyzed': len(predictions),
            'faces': faces
        }
    
    def explain(self, data):
        tensors, faces = data
        if not tensors:
            return None
        
        # Generate heatmap for first face only
        heatmap = self.gradcam.generate_heatmap(tensors[0])
        overlayed = self.gradcam.overlay_heatmap(heatmap, faces[0])
        return overlayed
    
    def run(self, file):
        # Load frames
        frames = self.load_input(file)
        if not frames:
            return {'error': 'No frames extracted'}
        
        # Preprocess
        data = self.preprocess(frames)
        tensors, faces = data
        if not tensors:
            return {'error': 'No face detected in video'}
        
        # Predict
        result = self.predict(data)
        
        # Explain
        explanation = self.explain(data)
        result['heatmap'] = explanation
        
        return result