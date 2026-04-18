import torch
import numpy as np
from torchvision import transforms
from src.config_reader import Config

class Preprocessor:
    def __init__(self):
        self.cfg = Config()
        self.transform = self.get_transforms()
    
    def get_transforms(self):
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def process(self, face_image):
        if face_image is None:
            return None
        
        # numpy array → tensor
        tensor = self.transform(face_image)
        
        # batch dimension add karo
        tensor = tensor.unsqueeze(0)
        
        return tensor