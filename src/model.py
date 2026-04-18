import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from src.config_reader import Config

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = Config()
        
        # Spatial branch
        self.spatial_branch = self.build_spatial_branch()
        
        # Frequency branch
        self.frequency_branch = self.build_frequency_branch()
        
        # Classifier
        self.classifier = self.build_classifier()
    
    def build_spatial_branch(self):
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model._fc = nn.Identity()
        return model
    
    def build_frequency_branch(self):
        return nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256)
        )
    
    def build_classifier(self):
        return nn.Sequential(
            nn.Linear(1792 + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Spatial features
        spatial_feat = self.spatial_branch(x)
        
        # Frequency features
        gray = x.mean(dim=1, keepdim=True)
        freq_map = torch.fft.fft2(gray)
        freq_map = torch.log1p(torch.abs(freq_map))
        freq_feat = self.frequency_branch(freq_map)
        
        # Combine aur classify
        combined = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.classifier(combined)