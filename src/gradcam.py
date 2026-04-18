import torch
import numpy as np
import cv2
from src.config_reader import Config

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.cfg = Config()
        self.gradients = None
        self.activations = None
        
        # Register hooks on last conv layer
        target_layer = self.model.spatial_branch._blocks[-1]
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_backward_hook(self._save_gradients)
    
    def _save_activations(self, module, input, output):
        # Save forward pass activations
        self.activations = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        # Save backward pass gradients
        self.gradients = grad_output[0].detach()
    
    def generate_heatmap(self, image_tensor):
        # Forward pass
        self.model.eval()
        output = self.model(image_tensor)
        
        # Backward pass
        self.model.zero_grad()
        output.backward()
        
        # Compute weights from gradients
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        heatmap = (weights * self.activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        
        # Normalize heatmap to 0-1
        heatmap = heatmap.squeeze().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap
    
    def overlay_heatmap(self, heatmap, original_image):
        # Convert heatmap to color
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap), 
            cv2.COLORMAP_JET
        )
        
        # Overlay heatmap on original image
        overlayed = cv2.addWeighted(original_image, 0.6, heatmap_colored, 0.4, 0)
        
        return overlayed