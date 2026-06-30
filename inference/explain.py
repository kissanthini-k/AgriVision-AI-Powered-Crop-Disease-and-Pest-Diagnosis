"""
Explainability Module.

Generates Grad-CAM and Saliency maps to explain model predictions visually.
"""

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from inference.predict import load_model
from utils.config import NUM_CLASSES

def get_target_layer(model: torch.nn.Module, model_name: str):
    """Finds the last convolutional layer for Grad-CAM based on the architecture."""
    if model_name == "convnext_tiny":
        # ConvNeXt features ends with a norm layer, the last conv block is features[7][-1]
        try:
            return model.features[7][-1].block[5] # The last depthwise conv or point-wise conv
        except:
            return model.features[7][-1]
    elif model_name == "efficientnet_b4":
        return model.features[-1]
    elif model_name == "swin_t":
        # Swin doesn't use standard Conv layers for Grad-CAM easily without reshaping
        raise NotImplementedError("Grad-CAM for Swin Transformer is not supported natively via this method.")
    else:
        return list(model.children())[-2]

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx=None):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(x)
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1).item()
            
        score = output[:, class_idx]
        score.backward(retain_graph=True)
        
        # Global average pooling on gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        
        # ReLU on CAM
        cam = F.relu(cam)
        
        # Normalize between 0 and 1
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.cpu().detach().numpy()

def generate_gradcam(image_tensor: torch.Tensor, model_path: str, model_name: str, class_idx: int = None):
    """Generates a Grad-CAM heatmap tensor."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, model_name)
    model.to(device)
    
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)
    
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)
    
    cam = grad_cam(image_tensor, class_idx)
    return cam

def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlays the heatmap onto the original BGR image."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    # original_img is assumed to be BGR
    overlay = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
    return overlay

def generate_saliency_map(image_tensor: torch.Tensor, model_path: str, model_name: str, class_idx: int = None):
    """Generates a pixel-level saliency map."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, model_name)
    model.to(device)
    
    image_tensor = image_tensor.to(device)
    image_tensor.requires_grad_(True)
    
    model.eval()
    model.zero_grad()
    
    output = model(image_tensor)
    if class_idx is None:
        class_idx = torch.argmax(output, dim=1).item()
        
    score = output[:, class_idx]
    score.backward()
    
    saliency, _ = torch.max(image_tensor.grad.data.abs(), dim=1)
    saliency = saliency.squeeze().cpu().numpy()
    
    # Normalize
    saliency -= saliency.min()
    saliency /= saliency.max()
    
    return saliency

if __name__ == "__main__":
    print("Explainability module ready. Use generate_gradcam() or generate_saliency_map() in your scripts.")
