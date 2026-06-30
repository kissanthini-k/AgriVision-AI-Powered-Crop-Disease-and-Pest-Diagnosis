"""
Grad-CAM heatmap generation module for PyTorch.

Implements Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize
which regions of the leaf the model is focusing on using PyTorch Hooks.
"""

import numpy as np
import cv2
import torch

class PyTorchGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
        
    def __call__(self, x, class_idx):
        # Temporarily enable gradient computation
        was_training = self.model.training
        self.model.eval()
        
        with torch.enable_grad():
            self.model.zero_grad()
            output = self.model(x)
            
            # Create a one-hot tensor for the target class
            one_hot = torch.zeros((1, output.size()[-1]), dtype=torch.float32, device=x.device)
            one_hot[0][class_idx] = 1
            
            # Backward pass
            output.backward(gradient=one_hot, retain_graph=True)
            
        # Restore training state
        if was_training:
            self.model.train()
            
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)
            
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Global average pooling on gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weight the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize
        cam = cam - np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam = cam / cam_max
            
        # Resize to input size
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        
        return cam

def generate_gradcam(model: torch.nn.Module, image_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap for a specific class.
    
    Args:
        model: The trained PyTorch model.
        image_tensor: Preprocessed input image tensor of shape (1, C, H, W).
        class_idx: The index of the predicted class.
        
    Returns:
        A normalized heatmap as a 2D numpy array of shape (224, 224) in [0, 1].
    """
    # EfficientNetB4 last feature block
    try:
        target_layer = model.features[-1]
    except Exception:
        return np.zeros((224, 224))
        
    # Get model device and move tensor
    device = next(model.parameters()).device
    
    # Enable gradients for input tensor
    image_tensor = image_tensor.clone().detach().to(device).requires_grad_(True)
    
    grad_cam = PyTorchGradCAM(model, target_layer)
    heatmap = grad_cam(image_tensor, class_idx)
    
    return heatmap

def overlay_heatmap(original_img: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Overlays the Grad-CAM heatmap on the original image.
    
    Args:
        original_img: RGB original image array (H, W, 3).
        heatmap: 2D array heatmap [0, 1].
        alpha: Blending weight for the heatmap.
        colormap: OpenCV colormap code.
        
    Returns:
        BGR image array with blended heatmap.
    """
    # Resize original image to match heatmap if needed
    h_img = cv2.resize(original_img, (224, 224))
    
    # Scale heatmap to [0, 255] and apply colormap
    heatmap_scaled = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, colormap)
    
    # Convert original to BGR for OpenCV blending
    h_img_bgr = cv2.cvtColor(h_img, cv2.COLOR_RGB2BGR)
    
    # Blend images
    blended = cv2.addWeighted(heatmap_colored, alpha, h_img_bgr, 1 - alpha, 0)
    return blended
