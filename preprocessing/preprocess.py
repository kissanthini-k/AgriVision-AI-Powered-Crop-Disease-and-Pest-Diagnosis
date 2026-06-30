"""
Image preprocessing module for PyTorch.

Handles resizing, ImageNet normalization, and tensor preparation
for model inference.
"""

import cv2
import sys
import os
import torch
from torchvision import transforms
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import IMAGENET_MEAN, IMAGENET_STD
from preprocessing.validate_image import validate

def prepare_for_inference(path: str) -> torch.Tensor:
    """
    Validates, loads, resizes, normalizes, and prepares an image tensor
    ready for PyTorch model prediction.
    
    Args:
        path: Path to the image file.
        
    Returns:
        PyTorch Tensor of shape (1, 3, 224, 224) ready for inference.
        
    Raises:
        ValueError: If validation fails.
    """
    val_result = validate(path)
    if not val_result["valid"]:
        raise ValueError(f"Validation failed: {val_result['reason']}")
        
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image for torchvision transforms
    pil_img = Image.fromarray(img_rgb)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    tensor = transform(pil_img)
    batched_tensor = tensor.unsqueeze(0)  # Add batch dimension [1, C, H, W]
    
    return batched_tensor
