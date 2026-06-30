"""
Model builder module for PyTorch.

Constructs high-performance architectures (ConvNeXt, EfficientNet, Swin)
tailored for crop and pest disease classification.
"""

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, ConvNeXt_Tiny_Weights,
    efficientnet_b4, EfficientNet_B4_Weights,
    swin_t, Swin_T_Weights
)

def build_model(num_classes: int, model_name: str = "convnext_tiny", trainable_backbone: bool = False) -> nn.Module:
    """
    Builds the transfer learning model with the specified backbone.
    
    Args:
        num_classes: Number of output classes.
        model_name: "convnext_tiny", "efficientnet_b4", or "swin_t".
        trainable_backbone: If False, freezes the backbone weights.
        
    Returns:
        A PyTorch nn.Module.
    """
    if model_name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = convnext_tiny(weights=weights)
        in_features = model.classifier[2].in_features
        
        # Freeze backbone
        if not trainable_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # Custom Head with strong dropout
        model.classifier = nn.Sequential(
            model.classifier[0],  # LayerNorm2d
            model.classifier[1],  # Flatten
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "efficientnet_b4":
        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights)
        in_features = model.classifier[1].in_features
        
        if not trainable_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
    elif model_name == "swin_t":
        weights = Swin_T_Weights.IMAGENET1K_V1
        model = swin_t(weights=weights)
        in_features = model.head.in_features
        
        if not trainable_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        model.head = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
        
    return model
