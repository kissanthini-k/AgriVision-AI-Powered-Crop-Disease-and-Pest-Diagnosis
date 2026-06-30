"""
Advanced Inference prediction module for PyTorch.

Handles loading the saved PyTorch model and performing classification on 
individual images, batches, or folders with Top-5 prediction support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import glob
from typing import List, Dict, Union
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.build_model import build_model
from preprocessing.preprocess import prepare_for_inference
from utils.config import NUM_CLASSES

# Cache the loaded model to avoid reloading for every request
_cached_model = None
_device = None

def load_model(model_path: str, model_name: str = "convnext_tiny") -> nn.Module:
    """
    Loads the PyTorch model state dict from disk and caches it in memory.
    """
    global _cached_model, _device
    if _cached_model is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = build_model(NUM_CLASSES, model_name=model_name, trainable_backbone=False)
            
            state_dict = torch.load(model_path, map_location=_device)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # Handle DataParallel wrapped weights if any
            new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            
            model.to(_device)
            model.eval()
            _cached_model = model
        except Exception as e:
            raise FileNotFoundError(f"Failed to load PyTorch model from {model_path}: {str(e)}")
    return _cached_model

def get_top_k(probs: torch.Tensor, class_names: list, k: int = 5) -> List[Dict]:
    """Helper to extract top-k predictions."""
    topk_probs, topk_indices = torch.topk(probs, k)
    results = []
    for p, idx in zip(topk_probs.cpu().numpy(), topk_indices.cpu().numpy()):
        results.append({
            "class": class_names[idx],
            "confidence": float(p)
        })
    return results

def predict(image_tensor: torch.Tensor, model: nn.Module, class_names: list) -> dict:
    """
    Runs model inference on a batched image tensor.
    Returns Top-5 predictions and full probability distribution.
    """
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    image_tensor = image_tensor.to(_device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)[0]
            
    probs_cpu = probs.cpu().numpy()
    class_idx = int(probs_cpu.argmax())
    confidence = float(probs_cpu[class_idx])
    
    k = min(5, len(class_names))
    top_k_preds = get_top_k(probs, class_names, k)
    
    all_scores = {class_names[i]: float(probs_cpu[i]) for i in range(len(class_names))}
    
    return {
        "class": class_names[class_idx],
        "class_idx": class_idx,
        "confidence": confidence,
        "top_5": top_k_preds,
        "all_scores": all_scores
    }

def predict_with_tta(image_path: str, model: nn.Module, class_names: list) -> dict:
    """
    Runs model inference using Test-Time Augmentation (TTA) for maximum robustness.
    Generates multiple crops and flips, runs batched inference, and averages the probabilities.
    """
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    from preprocessing.validate_image import validate
    from torchvision import transforms
    from PIL import Image
    import cv2
    from utils.config import IMAGENET_MEAN, IMAGENET_STD
    
    val_result = validate(image_path)
    if not val_result["valid"]:
        raise ValueError(f"Validation failed: {val_result['reason']}")
        
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    
    # Base transforms
    resize = transforms.Resize(256)
    center_crop = transforms.CenterCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    # Generate variations (Center Crop + Horizontal Flip)
    img_resized = resize(pil_img)
    base_crop = center_crop(img_resized)
    flipped_crop = transforms.functional.hflip(base_crop)
    
    all_crops = [base_crop, flipped_crop]
    
    # Convert all to normalized tensors
    tensors = [normalize(to_tensor(c)) for c in all_crops]
    batch_tensor = torch.stack(tensors).to(_device)  # shape: [2, 3, 224, 224]
    
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(batch_tensor)
            probs = F.softmax(logits, dim=1)  # shape: [10, NUM_CLASSES]
            
    # Average probabilities across all 10 augmentations
    avg_probs = probs.mean(dim=0)
    
    probs_cpu = avg_probs.cpu().numpy()
    class_idx = int(probs_cpu.argmax())
    confidence = float(probs_cpu[class_idx])
    
    k = min(5, len(class_names))
    top_k_preds = get_top_k(avg_probs, class_names, k)
    all_scores = {class_names[i]: float(probs_cpu[i]) for i in range(len(class_names))}
    
    return {
        "class": class_names[class_idx],
        "class_idx": class_idx,
        "confidence": confidence,
        "top_5": top_k_preds,
        "all_scores": all_scores
    }

def predict_batch(image_tensors: torch.Tensor, model: nn.Module, class_names: list) -> List[dict]:
    """
    Predicts a batch of images at once for high throughput.
    """
    global _device
    image_tensors = image_tensors.to(_device)
    
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            logits = model(image_tensors)
            probs_batch = F.softmax(logits, dim=1)
            
    results = []
    k = min(5, len(class_names))
    for probs in probs_batch:
        probs_cpu = probs.cpu().numpy()
        class_idx = int(probs_cpu.argmax())
        
        results.append({
            "class": class_names[class_idx],
            "confidence": float(probs_cpu[class_idx]),
            "top_5": get_top_k(probs, class_names, k)
        })
    return results

def predict_folder(folder_path: str, model_path: str, model_name: str, class_names: list, batch_size: int = 32) -> List[dict]:
    """
    Predicts all images in a folder using batch processing.
    """
    model = load_model(model_path, model_name)
    
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in valid_exts]
    
    all_results = []
    
    # Process in batches
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting folder"):
        batch_paths = image_paths[i:i + batch_size]
        tensors = []
        valid_paths = []
        for path in batch_paths:
            try:
                tensors.append(prepare_for_inference(path))
                valid_paths.append(path)
            except Exception as e:
                print(f"Skipping {path}: {e}")
                
        if tensors:
            batch_tensor = torch.cat(tensors, dim=0)
            batch_results = predict_batch(batch_tensor, model, class_names)
            for path, res in zip(valid_paths, batch_results):
                res["filepath"] = path
                all_results.append(res)
                
    return all_results
