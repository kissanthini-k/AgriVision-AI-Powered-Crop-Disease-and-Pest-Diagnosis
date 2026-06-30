"""
Image validation module.

Provides functions to check if an uploaded image meets the requirements:
valid format, minimum resolution, and sufficient sharpness (blur detection).
"""

import os
import cv2
import numpy as np

def check_format(path: str) -> bool:
    """
    Checks if the image has a valid file extension.
    
    Args:
        path: Path to the image file.
        
    Returns:
        True if format is .jpg, .jpeg, or .png, False otherwise.
    """
    ext = os.path.splitext(path)[1].lower()
    return ext in [".jpg", ".jpeg", ".png"]

def check_resolution(img: np.ndarray, min_size: int = 32) -> bool:
    """
    Checks if the image resolution is at least min_size x min_size.
    
    Args:
        img: Numpy array of the image.
        min_size: Minimum width and height.
        
    Returns:
        True if resolution is sufficient, False otherwise.
    """
    h, w = img.shape[:2]
    return h >= min_size and w >= min_size

def check_blur(img: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Detects blur by computing the variance of the Laplacian.
    
    Args:
        img: Numpy array of the image (BGR or RGB).
        threshold: Minimum variance to be considered sharp.
        
    Returns:
        True if image is sharp enough, False if blurry.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance >= threshold

def validate(path: str) -> dict:
    """
    Runs all validation checks on an image.
    
    Args:
        path: Path to the image file.
        
    Returns:
        A dictionary containing "valid" (bool) and "reason" (str).
    """
    if not os.path.exists(path):
        return {"valid": False, "reason": "File does not exist."}
        
    if not check_format(path):
        return {"valid": False, "reason": "Invalid format. Only JPG/PNG allowed."}
        
    img = cv2.imread(path)
    if img is None:
        return {"valid": False, "reason": "Cannot read image file."}
        
    # Check resolution
    if not check_resolution(img):
        return {"valid": False, "reason": "Resolution too low. Minimum 32x32px required."}
        
    # Check blur (lowered threshold to 20.0 for macro shots with blurred backgrounds)
    if not check_blur(img, threshold=20.0):
        return {"valid": False, "reason": "Image is too blurry. Please capture a clearer photo."}
        
    return {"valid": True, "reason": "Valid"}
