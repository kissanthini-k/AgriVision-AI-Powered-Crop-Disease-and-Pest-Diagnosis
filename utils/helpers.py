"""
Shared utility functions for the Agri Disease Detector.

Includes helpers for directory creation, image loading, and base64 conversion
used in both the backend pipeline and the GUI.
"""

import os
import base64
import numpy as np
import cv2
from PIL import Image

def ensure_dir(path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.
    
    Args:
        path: The directory path to verify or create.
    """
    os.makedirs(path, exist_ok=True)

def load_image_as_array(path: str) -> np.ndarray:
    """
    Loads an image from the filesystem as a numpy array in RGB format.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Numpy array representing the image.
        
    Raises:
        FileNotFoundError: If the image cannot be read.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at path: {path}")
        
    # Read using OpenCV, convert from BGR to RGB
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Failed to read image at path: {path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb

def array_to_base64(img_array: np.ndarray) -> str:
    """
    Converts a numpy image array to a base64 encoded string.
    Useful for embedding images in HTML for PDF generation.
    
    Args:
        img_array: Numpy array of the image (RGB).
        
    Returns:
        Base64 string of the image in JPEG format.
    """
    # Convert RGB array to PIL Image
    img = Image.fromarray(img_array.astype('uint8'))
    
    import io
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"
