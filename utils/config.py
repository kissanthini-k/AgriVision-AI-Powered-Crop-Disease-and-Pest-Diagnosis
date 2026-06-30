"""
Central configuration for the Agri Disease Detector project.

This module defines constants for paths, class labels, model settings,
and other thresholds used across the pipeline.
"""

import os
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# --- Model & Training Config ---
MODEL_PATH = os.getenv("MODEL_PATH", "models/saved/best_model.pth")
MODEL_ARCHITECTURE = os.getenv("MODEL_ARCHITECTURE", "convnext_tiny")  # Updated: training with convnext_tiny
CONFIDENCE_THRESHOLD = 0.05


# ImageNet normalization statistics
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Class Definitions ---
# Dynamically loaded from data/augmented/ to ensure exact mapping with training folders.
# NUM_CLASSES and CLASS_NAMES auto-update whenever folders are added or removed.
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "augmented")
try:
    CLASS_NAMES = sorted([d for d in os.listdir(RAW_DIR) if os.path.isdir(os.path.join(RAW_DIR, d))])
except Exception:
    CLASS_NAMES = []

NUM_CLASSES = len(CLASS_NAMES)

# --- General System Config ---
PDF_OUTPUT_DIR = os.getenv("PDF_OUTPUT_DIR", "outputs/reports/")

SUPPORTED_CROPS = [
    "rice", "wheat", "cotton", "sugarcane", "tomato",
    "potato", "maize", "soybean", "chilli"
]

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "mr": "Marathi"
}

# 15 ICAR Agro-climatic zones mapping to states
ICAR_ZONES = {
    "Western Himalayan Region":          ["Jammu and Kashmir", "Himachal Pradesh", "Uttarakhand"],
    "Eastern Himalayan Region":          ["Assam", "Sikkim", "West Bengal", "Arunachal Pradesh", "Nagaland", "Manipur", "Mizoram", "Tripura", "Meghalaya"],
    "Lower Gangetic Plains Region":      ["West Bengal"],
    "Middle Gangetic Plains Region":     ["Uttar Pradesh", "Bihar"],
    "Upper Gangetic Plains Region":      ["Uttar Pradesh"],
    "Trans-Gangetic Plains Region":      ["Punjab", "Haryana", "Delhi", "Rajasthan"],
    "Eastern Plateau and Hills Region":  ["Jharkhand", "Chhattisgarh", "Odisha", "Maharashtra"],
    "Central Plateau and Hills Region":  ["Madhya Pradesh", "Rajasthan", "Uttar Pradesh"],
    "Western Plateau and Hills Region":  ["Maharashtra", "Madhya Pradesh", "Rajasthan"],
    "Southern Plateau and Hills Region": ["Andhra Pradesh", "Karnataka", "Tamil Nadu"],
    "East Coast Plains and Hills Region":["Odisha", "Andhra Pradesh", "Tamil Nadu", "Puducherry"],
    "West Coast Plains and Ghats Region":["Tamil Nadu", "Kerala", "Goa", "Karnataka", "Maharashtra"],
    "Gujarat Plains and Hills Region":   ["Gujarat"],
    "Western Dry Region":                ["Rajasthan"],
    "The Islands Region":                ["Andaman and Nicobar Islands", "Lakshadweep"]
}