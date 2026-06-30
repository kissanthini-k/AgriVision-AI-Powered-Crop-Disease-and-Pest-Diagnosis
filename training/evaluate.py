"""
Advanced Evaluation Module for PyTorch.

Evaluates a saved model and generates:
- Top-1 and Top-5 Accuracy
- F1, Precision, Recall
- Confusion Matrix (CSV and Heatmap PNG)
- Multi-class ROC Curve and AUC
- Comprehensive metrics.json
"""
import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import MODEL_PATH, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD, CLASS_NAMES
from utils.logger import setup_logger
from models.build_model import build_model

logger = setup_logger(__name__)

def evaluate_model(data_dir: str, model_name: str = "convnext_tiny", output_dir: str = "outputs/evaluation"):
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Evaluating on {device}...")
    
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        return
        
    model = build_model(NUM_CLASSES, model_name=model_name, trainable_backbone=False)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except RuntimeError:
        # Handle DataParallel wrapped weights if any
        state_dict = torch.load(MODEL_PATH, map_location=device)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        
    model = model.to(device)
    model.eval()

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transform)
    num_workers = 0 if os.name == 'nt' else 4
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    classes = dataset.classes
    if len(classes) != NUM_CLASSES:
        logger.warning(f"Dataset classes ({len(classes)}) != Config NUM_CLASSES ({NUM_CLASSES})")
    
    y_true = []
    y_pred = []
    y_scores = []
    
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    logger.info("Running inference...")
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            for inputs, labels in tqdm(loader, desc="Evaluating"):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                
                _, preds = torch.max(probs, 1)
                
                # Top-5 Accuracy calculation
                # if classes < 5, topk fails, so dynamically size k
                k = min(5, len(classes))
                _, topk_preds = probs.topk(k, dim=1)
                top5_correct += torch.sum(topk_preds == labels.unsqueeze(dim=1)).item()
                top1_correct += torch.sum(preds == labels).item()
                total += labels.size(0)
                
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_scores = np.array(y_scores)
    
    top1_acc = top1_correct / total
    top5_acc = top5_correct / total
    
    logger.info(f"Top-1 Accuracy: {top1_acc:.4f}")
    logger.info(f"Top-5 Accuracy: {top5_acc:.4f}")
    
    # 1. Classification Report
    report_dict = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=classes, zero_division=0)
    
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report_str)
        
    # 2. Confusion Matrix
    logger.info("Generating Confusion Matrix...")
    cm = confusion_matrix(y_true, y_pred)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(os.path.join(output_dir, "confusion_matrix.csv"))
    
    # Heatmap
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, cmap="Blues", xticklabels=False, yticklabels=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. ROC / AUC Calculation
    logger.info("Calculating ROC and AUC...")
    try:
        y_true_bin = label_binarize(y_true, classes=range(len(classes)))
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        # Micro-average ROC
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Plot ROC curve for micro-average
        plt.figure()
        plt.plot(fpr["micro"], tpr["micro"], label=f'Micro-average ROC (area = {roc_auc["micro"]:0.2f})', color='deeppink', linestyle=':', linewidth=4)
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        logger.warning(f"Failed to calculate ROC/AUC: {str(e)}")
        roc_auc = {"micro": 0.0}
    
    # Save Metrics JSON
    metrics = {
        "top1_accuracy": top1_acc,
        "top5_accuracy": top5_acc,
        "macro_f1": report_dict['macro avg']['f1-score'],
        "weighted_f1": report_dict['weighted avg']['f1-score'],
        "micro_auc": roc_auc.get("micro", 0.0),
        "total_samples": total
    }
    
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    logger.info(f"Metrics saved to {output_dir}/metrics.json")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/augmented/", help="Test data directory")
    parser.add_argument("--model", type=str, default="convnext_tiny", help="Model architecture")
    parser.add_argument("--out_dir", type=str, default="outputs/evaluation", help="Output directory")
    args = parser.parse_args()
    
    evaluate_model(args.data_dir, args.model, args.out_dir)
