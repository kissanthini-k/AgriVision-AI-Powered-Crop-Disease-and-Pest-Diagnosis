"""
Advanced Training Pipeline Module for PyTorch.

Features:
- Mixed Precision Training (AMP)
- TensorBoard & CSV Logging
- Cosine Annealing LR Scheduler with Warm Restarts
- Linear Warmup for Phase 1
- Class-Weighted Loss (handles 160 pest vs 17 disease imbalance)
- Early Stopping with restart-aware patience
- Model Checkpointing (best val acc)
- Advanced Augmentations (v2 transforms + GaussianBlur + Grayscale)
- MixUp / CutMix batch augmentation
- Gradient Clipping & Weight Decay
- Multi-architecture Support (ConvNeXt-Tiny, EfficientNet-B4, Swin-T)
"""

import os
import sys
import csv
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2
from tensorboardX import SummaryWriter
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.config import MODEL_PATH, NUM_CLASSES, IMAGENET_MEAN, IMAGENET_STD
from utils.logger import setup_logger
from models.build_model import build_model

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Early Stopping
# ---------------------------------------------------------------------------
class EarlyStopping:
    """
    Stops training when val_loss stops improving.

    patience is set high (12) so that CosineAnnealingWarmRestarts
    (T_0=5, T_mult=2 → restarts at ep 5, 15, 35) does not trigger a
    false-positive stop during the natural LR rise at each restart.
    """
    def __init__(self, patience=12, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------
def train(data_dir: str, model_name: str = "convnext_tiny", resume_ckpt: str = None):

    # ------------------------------------------------------------------
    # 1. Device Setup
    # ------------------------------------------------------------------
    logger.info("Identifying CPU/GPU for PyTorch...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        num_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU detected! Found {num_gpus} GPU(s) ({gpu_name}). Training on GPU.")
    else:
        logger.info("No GPU detected. Falling back to CPU for training.")

    # ------------------------------------------------------------------
    # 2. Data Preparation
    # ------------------------------------------------------------------
    logger.info("Setting up datasets with v2 Augmentations...")
    batch_size = 64 if "tiny" in model_name else 32
    img_size = (224, 224)
    num_workers = 0 if os.name == 'nt' else 4

    # --- Augmentation Pipelines ---
    # Train: stronger pipeline with GaussianBlur + RandomGrayscale to help
    # the minority disease classes generalise better.
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=img_size, scale=(0.7, 1.0), antialias=True),   # wider crop range
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(degrees=20),                                             # slightly more rotation
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),  # stronger jitter
        v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),                         # NEW: blur for robustness
        v2.RandomGrayscale(p=0.05),                                                # NEW: rare grayscale
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        v2.RandomErasing(p=0.15),                                                  # slightly more erasing
    ])

    val_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize(256, antialias=True),
        v2.CenterCrop(img_size),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # Load full dataset (no transform yet — applied per-split below)
    import copy
    full_dataset = datasets.ImageFolder(root=data_dir)
    val_dataset  = copy.copy(full_dataset)

    full_dataset.transform = train_transform
    val_dataset.transform  = val_transform

    # 80/20 split (same seed → reproducible)
    val_size   = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    generator  = torch.Generator().manual_seed(42)

    train_ds, _ = random_split(full_dataset, [train_size, val_size], generator=generator)
    _, val_ds   = random_split(val_dataset,  [train_size, val_size], generator=generator)

    # --- MixUp / CutMix ---
    cutmix         = v2.CutMix(num_classes=NUM_CLASSES)
    mixup          = v2.MixUp(num_classes=NUM_CLASSES)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    def collate_fn(batch):
        return cutmix_or_mixup(*torch.utils.data.default_collate(batch))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # ------------------------------------------------------------------
    # 3. Class Weights  ← KEY FIX for 160-pest / 17-disease imbalance
    # ------------------------------------------------------------------
    logger.info("Computing class weights to handle dataset imbalance...")
    all_labels    = [label for _, label in full_dataset.samples]
    class_counts  = Counter(all_labels)
    total_samples = len(all_labels)

    weights = []
    for i in range(NUM_CLASSES):
        count = class_counts.get(i, 1)          # avoid div-by-zero for unseen classes
        weights.append(total_samples / (NUM_CLASSES * count))

    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class weights computed. Min: {min(weights):.4f} | Max: {max(weights):.4f}")

    # ------------------------------------------------------------------
    # 4. Logging Setup
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    log_csv_path = os.path.join(os.path.dirname(MODEL_PATH), "training_log.csv")
    tb_writer    = SummaryWriter(log_dir=os.path.join(os.path.dirname(MODEL_PATH), "runs"))

    with open(log_csv_path, mode='w', newline='') as f:
        csv.writer(f).writerow(["phase", "epoch", "loss", "accuracy",
                                 "val_loss", "val_accuracy", "time", "lr"])

    # ------------------------------------------------------------------
    # 5. Model Setup
    # ------------------------------------------------------------------
    model = build_model(NUM_CLASSES, model_name=model_name, trainable_backbone=False)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Label smoothing + class weights — handles both soft MixUp targets and
    # the class imbalance between pests and diseases.
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, weight=class_weights)

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

    start_epoch  = 0
    best_val_acc = 0.0

    # Resume from checkpoint if provided
    if resume_ckpt and os.path.exists(resume_ckpt):
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
        ckpt = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        start_epoch  = ckpt.get('epoch', 0)
        logger.info(f"Resumed at epoch {start_epoch} with Best Val Acc: {best_val_acc:.4f}")

    # ------------------------------------------------------------------
    # Helper: save best checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(epoch, optimizer, val_acc):
        checkpoint = {
            'epoch':               epoch + 1,
            'model_state_dict':    model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc':        val_acc,
        }
        torch.save(checkpoint, MODEL_PATH.replace('.pth', '_checkpoint.pth'))
        # Pure weights file for inference (strip DataParallel wrapper if needed)
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, MODEL_PATH)
        logger.info(f"  ✓ Saved best model  Val Acc: {val_acc:.4f}")

    # ------------------------------------------------------------------
    # Helper: single epoch (train or val)
    # ------------------------------------------------------------------
    def run_epoch(loader, optimizer, scheduler, is_training=True):
        model.train() if is_training else model.eval()

        running_loss = 0.0
        correct      = 0
        total        = 0

        desc = "Training" if is_training else "Validation"
        pbar = tqdm(loader, desc=desc, leave=False,
                    colour='green' if is_training else 'blue')

        with torch.set_grad_enabled(is_training):
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)

                if is_training:
                    optimizer.zero_grad()

                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)

                if is_training:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler:
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)

                # Soft targets from MixUp → argmax for accuracy metric
                if labels.ndim > 1:
                    labels = labels.argmax(dim=1)

                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels).item()
                total   += inputs.size(0)

                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        return running_loss / total, correct / total

    # ==================================================================
    # PHASE 1 — Train head only (frozen backbone), 10 epochs + warmup
    # ==================================================================
    if start_epoch < 10:
        logger.info("=" * 60)
        logger.info("PHASE 1: Training Custom Head (Frozen Backbone)")
        logger.info("=" * 60)

        head_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer_p1 = optim.AdamW(head_params, lr=1e-3, weight_decay=1e-4)

        # Linear warmup: LR ramps from 1e-4 → 1e-3 over first 3 epochs
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer_p1, start_factor=0.1, end_factor=1.0, total_iters=3
        )

        for epoch in range(start_epoch, 10):
            start_time = time.time()

            train_loss, train_acc = run_epoch(
                train_loader, optimizer_p1, scheduler=warmup_scheduler, is_training=True
            )
            val_loss, val_acc = run_epoch(
                val_loader, optimizer_p1, scheduler=None, is_training=False
            )

            epoch_time = time.time() - start_time
            current_lr = optimizer_p1.param_groups[0]['lr']

            logger.info(
                f"Phase 1 | Epoch {epoch+1:02d}/10 | "
                f"TL: {train_loss:.4f}  TA: {train_acc:.4f} | "
                f"VL: {val_loss:.4f}  VA: {val_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )

            tb_writer.add_scalar('Loss/train',     train_loss, epoch)
            tb_writer.add_scalar('Loss/val',       val_loss,   epoch)
            tb_writer.add_scalar('Accuracy/train', train_acc,  epoch)
            tb_writer.add_scalar('Accuracy/val',   val_acc,    epoch)
            tb_writer.add_scalar('LR',             current_lr, epoch)

            with open(log_csv_path, mode='a', newline='') as f:
                csv.writer(f).writerow([
                    "phase1", epoch+1, train_loss, train_acc,
                    val_loss, val_acc, epoch_time, current_lr
                ])

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(epoch, optimizer_p1, val_acc)

    # ==================================================================
    # PHASE 2 — Fine-tune full backbone, up to 40 epochs
    # ==================================================================
    logger.info("=" * 60)
    logger.info("PHASE 2: Fine-tuning Full Backbone")
    logger.info("=" * 60)

    # Reload best Phase 1 weights before unfreezing
    best_weights_path = MODEL_PATH.replace('.pth', '_checkpoint.pth')
    if os.path.exists(best_weights_path):
        ckpt = torch.load(best_weights_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info("Loaded best Phase 1 weights before fine-tuning.")

    # Unfreeze all layers
    for param in model.parameters():
        param.requires_grad = True

    # Lower LR for fine-tuning to avoid destroying pretrained features
    optimizer_p2 = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)

    # Cosine annealing with warm restarts
    # T_0=5, T_mult=2 → LR restarts at ep 5, 15, 35 (relative to Phase 2 start)
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_p2, T_0=5, T_mult=2
    )

    # patience=12 so early stopping survives the LR rise at each warm restart
    early_stopping = EarlyStopping(patience=12, min_delta=0.001)

    phase2_start = max(0, start_epoch - 10)

    for epoch in range(phase2_start, 40):   # 40 epochs max (was 30)
        actual_epoch = epoch + 10
        start_time   = time.time()

        train_loss, train_acc = run_epoch(
            train_loader, optimizer_p2, scheduler=scheduler_p2, is_training=True
        )
        val_loss, val_acc = run_epoch(
            val_loader, optimizer_p2, scheduler=None, is_training=False
        )

        epoch_time = time.time() - start_time
        current_lr = optimizer_p2.param_groups[0]['lr']

        logger.info(
            f"Phase 2 | Epoch {actual_epoch+1:02d}/50 | "
            f"TL: {train_loss:.4f}  TA: {train_acc:.4f} | "
            f"VL: {val_loss:.4f}  VA: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        tb_writer.add_scalar('Loss/train',     train_loss,  actual_epoch)
        tb_writer.add_scalar('Loss/val',       val_loss,    actual_epoch)
        tb_writer.add_scalar('Accuracy/train', train_acc,   actual_epoch)
        tb_writer.add_scalar('Accuracy/val',   val_acc,     actual_epoch)
        tb_writer.add_scalar('LR',             current_lr,  actual_epoch)

        with open(log_csv_path, mode='a', newline='') as f:
            csv.writer(f).writerow([
                "phase2", actual_epoch+1, train_loss, train_acc,
                val_loss, val_acc, epoch_time, current_lr
            ])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(actual_epoch, optimizer_p2, val_acc)

        early_stopping(val_loss)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered — model stopped improving.")
            break

    tb_writer.close()
    logger.info(f"Training complete. Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Best model saved to: {MODEL_PATH}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CropSense AI — Training Pipeline")
    parser.add_argument(
        "--data_dir", type=str, default="data/augmented/",
        help="Path to training data directory (ImageFolder structure)"
    )
    parser.add_argument(
        "--model", type=str, default="convnext_tiny",
        choices=["convnext_tiny", "efficientnet_b4", "swin_t"],
        help="Model architecture to train"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint .pth file to resume training from"
    )
    args = parser.parse_args()

    train(args.data_dir, args.model, args.resume)