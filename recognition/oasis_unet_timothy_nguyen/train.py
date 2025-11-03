
"""
Training and validation pipeline for the 2D U-Net model on OASIS PNG slices.

Overview
--------
This script ties together the dataset loader, model, optimizer, and metrics. 
It supports both UNet and UNet2D models defined in modules.py, computes 
per-class Dice metrics, and saves loss/metric curves and checkpoints. The 
pipeline is fully runnable on CPU or GPU.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# Local imports for dataset and model definitions
from dataset import OASIS2DSegmentation
import modules

# -----------------------------
# Utilities
# -----------------------------
def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    Build a U-Net (or UNet2D) from modules.py and place it on the selected device.

    Parameters
    ----------
    num_classes : int
        Number of output segmentation classes.
    device : torch.device
        Target device (CPU or CUDA).

    Returns
    -------
    nn.Module
        Instantiated model moved to the given device.

    Notes
    -----
    - The function tries both UNet and UNet2D constructors.
    - If a constructor fails due to missing args, it retries with defaults.
    """
    # Prefer the UNet class if present
    if hasattr(modules, "UNet"):
        try:
            m = modules.UNet(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            # Fallback: if constructor signature differs, call without keyword args
            m = modules.UNet().to(device)
            return m
    # Otherwise, look for UNet2D variant
    if hasattr(modules, "UNet2D"):
        try:
            m = modules.UNet2D(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            m = modules.UNet2D().to(device)
            return m
    # Raise an error if neither model is found
    raise RuntimeError("No compatible model found in modules.py (expected UNet or UNet2D).")

def dice_per_class(pred_logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute mean per-class Dice score for a batch.

    Parameters
    ----------
    pred_logits : torch.Tensor
        Raw model outputs with shape (B, C, H, W).
    target : torch.Tensor
        Ground truth integer masks with shape (B, H, W).
    num_classes : int
        Number of segmentation classes.

    Returns
    -------
    torch.Tensor
        Vector of Dice scores (C,) averaged over the batch.
    """
    pred = pred_logits.argmax(dim=1)  # convert logits to discrete predictions
    dices = []
    eps = 1e-6  # smoothing to avoid divide-by-zero
    for c in range(num_classes):
        pred_c = (pred == c).float()
        targ_c = (target == c).float()
        inter = (pred_c * targ_c).sum(dim=(1, 2))  # intersection per image
        denom = pred_c.sum(dim=(1, 2)) + targ_c.sum(dim=(1, 2)) + eps
        d = (2.0 * inter + eps) / denom  # Dice coefficient formula
        dices.append(d)
    dices = torch.stack(dices, dim=1)  # (B,C)
    return dices.mean(dim=0)  # return average per-class Dice

def save_curves(save_dir: Path, train_losses, val_losses, train_dices, val_dices):
    """
    Save training/validation loss and Dice curves as PNG plots.

    Parameters
    ----------
    save_dir : Path
        Directory to store curve images.
    train_losses, val_losses : list[float]
        Lists of loss values per epoch.
    train_dices, val_dices : list[float]
        Lists of mean Dice per epoch.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    # ---- Loss curve ----
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss.png")
    plt.close()
    # ---- Dice curve ----
    plt.figure()
    plt.plot([float(x) for x in train_dices], label="train (mean Dice)")
    plt.plot([float(x) for x in val_dices], label="val (mean Dice)")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Dice")
    plt.title("Training/Validation Mean Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "dice.png")
    plt.close()

# -----------------------------
# Training / Validation loops
# -----------------------------
def train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float]:
    """
    Perform a single training epoch.

    Returns
    -------
    Tuple[float, float]
        (mean_loss, mean_dice)
    """
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    # iterate over mini-batches
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        # compute dice for this batch
        with torch.no_grad():
            per_class = dice_per_class(logits, masks, num_classes)
            mean_dice = float(per_class.mean().item())
        total_loss += float(loss.item())
        total_dice += mean_dice
        n_batches += 1
    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)

def validate(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> Tuple[float, float]:
    """
    Evaluate the model on the validation split.

    Returns
    -------
    Tuple[float, float]
        (mean_loss, mean_dice)
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = 0
    for imgs, masks in loader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, masks)

        per_class = dice_per_class(logits, masks, num_classes)
        mean_dice = float(per_class.mean().item())

        total_loss += float(loss.item())
        total_dice += mean_dice
        n_batches += 1
    return total_loss / max(n_batches, 1), total_dice / max(n_batches, 1)

# -----------------------------
# Main entry point
# -----------------------------
def parse_args():
    """Parse CLI arguments for training configuration."""
    p = argparse.ArgumentParser(description="Train 2D UNet on canonical OASIS layout")
    p.add_argument("--root", type=str, default="./OASIS", help="Path to OASIS dataset root")
    p.add_argument("--epochs", type=int, default=12, help="Number of epochs to train")
    p.add_argument("--batch-size", type=int, default=4, help="Batch size for training/validation")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    p.add_argument("--num-classes", type=int, default=4, help="Number of segmentation classes")
    p.add_argument("--save-dir", type=str, default="./trained_models/oasis_unet", help="Directory for outputs")
    p.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader workers")
    p.add_argument("--no-class-weights", action="store_true", help="Disable inverse-frequency class weighting")
    return p.parse_args()

def main():
    """Main training control flow."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    # Initialize datasets; verify canonical folder layout
    try:
        train_ds = OASIS2DSegmentation(root=args.root, split="train",
                                       num_classes=args.num_classes, norm=True)
        val_ds = OASIS2DSegmentation(root=args.root, split="val",
                                     num_classes=args.num_classes, norm=True)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)
    # DataLoaders for train/val
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    # Initialize model
    model = build_model(args.num_classes, device)
    # Optionally compute class weights to balance loss
    if args.no_class_weights:
        class_weights = None
    else:
        try:
            class_weights = train_ds.calculate_class_weights().to(device)
        except Exception:
            class_weights = None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Initialize trackers
    best_val_dice = -1.0
    train_losses, val_losses = [], []
    train_dices, val_dices = [], []
    # ---- Training loop ----
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_dice = train_one_epoch(model, optimizer, criterion, train_loader, device, args.num_classes)
        va_loss, va_dice = validate(model, criterion, val_loader, device, args.num_classes)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_dices.append(tr_dice)
        val_dices.append(va_dice)
        print(f"[Epoch {epoch:03d}] "
              f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
              f"Train Dice: {tr_dice:.4f} | Val Dice: {va_dice:.4f}")
        # Save checkpoint when validation Dice improves
        if va_dice > best_val_dice:
            best_val_dice = va_dice
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_dice": best_val_dice,
                "num_classes": args.num_classes,
            }
            torch.save(ckpt, save_dir / "best_model.pth")
    # Save learning curves
    save_curves(save_dir, train_losses, val_losses, train_dices, val_dices)
    print(f"Training complete. Best Val Dice: {best_val_dice:.4f}. "
          f"Artifacts saved to: {save_dir}")


if __name__ == "__main__":
    main()
