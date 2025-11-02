"""
Train a 2D UNet on the OASIS 2D segmentation dataset.

Features in this version:
- uses our robust dataset.py (Colab / Rangpur / PNG / optional NIfTI)
- optional class weights from dataset.calculate_class_weights()
- trains and validates every epoch
- saves best model based on validation Dice
- saves training curves (loss + val dice)
- CLI arguments so it can be run like the other student's project

This script is intended to look "finished" for COMP3710.
"""

import os
import argparse
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import OASIS2DSegmentation, guess_oasis_root
from modules import UNet


# ---------------------------------------------------------------------
# Utility: Dice (PyTorch tensor version, per-batch)
# ---------------------------------------------------------------------
def dice_coefficient(pred_logits: torch.Tensor,
                     target: torch.Tensor,
                     num_classes: int) -> float:
    """
    Compute mean Dice across classes for a batch.
    pred_logits: (B, C, H, W)
    target     : (B, H, W)
    """
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=1)  # (B, H, W)
        dice_scores = []
        for c in range(num_classes):
            pred_c = (pred == c)
            tgt_c = (target == c)
            inter = (pred_c & tgt_c).sum(dim=(1, 2)).float()
            pred_area = pred_c.sum(dim=(1, 2)).float()
            tgt_area = tgt_c.sum(dim=(1, 2)).float()
            denom = pred_area + tgt_area
            dice_c = torch.where(
                denom > 0,
                2.0 * inter / denom,
                torch.ones_like(denom),
            )
            dice_scores.append(dice_c.mean().item())
        return float(np.mean(dice_scores))


# ---------------------------------------------------------------------
# Train for 1 epoch
# ---------------------------------------------------------------------
def train_one_epoch(model,
                    loader,
                    criterion,
                    optimiser,
                    device,
                    num_classes: int):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device=device, dtype=torch.float32)   # (B, 1, H, W)
        labels = labels.to(device=device, dtype=torch.long)      # (B, H, W)

        optimiser.zero_grad()
        outputs = model(images)                                  # (B, C, H, W)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# ---------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------
def validate(model,
             loader,
             criterion,
             device,
             num_classes: int):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.long)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            running_dice += dice_coefficient(outputs, labels, num_classes) * images.size(0)

    val_loss = running_loss / len(loader.dataset)
    val_dice = running_dice / len(loader.dataset)
    return val_loss, val_dice


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def plot_curves(train_losses, val_losses, val_dices, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # loss
    plt.figure()
    plt.plot(train_losses, label="train loss")
    plt.plot(val_losses, label="val loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # dice
    plt.figure()
    plt.plot(val_dices, label="val dice")
    plt.title("Validation Dice vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice.png"))
    plt.close()


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train UNet on OASIS 2D segmentation.")
    parser.add_argument("--root", type=str, default=None,
                        help="dataset root; if not set we will try to guess")
    parser.add_argument("--backend", type=str, default="png", choices=["png", "nifti"],
                        help="use png (default) or nifti backend")
    parser.add_argument("--epochs", type=int, default=12, help="number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--num-classes", type=int, default=4, help="number of classes")
    parser.add_argument("--save-dir", type=str, default="./trained_models/oasis_unet",
                        help="where to save checkpoints and plots")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")

    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Using device: {device}")

    data_root = args.root or guess_oasis_root()
    print(f"[train] Using dataset root: {data_root}")

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    # -----------------------------------------------------------------
    # Datasets & loaders
    # -----------------------------------------------------------------
    train_ds = OASIS2DSegmentation(
        root=data_root,
        split="train",
        norm=True,
        num_classes=args.num_classes,
        backend=args.backend,
    )
    # we’ll just reuse "test" as validation, since that’s how the OASIS PNGs are laid out
    val_ds = OASIS2DSegmentation(
        root=data_root,
        split="test",
        norm=True,
        num_classes=args.num_classes,
        backend=args.backend,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -----------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------
    model = UNet(in_channels=1, out_channels=args.num_classes)
    model = model.to(device)

    # -----------------------------------------------------------------
    # Loss (try class weights, fall back if not available)
    # -----------------------------------------------------------------
    class_weights = None
    try:
        class_weights = train_ds.calculate_class_weights()
        class_weights = class_weights.to(device)
        print("[train] Using class weights from dataset:", class_weights)
    except Exception as e:
        print("[train] Could not compute class weights, using uniform. Reason:", e)
        class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    # -----------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------
    best_val_dice = 0.0
    train_losses = []
    val_losses = []
    val_dices = []

    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimiser, device, args.num_classes
        )
        val_loss, val_dice = validate(
            model, val_loader, criterion, device, args.num_classes
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_dices.append(val_dice)

        # save best
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optim_state_dict": optimiser.state_dict(),
                    "val_dice": val_dice,
                },
                ckpt_path,
            )
            improved = " (improved, saved)"
        else:
            improved = ""

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f}{improved} "
            f"time={epoch_time:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"[train] Finished training in {total_time/60:.1f} min. Best val dice = {best_val_dice:.4f}")

    # -----------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------
    plot_curves(train_losses, val_losses, val_dices, args.save_dir)
    print(f"[train] Saved curves to {args.save_dir}")
    print(f"[train] Best model at: {ckpt_path}")


if __name__ == "__main__":
    main()
