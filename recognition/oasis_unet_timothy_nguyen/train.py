"""
Defensive training script for the OASIS 2D project.

- forces local imports (so Colab doesn't import some random 'dataset' pkg)
- uses our OASIS2DSegmentation (PNG / NIfTI)
- can handle both UNet(...) and UNet2D(...) style modules
- falls back to splitting the train set if there is no test/val split
- saves best model and training curves
"""

import os
import sys
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# ---------------------------------------------------------------------
# make sure we import from THIS folder, not a pip package called 'dataset'
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataset import OASIS2DSegmentation, guess_oasis_root
import modules  # we'll pick the right class from here dynamically


# ---------------------------------------------------------------------
# Dice helper
# ---------------------------------------------------------------------
def dice_coefficient(pred_logits: torch.Tensor,
                     target: torch.Tensor,
                     num_classes: int) -> float:
    with torch.no_grad():
        pred = torch.argmax(pred_logits, dim=1)  # (B, H, W)
        dices = []
        for c in range(num_classes):
            pred_c = (pred == c)
            tgt_c = (target == c)
            inter = (pred_c & tgt_c).sum(dim=(1, 2)).float()
            denom = pred_c.sum(dim=(1, 2)).float() + tgt_c.sum(dim=(1, 2)).float()
            dice_c = torch.where(denom > 0, 2.0 * inter / denom, torch.ones_like(denom))
            dices.append(dice_c.mean().item())
        return float(np.mean(dices))


def train_one_epoch(model, loader, criterion, optimiser, device, num_classes):
    model.train()
    total_loss = 0.0

    for imgs, lbls in loader:
        imgs = imgs.to(device, dtype=torch.float32)
        lbls = lbls.to(device, dtype=torch.long)

        optimser_zero = optimiser.zero_grad
        optimser_zero()

        outputs = model(imgs)
        loss = criterion(outputs, lbls)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, dtype=torch.float32)
            lbls = lbls.to(device, dtype=torch.long)

            outputs = model(imgs)
            loss = criterion(outputs, lbls)

            total_loss += loss.item() * imgs.size(0)
            total_dice += dice_coefficient(outputs, lbls, num_classes) * imgs.size(0)

    return total_loss / len(loader.dataset), total_dice / len(loader.dataset)


def plot_curves(train_losses, val_losses, val_dices, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # loss
    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"))
    plt.close()

    # dice
    plt.figure()
    plt.plot(val_dices, label="val dice")
    plt.xlabel("epoch")
    plt.ylabel("dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dice.png"))
    plt.close()


# ---------------------------------------------------------------------
# model builder that survives different module names / signatures
# ---------------------------------------------------------------------
def build_model(num_classes: int, device: torch.device):
    """
    Try to create the UNet no matter what the student's modules.py called it.
    We'll try, in order:
      1. modules.UNet(in_channels=1, out_channels=num_classes)
      2. modules.UNet2D(in_channels=1, out_channels=num_classes)
      3. modules.UNet()           # fallback
      4. modules.UNet2D()         # fallback
    """
    # 1) UNet(...)
    if hasattr(modules, "UNet"):
        try:
            m = modules.UNet(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            # maybe UNet() has no args
            try:
                m = modules.UNet()
                return m.to(device)
            except Exception:
                pass

    # 2) UNet2D(...)
    if hasattr(modules, "UNet2D"):
        try:
            m = modules.UNet2D(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            m = modules.UNet2D()
            return m.to(device)

    raise RuntimeError("Could not find a suitable UNet / UNet2D class in modules.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="dataset root (optional)")
    parser.add_argument("--backend", type=str, default="png", choices=["png", "nifti"])
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=4)
    parser.add_argument("--save-dir", type=str, default="./trained_models/oasis_unet")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[train] Using device:", device)

    root = args.root or guess_oasis_root()
    print("[train] Using dataset root:", root)

    # ----------------- build train dataset -----------------
    train_ds = OASIS2DSegmentation(
        root=root,
        split="train",
        norm=True,
        num_classes=args.num_classes,
        backend=args.backend,
    )
    print(f"[train] train_ds: fake_mode={train_ds.fake_mode}, len={len(train_ds)}")

    # ----------------- build val dataset -------------------
    val_ds = OASIS2DSegmentation(
        root=root,
        split="test",
        norm=True,
        num_classes=args.num_classes,
        backend=args.backend,
    )
    if val_ds.fake_mode:
        # no test split → split train 90/10
        print("[train] No /test split found; splitting train into train/val")
        total_len = len(train_ds)
        val_len = max(1, int(0.1 * total_len))
        train_len = total_len - val_len
        train_ds, val_ds = random_split(train_ds, [train_len, val_len])
        # random_split returns Subset → no calculate_class_weights()
        can_calc_weights = False
    else:
        can_calc_weights = True

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

    # ----------------- model -----------------
    model = build_model(args.num_classes, device)
    print("[train] Model built:", model.__class__.__name__)

    # ----------------- loss (try class weights) -----------------
    class_weights = None
    if can_calc_weights and hasattr(train_ds, "calculate_class_weights"):
        try:
            class_weights = train_ds.calculate_class_weights().to(device)
            print("[train] Using class weights:", class_weights)
        except Exception as e:
            print("[train] Could not compute class weights:", e)
            class_weights = None

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimiser = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, "best_model.pth")

    best_val_dice = 0.0
    train_losses, val_losses, val_dices = [], [], []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
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
        improved = ""
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
            improved = " (saved)"

        t1 = time.time() - t0
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_dice={val_dice:.4f}{improved} "
            f"time={t1:.1f}s"
        )

    # plots
    plot_curves(train_losses, val_losses, val_dices, args.save_dir)
    print("[train] Best val Dice:", best_val_dice)
    print("[train] Saved to:", ckpt_path)


if __name__ == "__main__":
    main()
