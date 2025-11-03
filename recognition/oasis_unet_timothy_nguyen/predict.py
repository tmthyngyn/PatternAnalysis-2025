# predict.py
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import OASIS2DSegmentation
import modules


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    if hasattr(modules, "UNet"):
        try:
            m = modules.UNet(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            m = modules.UNet().to(device)
            return m

    if hasattr(modules, "UNet2D"):
        try:
            m = modules.UNet2D(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            m = modules.UNet2D().to(device)
            return m

    raise RuntimeError("No compatible model found in modules.py (expected UNet or UNet2D).")


def parse_args():
    p = argparse.ArgumentParser(description="Predict/visualise using trained UNet on OASIS")
    p.add_argument("--root", type=str, default="./OASIS", help="Path to OASIS/ canonical tree")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--ckpt", type=str, default="trained_models/oasis_unet/best_model.pth")
    p.add_argument("--out", type=str, default="outputs/prediction_example.png")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0, help="Dataset index to visualise")
    return p.parse_args()


def load_checkpoint(model: nn.Module, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    return ckpt


@torch.no_grad()
def predict_one(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    img: (1,1,H,W) or (1,H,W) -> ensures batch dimension
    returns: (H,W) argmax mask
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # (1,1,H,W)
    logits = model(img)  # (1,C,H,W)
    pred = logits.argmax(dim=1)[0]  # (H,W)
    return pred.cpu()


def render_triplet(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, save_path: Path):
    """
    img:  (H,W) float32 (z-scored)
    gt:   (H,W) int
    pred: (H,W) int
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, interpolation="nearest")
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, interpolation="nearest")
    plt.title("Prediction")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset (enforced canonical layout)
    try:
        ds = OASIS2DSegmentation(root=args.root, split=args.split,
                                 num_classes=args.num_classes, norm=True)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)

    if len(ds) == 0:
        print(f"No data found in split '{args.split}' under {args.root}.")
        sys.exit(2)

    idx = max(0, min(args.index, len(ds) - 1))
    img_t, gt_t = ds[idx]  # img: (1,H,W) float32, gt: (H,W) long

    # Build/load model
    model = build_model(args.num_classes, device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(2)

    load_checkpoint(model, ckpt_path)
    model.eval()

    # Predict
    img_in = img_t.unsqueeze(0).to(device)  # (1,1,H,W)
    pred_t = predict_one(model, img_in)

    # Render
    img_np = img_t.squeeze(0).cpu().numpy()
    gt_np = gt_t.cpu().numpy()
    pred_np = pred_t.cpu().numpy()

    render_triplet(img_np, gt_np, pred_np, Path(args.out))
    print(f"Saved visualisation to: {args.out}")


if __name__ == "__main__":
    main()
