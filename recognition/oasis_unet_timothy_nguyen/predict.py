
"""
Prediction and visualisation for trained 2D U-Net on OASIS PNG slices.

Overview
--------
- Loads dataset (OASIS2DSegmentation) in PNG format.
- Rebuilds model from modules.py and loads the saved checkpoint.
- Runs inference on a selected image slice.
- Saves a side-by-side figure showing input, ground truth, and prediction.

This script helps verify that the model produces reasonable segmentations after training.
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# Local imports
from dataset import OASIS2DSegmentation
import modules


def build_model(num_classes: int, device: torch.device) -> nn.Module:
    """
    Builds a UNet or UNet2D model from modules.py and sends it to the target device.

    Parameters
    ----------
    num_classes : int
        Number of output segmentation classes.
    device : torch.device
        Device to move the model to (CPU or CUDA).

    Returns
    -------
    nn.Module
        The instantiated model placed on the given device.
    """
    if hasattr(modules, "UNet"):
        try:
            # Attempt to construct model using keyword args
            m = modules.UNet(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            # Fallback for constructors that take no arguments
            m = modules.UNet().to(device)
            return m
    if hasattr(modules, "UNet2D"):
        try:
            m = modules.UNet2D(in_channels=1, out_channels=num_classes)
            return m.to(device)
        except TypeError:
            m = modules.UNet2D().to(device)
            return m
    # If neither UNet nor UNet2D exists, raise a runtime error
    raise RuntimeError("No compatible model found in modules.py (expected UNet or UNet2D).")

def parse_args():
    """
    Parse command-line arguments for inference and visualization.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for input/output paths and options.
    """
    p = argparse.ArgumentParser(description="Predict/visualise using trained UNet on OASIS")
    p.add_argument("--root", type=str, default="./OASIS", help="Path to OASIS/ canonical tree")
    p.add_argument("--num-classes", type=int, default=4)
    p.add_argument("--ckpt", type=str, default="trained_models/oasis_unet/best_model.pth")
    p.add_argument("--out", type=str, default="outputs/prediction_example.png")
    p.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    p.add_argument("--index", type=int, default=0, help="Dataset index to visualise")
    return p.parse_args()

def load_checkpoint(model: nn.Module, ckpt_path: Path):
    """
    Loads a saved model checkpoint into the given model.

    Parameters
    ----------
    model : nn.Module
        Instantiated U-Net model.
    ckpt_path : Path
        Path to the .pth checkpoint file.

    Returns
    -------
    dict
        The checkpoint dictionary (contains model_state, optimizer_state, etc.).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")  # Load checkpoint to CPU by default
    state = ckpt.get("model_state", ckpt)  # Extract model state dict if wrapped
    model.load_state_dict(state, strict=False)  # Load weights into the model
    return ckpt


def predict_one(model: nn.Module, img: torch.Tensor) -> torch.Tensor:
    """
    Perform forward pass on a single image tensor.

    Parameters
    ----------
    model : nn.Module
        Trained U-Net model.
    img : torch.Tensor
        Input tensor of shape (1,1,H,W) or (1,H,W).

    Returns
    -------
    torch.Tensor
        Predicted segmentation mask of shape (H,W) with integer class labels.
    """
    if img.ndim == 3:
        img = img.unsqueeze(0)  # Ensure batch dimension -> (1,1,H,W)
    logits = model(img)  # Forward pass -> raw logits (1,C,H,W)
    pred = logits.argmax(dim=1)[0]  # Convert to predicted label map (H,W)
    return pred.cpu()  # Return prediction on CPU for visualization

def render_triplet(img: np.ndarray, gt: np.ndarray, pred: np.ndarray, save_path: Path):
    """
    Render a triplet of input image, ground-truth, and prediction.

    Parameters
    ----------
    img : np.ndarray
        Input grayscale image (H,W).
    gt : np.ndarray
        Ground truth label mask (H,W).
    pred : np.ndarray
        Predicted label mask (H,W).
    save_path : Path
        Path where the resulting visualization will be saved.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # Create a 3-panel matplotlib figure
    plt.figure(figsize=(12, 4))
    # Panel 1: Input grayscale image
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap="gray")
    plt.title("Input")
    plt.axis("off")
    # Panel 2: Ground truth segmentation
    plt.subplot(1, 3, 2)
    plt.imshow(gt, interpolation="nearest")
    plt.title("Ground Truth")
    plt.axis("off")
    # Panel 3: Model prediction
    plt.subplot(1, 3, 3)
    plt.imshow(pred, interpolation="nearest")
    plt.title("Prediction")
    plt.axis("off")
    # Save visualization
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    """
    Entry point for inference and visualization.

    Steps:
    ------
    1. Parse command-line arguments.
    2. Load the OASIS dataset (PNG backend).
    3. Rebuild the model and load checkpoint weights.
    4. Perform prediction on one example.
    5. Render and save a visualization figure.
    """
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize dataset (checks canonical OASIS layout)
    try:
        ds = OASIS2DSegmentation(root=args.root, split=args.split,
                                 num_classes=args.num_classes, norm=True)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(2)
    # Exit gracefully if dataset is empty
    if len(ds) == 0:
        print(f"No data found in split '{args.split}' under {args.root}.")
        sys.exit(2)
    # Select sample index (clamped to dataset length)
    idx = max(0, min(args.index, len(ds) - 1))
    img_t, gt_t = ds[idx]  # Get one sample (image, ground truth mask)
    # Build and load model checkpoint
    model = build_model(args.num_classes, device)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Checkpoint not found at: {ckpt_path}")
        sys.exit(2)
    load_checkpoint(model, ckpt_path)
    model.eval()  # Set model to evaluation mode (disables dropout/batchnorm updates)
    # Run inference on selected sample
    img_in = img_t.unsqueeze(0).to(device)  # Add batch dimension -> (1,1,H,W)
    pred_t = predict_one(model, img_in)     # Get predicted segmentation mask
    # Convert tensors to numpy for visualization
    img_np = img_t.squeeze(0).cpu().numpy()  # (H,W) float32
    gt_np = gt_t.cpu().numpy()               # (H,W) int
    pred_np = pred_t.cpu().numpy()           # (H,W) int
    # Render and save visualization
    render_triplet(img_np, gt_np, pred_np, Path(args.out))
    print(f"Saved visualisation to: {args.out}")


if __name__ == "__main__":
    # When executed directly, run the main() function
    main()
