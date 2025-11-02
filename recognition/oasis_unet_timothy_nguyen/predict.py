import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# make sure we import our local modules, not random pip ones
THIS_DIR = os.path.dirname(__file__)
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dataset import OASIS2DSegmentation, guess_oasis_root
import modules  # to get UNet / UNet2D


def build_model(num_classes: int, device: torch.device):
    # same defensive builder we used in train.py
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

    raise RuntimeError("Could not find UNet / UNet2D in modules.py")


def load_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # case 1: it's a full checkpoint (what train.py saved)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"[predict] Loaded model_state_dict from checkpoint: {ckpt_path}")
    else:
        # case 2: it's a bare state_dict
        model.load_state_dict(ckpt)
        print(f"[predict] Loaded raw state_dict: {ckpt_path}")

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[predict] Using device:", device)

    data_root = guess_oasis_root()
    print("[predict] Using dataset root:", data_root)

    # try to find a checkpoint in the usual places
    candidate_ckpts = [
        "trained_models/oasis_unet/best_model.pth",
        "checkpoints/oasis_unet.pth",
    ]
    ckpt_path = None
    for p in candidate_ckpts:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        raise FileNotFoundError(
            "No checkpoint found. Looked in: "
            + ", ".join(candidate_ckpts)
        )

    # build dataset (just grab a few samples)
    ds = OASIS2DSegmentation(
        root=data_root,
        split="train",      # or "test" if you add it later
        norm=True,
        num_classes=4,
        backend="png",
    )
    print(f"[predict] Dataset len={len(ds)}, fake_mode={ds.fake_mode}")

    # build model and load weights
    model = build_model(num_classes=4, device=device)
    model = load_checkpoint(model, ckpt_path, device)
    model.eval()

    # pick an index to visualise
    idx = 0
    img, gt = ds[idx]     # img: (1,H,W), gt: (H,W)
    img_in = img.unsqueeze(0).to(device)  # (1,1,H,W)

    with torch.no_grad():
        logits = model(img_in)            # (1,C,H,W)
        pred = torch.argmax(logits, dim=1).cpu().squeeze(0).numpy()

    # convert GT to numpy
    gt = gt.numpy()
    img_vis = img.squeeze(0).numpy()

    # plot
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_vis, cmap="gray")
    plt.title("image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap="viridis", vmin=0, vmax=3)
    plt.title("gt")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap="viridis", vmin=0, vmax=3)
    plt.title("pred")
    plt.axis("off")

    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    save_path = "outputs/prediction_example.png"
    plt.savefig(save_path)
    print(f"[predict] Saved visualisation to {save_path}")



if __name__ == "__main__":
    main()
