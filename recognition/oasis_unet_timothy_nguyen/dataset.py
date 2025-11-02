import os, glob, re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# Known dataset roots
RANGPUR_OASIS = "/home/groups/comp3710/OASIS"
COLAB_OASIS   = "/content/drive/MyDrive/comp3710/OASIS"


def guess_oasis_root():
    """Automatically determine the dataset root for Colab, Rangpur, or local use."""
    if "OASIS_DIR" in os.environ:
        return os.environ["OASIS_DIR"]
    if os.path.exists(COLAB_OASIS):
        return COLAB_OASIS
    if os.path.exists(RANGPUR_OASIS):
        return RANGPUR_OASIS
    return "./data/OASIS"


def natural_sort_key(path):
    """Ensure that slice_2 comes before slice_10 when sorting filenames."""
    return [int(t) if t.isdigit() else t.lower()
            for t in re.findall(r"\d+|\D+", os.path.basename(path))]


class OASIS2DSegmentation(Dataset):
    """
    2D OASIS Brain Segmentation Dataset Loader.

    Supports both:
      • Colab layout: <root>/train/images and <root>/train/labels
      • Rangpur layout: <root>/keras_png_slices_train and <root>/keras_png_slices_seg_train
    """

    def __init__(self, root=None, split="train", norm=True, num_classes=4):
        self.root = root or guess_oasis_root()
        self.split = split
        self.norm = norm
        self.num_classes = num_classes

        # Expected folder layouts
        colab_img_dir = os.path.join(self.root, split, "images")
        colab_lbl_dir = os.path.join(self.root, split, "labels")
        rangpur_img_dir = os.path.join(self.root, f"keras_png_slices_{split}")
        rangpur_lbl_dir = os.path.join(self.root, f"keras_png_slices_seg_{split}")

        # Check which layout exists
        if os.path.exists(colab_img_dir) and os.path.exists(colab_lbl_dir):
            print("[dataset] Using Colab-style layout")
            self.imgs = sorted(glob.glob(os.path.join(colab_img_dir, "*.*png")), key=natural_sort_key)
            self.lbls = sorted(glob.glob(os.path.join(colab_lbl_dir, "*.*png")), key=natural_sort_key)

        elif os.path.exists(rangpur_img_dir) and os.path.exists(rangpur_lbl_dir):
            print("[dataset] Using Rangpur layout")
            self.imgs = sorted(glob.glob(os.path.join(rangpur_img_dir, "*.*png")), key=natural_sort_key)
            self.lbls = sorted(glob.glob(os.path.join(rangpur_lbl_dir, "*.*png")), key=natural_sort_key)

        else:
            print(f"[dataset] WARNING: No dataset found under {self.root}. Using fake data.")
            self.imgs, self.lbls = [], []

        self.fake_mode = len(self.imgs) == 0
        if self.fake_mode:
            self.length = 8  # small dummy dataset

    def __len__(self):
        return self.length if self.fake_mode else len(self.imgs)

    def _remap_labels(self, lbl: np.ndarray) -> np.ndarray:
        """
        Ensure label values are in [0, num_classes-1].
        Handles cases like 0,63,126,... or 0..255 or 0..4 when num_classes=4.
        """
        unique_vals = np.unique(lbl)
        if unique_vals.min() >= 0 and unique_vals.max() < self.num_classes:
            return lbl
        if len(unique_vals) <= self.num_classes:
            remap = {v: i for i, v in enumerate(sorted(unique_vals))}
            return np.vectorize(remap.get)(lbl).astype(np.int64)
        return np.clip(lbl, 0, self.num_classes - 1).astype(np.int64)

    def __getitem__(self, idx):
        # Fallback for fake mode
        if self.fake_mode:
            img = np.random.randn(1, 256, 256).astype(np.float32)
            mask = np.random.randint(0, self.num_classes, size=(256, 256), dtype=np.int64)
            return torch.from_numpy(img), torch.from_numpy(mask)

        # Load real image and mask
        img_path, lbl_path = self.imgs[idx], self.lbls[idx]

        # --- Image ---
        img = Image.open(img_path).convert("L")
        img = np.array(img).astype(np.float32)
        if self.norm:
            img = (img - img.mean()) / (img.std() + 1e-8)
        img = np.expand_dims(img, 0)  # (1, H, W)

        # --- Label ---
        lbl = Image.open(lbl_path).convert("L")
        lbl = np.array(lbl).astype(np.int64)
        lbl = self._remap_labels(lbl)

        return torch.from_numpy(img), torch.from_numpy(lbl)
