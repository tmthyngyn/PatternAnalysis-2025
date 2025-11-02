import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image   # new: for PNGs

RANGPUR_OASIS = "/home/groups/comp3710/OASIS"
COLAB_OASIS = "/content/drive/MyDrive/comp3710/OASIS"

def guess_oasis_root():
    if "OASIS_DIR" in os.environ:
        return os.environ["OASIS_DIR"]
    if os.path.exists(COLAB_OASIS):
        return COLAB_OASIS
    if os.path.exists(RANGPUR_OASIS):
        return RANGPUR_OASIS
    return "./data/OASIS"

class OASIS2DSegmentation(Dataset):
    def __init__(self, root=None, split="train", norm=True, num_classes=4):
        self.root = root or guess_oasis_root()
        self.split = split
        self.norm = norm
        self.num_classes = num_classes

        # this is the ACTUAL rangpur layout
        img_dir = os.path.join(self.root, f"keras_png_slices_{split}")
        seg_dir = os.path.join(self.root, f"keras_png_slices_seg_{split}")

        colab_img_dir = os.path.join(self.root, split, "images")
        colab_lbl_dir = os.path.join(self.root, split, "labels")

        if os.path.exists(colab_img_dir) and os.path.exists(colab_lbl_dir):
            self.imgs = sorted(glob.glob(os.path.join(colab_img_dir, "*.*png")))
            self.lbls = sorted(glob.glob(os.path.join(colab_lbl_dir, "*.*png")))
        elif os.path.exists(img_dir) and os.path.exists(seg_dir):
            self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.*png")))
            self.lbls = sorted(glob.glob(os.path.join(seg_dir, "*.*png")))
        else:
            self.imgs = []
            self.lbls = []
        self.fake_mode = len(self.imgs) == 0
        if self.fake_mode:
            self.length = 8  # small fake dataset

    def __len__(self):
        return self.length if self.fake_mode else len(self.imgs)
    
    def _remap_labels(self, lbl: np.ndarray) -> np.ndarray:
        unique_vals = np.unique(lbl)
        if unique_vals.min() >= 0 and unique_vals.max() < self.num_classes:
            return lbl
        if len(unique_vals) <= self.num_classes:
            remap = {v: i for i, v in enumerate(sorted(unique_vals))}
            lbl = np.vectorize(remap.get)(lbl).astype(np.int64)
            return lbl
        return np.clip(lbl, 0, self.num_classes - 1).astype(np.int64)
    
    def __getitem__(self, idx):
        if self.fake_mode:
            img = np.random.randn(1, 256, 256).astype(np.float32)
            mask = np.random.randint(0, self.num_classes, size=(256, 256), dtype=np.int64)
            return torch.from_numpy(img), torch.from_numpy(mask)

        # load PNGs
        img_path = self.imgs[idx]
        lbl_path = self.lbls[idx]

        img = np.array(Image.open(img_path)).astype(np.float32)
        lbl = np.array(Image.open(lbl_path)).astype(np.int64)

        # normalise image
        if self.norm:
            img = (img - img.mean()) / (img.std() + 1e-8)

        # channel-first
        img = np.expand_dims(img, 0)
        lbl = self._remap_labels(lbl)

        return torch.from_numpy(img), torch.from_numpy(lbl)
