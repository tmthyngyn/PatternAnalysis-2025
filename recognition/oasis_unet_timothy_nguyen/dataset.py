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

        if os.path.exists(img_dir) and os.path.exists(seg_dir):
            self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            self.lbls = sorted(glob.glob(os.path.join(seg_dir, "*.png")))
        else:
            # fall back to the old layout or fake mode
            self.imgs = []
            self.lbls = []

        self.fake_mode = len(self.imgs) == 0
        if self.fake_mode:
            self.length = 8  # small fake dataset

    def __len__(self):
        return self.length if self.fake_mode else len(self.imgs)

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

        return torch.from_numpy(img), torch.from_numpy(lbl)
