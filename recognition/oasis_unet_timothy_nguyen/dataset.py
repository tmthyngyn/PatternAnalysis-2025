import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib

# Paths for both Colab and Rangpur
RANGPUR_OASIS = "/home/groups/comp3710/OASIS"
COLAB_OASIS = "/content/drive/MyDrive/comp3710/OASIS"

def guess_oasis_root():
    """Decide where to load OASIS data from depending on environment."""
    if "OASIS_DIR" in os.environ:
        return os.environ["OASIS_DIR"]
    if os.path.exists(COLAB_OASIS):
        return COLAB_OASIS
    if os.path.exists(RANGPUR_OASIS):
        return RANGPUR_OASIS
    return "./data/OASIS"

class OASIS2DSegmentation(Dataset):
    """2D OASIS brain segmentation dataset."""

    def __init__(self, root=None, split="train", norm=True, num_classes=4):
        self.root = root or guess_oasis_root()
        self.split = split
        self.norm = norm
        self.num_classes = num_classes

        self.imgs = sorted(glob.glob(os.path.join(self.root, split, "images", "*.nii*")))
        self.lbls = sorted(glob.glob(os.path.join(self.root, split, "labels", "*.nii*")))

        # If no data, fake mode (for local testing)
        self.fake_mode = len(self.imgs) == 0
        if self.fake_mode:
            self.length = 8  # tiny fake dataset

    def __len__(self):
        return self.length if self.fake_mode else len(self.imgs)

    def __getitem__(self, idx):
        if self.fake_mode:
            img = np.random.randn(1, 256, 256).astype(np.float32)
            mask = np.random.randint(0, self.num_classes, size=(256, 256), dtype=np.int64)
            return torch.from_numpy(img), torch.from_numpy(mask)

        img = nib.load(self.imgs[idx]).get_fdata(caching="unchanged")
        mask = nib.load(self.lbls[idx]).get_fdata(caching="unchanged")

        if img.ndim == 3: img = img[:, :, 0]
        if mask.ndim == 3: mask = mask[:, :, 0]

        img = img.astype(np.float32)
        if self.norm:
            img = (img - img.mean()) / (img.std() + 1e-8)

        img = np.expand_dims(img, 0)
        return torch.from_numpy(img), torch.from_numpy(mask.astype(np.int64))
