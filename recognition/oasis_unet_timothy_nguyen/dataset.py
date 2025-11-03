"""
Dataset and preprocessing utilities for OASIS PNG slices.

Overview
--------
This module provides a PyTorch Dataset class to load 2D axial slices
from the OASIS brain MRI dataset. It expects a canonical folder structure
with separate 'images/' and 'labels/' subfolders for each data split
(train/val/test). The dataset supports PNG image files for easy use on
Colab and local systems without NIfTI dependencies.
"""
import os
import re
import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# ---------------------------------------------------------------------
# Expected canonical layout for the dataset
#
# OASIS/
#   train/
#     images/
#     labels/
#   val/
#     images/
#     labels/
#   test/
#     images/
#     labels/
# ---------------------------------------------------------------------

EXPECTED_SPLITS = ("train", "val", "test")

def _enforce_oasis_layout(root: str) -> None:
    """Ensure canonical OASIS/ layout with required split folders."""
    missing = []
    # Verify that all expected split folders exist
    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            missing.append(f"{split}/")
    if missing:
        # Construct a detailed error message if any split folders are missing
        msg = [
            f"[OASIS layout error] Expected canonical layout under: {os.path.abspath(root)}",
            "",
            "Required folder structure:",
            "  OASIS/",
            "    train/images/",
            "    train/labels/",
            "    val/images/",
            "    val/labels/",
            "    test/images/",
            "    test/labels/",
            "",
            "Missing split folders:",
        ] + [f"  - {m}" for m in missing]
        raise FileNotFoundError("\n".join(msg))

# ---------------------------------------------------------------------
# PNG pairing helper functions
# ---------------------------------------------------------------------

# Regular expressions to detect file naming patterns
# Accept filenames like:
#   images: case_367_slice_20.nii.png  OR case_367_slice_20.png
#   labels: seg_367_slice_20.nii.png   OR seg_367_slice_20.png
_RX_NII_PNG = re.compile(
    r"^(?P<prefix>case|img|image|seg|label)?_?(?P<pid>\d+)_slice_(?P<sid>\d+)\.nii\.png$",
    re.IGNORECASE,
)
_RX_PNG = re.compile(
    r"^(?P<prefix>case|img|image|seg|label)?_?(?P<pid>\d+)_slice_(?P<sid>\d+)\.png$",
    re.IGNORECASE,
)

def _list_pngs(d: str) -> List[str]:
    """Return all .png and .PNG files sorted alphabetically."""
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.PNG")))

def _parse_png_key(path: str) -> Tuple[Optional[str], Optional[bool]]:
    """Return (key, is_label) from a PNG filename or (None, None) if no match."""
    b = os.path.basename(path)
    # Match either .nii.png or plain .png naming pattern
    m = _RX_NII_PNG.match(b) or _RX_PNG.match(b)
    if not m:
        return None, None
    pid = m.group("pid")  # patient ID
    sid = m.group("sid")  # slice ID
    prefix = (m.group("prefix") or "").lower()
    is_label = prefix in {"seg", "label"}  # determine if file is a label
    return f"{pid}_{sid}", is_label  # unique slice key

def _pair_pngs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    """Pair each image with its corresponding label by filename key."""
    img_paths = _list_pngs(images_dir)
    lbl_paths = _list_pngs(labels_dir)
    # Index image and label files by their parsed key
    by_key_img, bad_img = {}, []
    for p in img_paths:
        k, is_label = _parse_png_key(p)
        if k is None or is_label:  # ignore mislabelled or invalid files
            bad_img.append(os.path.basename(p))
        else:
            by_key_img[k] = p
    by_key_lbl, bad_lbl = {}, []
    for p in lbl_paths:
        k, is_label = _parse_png_key(p)
        if k is None or not is_label:  # ignore files not marked as labels
            bad_lbl.append(os.path.basename(p))
        else:
            by_key_lbl[k] = p
    # Intersect keys present in both images and labels
    common = sorted(set(by_key_img).intersection(by_key_lbl))
    pairs = [(by_key_img[k], by_key_lbl[k]) for k in common]
    # Raise an error if no valid pairs found
    if not pairs:
        msg = [
            "No paired .png files found.",
            f"Images dir: {images_dir} (count={len(img_paths)})",
            f"Labels dir: {labels_dir} (count={len(lbl_paths)})",
        ]
        # Provide details for debugging
        img_only = sorted(set(by_key_img) - set(by_key_lbl))
        lbl_only = sorted(set(by_key_lbl) - set(by_key_img))
        if img_only:
            msg.append("\nImage keys without matching labels (first 10):")
            msg += [f"  - {k}" for k in img_only[:10]]
        if lbl_only:
            msg.append("\nLabel keys without matching images (first 10):")
            msg += [f"  - {k}" for k in lbl_only[:10]]
        if bad_img:
            msg.append("\nUnparsable / misplaced files in images/ (first 10):")
            msg += [f"  - {n}" for n in bad_img[:10]]
        if bad_lbl:
            msg.append("\nUnparsable / misplaced files in labels/ (first 10):")
            msg += [f"  - {n}" for n in bad_lbl[:10]]
        msg.append(
            "\nExpected filename patterns like:\n"
            "  images: case_<PID>_slice_<SID>.nii.png  or  case_<PID>_slice_<SID>.png\n"
            "  labels: seg_<PID>_slice_<SID>.nii.png   or  seg_<PID>_slice_<SID>.png"
        )
        raise FileNotFoundError("\n".join(msg))
    # Warn if some files were not paired or are malformed
    leftover_img = sorted(set(by_key_img) - set(common))
    leftover_lbl = sorted(set(by_key_lbl) - set(common))
    if leftover_img or leftover_lbl or bad_img or bad_lbl:
        print("[warn] PNG: some files were not paired or were unparsable.")
        if leftover_img:
            print(f"  Unpaired images: {len(leftover_img)} (showing up to 5)")
            for k in leftover_img[:5]:
                print("   -", os.path.basename(by_key_img[k]))
        if leftover_lbl:
            print(f"  Unpaired labels: {len(leftover_lbl)} (showing up to 5)")
            for k in leftover_lbl[:5]:
                print("   -", os.path.basename(by_key_lbl[k]))
        if bad_img:
            print(f"  Bad image entries: {len(bad_img)} (showing up to 5)")
            for n in bad_img[:5]:
                print("   -", n)
        if bad_lbl:
            print(f"  Bad label entries: {len(bad_lbl)} (showing up to 5)")
            for n in bad_lbl[:5]:
                print("   -", n)

    return pairs


# ---------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------
class OASIS2DSegmentation(Dataset):
    """
    Canonicalised OASIS 2D dataset (PNG-only).
    Expects each split to contain 'images/' and 'labels/' folders.

    Returns
    -------
    image : (1, H, W) float32 tensor, z-scored if norm=True
    mask  : (H, W) int64 tensor with labels in [0..num_classes-1]
    """

    def __init__(
        self,
        root: str = "./OASIS",
        split: str = "train",
        num_classes: int = 4,
        norm: bool = True,
    ):
        super().__init__()
        assert split in EXPECTED_SPLITS, f"split must be one of {EXPECTED_SPLITS}"
        self.root = root
        self.split = split
        self.num_classes = int(num_classes)
        self.norm = bool(norm)
        # Validate dataset structure
        _enforce_oasis_layout(self.root)
        # Build absolute paths to image and label folders
        img_dir = os.path.join(self.root, split, "images")
        lbl_dir = os.path.join(self.root, split, "labels")
        # Ensure required subdirectories exist
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            raise FileNotFoundError(
                f"Missing required subfolders under {split}/. "
                f"Expected 'images/' and 'labels/' inside {os.path.join(self.root, split)}."
            )
        # Pair images and labels using helper
        self.pairs = _pair_pngs(img_dir, lbl_dir)
        if not self.pairs:
            raise FileNotFoundError(f"No valid image/label pairs found in split '{split}'.")

    def __len__(self) -> int:
        """Return number of (image, label) pairs in this dataset split."""
        return len(self.pairs)

    def _zscore(arr: np.ndarray) -> np.ndarray:
        """Apply z-score normalization to image array."""
        m = float(arr.mean())
        s = float(arr.std())
        if s == 0.0:  # avoid divide-by-zero
            s = 1.0
        return (arr - m) / s

    def _remap_labels(self, mask: np.ndarray) -> np.ndarray:
        """
        Map arbitrary integer labels to compact range [0..num_classes-1].
        Extra or unexpected labels are clipped to the last valid index.
        """
        uniq = np.unique(mask)  # get all label values in the mask
        # Create lookup table mapping each unique label to an index
        lut = {int(v): min(i, self.num_classes - 1) for i, v in enumerate(uniq)}
        # Replace labels using vectorized mapping
        out = np.vectorize(lambda v: lut[int(v)])(mask).astype(np.int64)
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return (image, mask) pair at index `idx`."""
        img_path, lbl_path = self.pairs[idx]
        # Load grayscale image as float32
        img = np.asarray(Image.open(img_path).convert("L")).astype(np.float32)
        # Optionally normalize to zero-mean and unit variance
        if self.norm:
            img = self._zscore(img)
        img = np.expand_dims(img, axis=0)  # add channel dimension → (1, H, W)
        # Load segmentation mask
        mask = np.asarray(Image.open(lbl_path))
        mask = self._remap_labels(mask)  # remap labels to consistent range
        # Convert to torch tensors
        return torch.from_numpy(img), torch.from_numpy(mask).long()

    def calculate_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights for this dataset split.
        These weights help balance rare vs. common classes during training.
        """
        counts = np.zeros(self.num_classes, dtype=np.int64)
        # Count label occurrences across all masks
        for _, lbl_path in self.pairs:
            m = np.asarray(Image.open(lbl_path))
            m = self._remap_labels(m)
            vals, cnt = np.unique(m, return_counts=True)
            for v, c in zip(vals, cnt):
                if v < self.num_classes:
                    counts[v] += int(c)
        # Avoid zero counts to prevent divide-by-zero
        counts = np.maximum(counts, 1)
        inv = 1.0 / counts.astype(np.float64)
        # Normalise so weights sum to num_classes
        w = inv / inv.sum() * self.num_classes
        return torch.tensor(w, dtype=torch.float32)
