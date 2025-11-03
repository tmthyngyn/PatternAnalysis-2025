# dataset.py  —  Simplified PNG-only version (no NIfTI support)

import os
import re
import glob
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# ---------------------------------------------------------------------
# Expected canonical layout
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
    for split in EXPECTED_SPLITS:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            missing.append(f"{split}/")
    if missing:
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
# PNG pairing
# ---------------------------------------------------------------------

# Accept patterns like:
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
    return sorted(glob.glob(os.path.join(d, "*.png")) + glob.glob(os.path.join(d, "*.PNG")))


def _parse_png_key(path: str) -> Tuple[Optional[str], Optional[bool]]:
    """Return (key, is_label) from a PNG filename or (None, None) if no match."""
    b = os.path.basename(path)
    m = _RX_NII_PNG.match(b) or _RX_PNG.match(b)
    if not m:
        return None, None
    pid = m.group("pid")
    sid = m.group("sid")
    prefix = (m.group("prefix") or "").lower()
    is_label = prefix in {"seg", "label"}
    return f"{pid}_{sid}", is_label


def _pair_pngs(images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
    img_paths = _list_pngs(images_dir)
    lbl_paths = _list_pngs(labels_dir)

    by_key_img, bad_img = {}, []
    for p in img_paths:
        k, is_label = _parse_png_key(p)
        if k is None or is_label:
            bad_img.append(os.path.basename(p))
        else:
            by_key_img[k] = p

    by_key_lbl, bad_lbl = {}, []
    for p in lbl_paths:
        k, is_label = _parse_png_key(p)
        if k is None or not is_label:
            bad_lbl.append(os.path.basename(p))
        else:
            by_key_lbl[k] = p

    common = sorted(set(by_key_img).intersection(by_key_lbl))
    pairs = [(by_key_img[k], by_key_lbl[k]) for k in common]

    if not pairs:
        msg = [
            "No paired .png files found.",
            f"Images dir: {images_dir} (count={len(img_paths)})",
            f"Labels dir: {labels_dir} (count={len(lbl_paths)})",
        ]
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
            print(f"  Bad images entries: {len(bad_img)} (showing up to 5)")
            for n in bad_img[:5]:
                print("   -", n)
        if bad_lbl:
            print(f"  Bad labels entries: {len(bad_lbl)} (showing up to 5)")
            for n in bad_lbl[:5]:
                print("   -", n)

    return pairs


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class OASIS2DSegmentation(Dataset):
    """
    Canonicalised OASIS 2D dataset (PNG-only).
    Expects each split to contain 'images/' and 'labels/' folders.

    Returns:
      image: (1,H,W) float32, z-scored if norm=True
      mask:  (H,W)   int64 with labels in [0..num_classes-1]
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

        _enforce_oasis_layout(self.root)

        img_dir = os.path.join(self.root, split, "images")
        lbl_dir = os.path.join(self.root, split, "labels")
        if not (os.path.isdir(img_dir) and os.path.isdir(lbl_dir)):
            raise FileNotFoundError(
                f"Missing required subfolders under {split}/. "
                f"Expected 'images/' and 'labels/' inside {os.path.join(self.root, split)}."
            )

        self.pairs = _pair_pngs(img_dir, lbl_dir)
        if not self.pairs:
            raise FileNotFoundError(f"No valid image/label pairs found in split '{split}'.")

    def __len__(self) -> int:
        return len(self.pairs)

    @staticmethod
    def _zscore(arr: np.ndarray) -> np.ndarray:
        m = float(arr.mean())
        s = float(arr.std())
        if s == 0.0:
            s = 1.0
        return (arr - m) / s

    def _remap_labels(self, mask: np.ndarray) -> np.ndarray:
        """
        Map arbitrary integer labels to compact range [0..num_classes-1].
        Extras are clipped to last class index.
        """
        uniq = np.unique(mask)
        lut = {int(v): min(i, self.num_classes - 1) for i, v in enumerate(uniq)}
        out = np.vectorize(lambda v: lut[int(v)])(mask).astype(np.int64)
        return out

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, lbl_path = self.pairs[idx]
        img = np.asarray(Image.open(img_path).convert("L")).astype(np.float32)
        if self.norm:
            img = self._zscore(img)
        img = np.expand_dims(img, axis=0)  # (1,H,W)

        mask = np.asarray(Image.open(lbl_path))
        mask = self._remap_labels(mask)

        return torch.from_numpy(img), torch.from_numpy(mask).long()

    def calculate_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for this split."""
        counts = np.zeros(self.num_classes, dtype=np.int64)
        for _, lbl_path in self.pairs:
            m = np.asarray(Image.open(lbl_path))
            m = self._remap_labels(m)
            vals, cnt = np.unique(m, return_counts=True)
            for v, c in zip(vals, cnt):
                if v < self.num_classes:
                    counts[v] += int(c)
        counts = np.maximum(counts, 1)
        inv = 1.0 / counts.astype(np.float64)
        w = inv / inv.sum() * self.num_classes
        return torch.tensor(w, dtype=torch.float32)
