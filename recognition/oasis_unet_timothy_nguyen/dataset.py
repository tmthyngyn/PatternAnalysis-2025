import os, glob, re
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

# try to use our own utils (for NIfTI), but don't die if it's not there yet
try:
    import utils as ds_utils
except ImportError:
    ds_utils = None

# known dataset roots
RANGPUR_OASIS = "/home/groups/comp3710/OASIS"
COLAB_OASIS   = "/content/drive/MyDrive/comp3710/OASIS"


def guess_oasis_root():
    """
    Try to guess where the OASIS data lives.
    Priority:
    1. OASIS_DIR env var
    2. Colab default path
    3. Rangpur path
    4. local ./data/OASIS
    """
    if "OASIS_DIR" in os.environ:
        return os.environ["OASIS_DIR"]
    if os.path.exists(COLAB_OASIS):
        return COLAB_OASIS
    if os.path.exists(RANGPUR_OASIS):
        return RANGPUR_OASIS
    return "./data/OASIS"


def natural_sort_key(path: str):
    """
    Sort filenames like ..._2.nii.png before ..._10.nii.png.
    """
    return [int(t) if t.isdigit() else t.lower()
            for t in re.findall(r"\d+|\D+", os.path.basename(path))]


class OASIS2DSegmentation(Dataset):
    """
    2D OASIS dataset loader with:
      - PNG support (your current workflow)
      - optional NIfTI support (to match COMP3710 appendix style)
      - helper utils for plotting and class-weight calculation

    Parameters
    ----------
    root : str, optional
        Dataset root folder. If None, will call guess_oasis_root().
    split : str, optional
        'train', 'test', or 'val' depending on your folder structure.
    norm : bool, optional
        If True, normalise images to zero mean / unit variance.
    num_classes : int, optional
        Number of segmentation classes.
    backend : str, optional
        'png' (default): look for PNGs in <root>/<split>/images + <root>/<split>/labels
        'nifti': load 2D NIfTI slices via utils.load_data_2D_from_directory(...)
    """

    def __init__(self,
                 root: str | None = None,
                 split: str = "train",
                 norm: bool = True,
                 num_classes: int = 4,
                 backend: str = "png"):

        self.root = root or guess_oasis_root()
        self.split = split
        self.norm = norm
        self.num_classes = num_classes
        self.backend = backend

        # ------------------------------------------------------------------
        # 1) COLAB layout  : <root>/<split>/images  +  <root>/<split>/labels
        # 2) RANGPUR layout: <root>/keras_png_slices_<split>  +  <root>/keras_png_slices_seg_<split>
        # 3) NIFTI layout  : handled via utils (only if backend='nifti')
        # ------------------------------------------------------------------
        colab_img_dir   = os.path.join(self.root, split, "images")
        colab_lbl_dir   = os.path.join(self.root, split, "labels")
        rangpur_img_dir = os.path.join(self.root, f"keras_png_slices_{split}")
        rangpur_lbl_dir = os.path.join(self.root, f"keras_png_slices_seg_{split}")

        # ------------------------------------------------------------------
        # BACKEND: NIfTI mode
        # ------------------------------------------------------------------
        if self.backend == "nifti":
            if ds_utils is None:
                raise RuntimeError(
                    "backend='nifti' was requested but utils.py could not be imported. "
                    "Add utils.py (we will write it next) or use backend='png'."
                )

            # we'll expect e.g. <root>/<split>/images_nifti and <root>/<split>/labels_nifti
            nifti_img_dir = os.path.join(self.root, split, "images_nifti")
            nifti_lbl_dir = os.path.join(self.root, split, "labels_nifti")

            if not (os.path.exists(nifti_img_dir) and os.path.exists(nifti_lbl_dir)):
                raise FileNotFoundError(
                    f"NIfTI backend selected but could not find:\n{nifti_img_dir}\n{nifti_lbl_dir}"
                )

            # this will return np arrays, we keep them in memory (small 2D slices)
            self.imgs = ds_utils.load_data_2D_from_directory(
                nifti_img_dir,
                norm_image=norm,
                one_hot=False,
                resized=False,
                resizing_masks=False,
                dtype=np.float32,
            )
            self.lbls = ds_utils.load_data_2D_from_directory(
                nifti_lbl_dir,
                norm_image=False,
                one_hot=False,
                resized=False,
                resizing_masks=True,   # nearest for masks
                dtype=np.int64,
            )

            if self.imgs.shape[0] != self.lbls.shape[0]:
                raise ValueError("NIfTI images and labels have different lengths")

            self.fake_mode = False
            return  # NIfTI path ends here

        # ------------------------------------------------------------------
        # BACKEND: PNG mode
        # ------------------------------------------------------------------
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
            self.length = 8  # small dummy set

    # ----------------------------------------------------------------------
    # Standard Dataset methods
    # ----------------------------------------------------------------------
    def __len__(self):
        return self.length if self.fake_mode else len(self.imgs)

    def _remap_labels(self, lbl: np.ndarray) -> np.ndarray:
        """
        Ensure labels are consecutive in [0, num_classes-1].
        """
        unique_vals = np.unique(lbl)
        # already fine
        if unique_vals.min() >= 0 and unique_vals.max() < self.num_classes:
            return lbl
        # small number of classes: remap
        if len(unique_vals) <= self.num_classes:
            remap = {v: i for i, v in enumerate(sorted(unique_vals))}
            return np.vectorize(remap.get)(lbl).astype(np.int64)
        # fallback: clip
        return np.clip(lbl, 0, self.num_classes - 1).astype(np.int64)

    def __getitem__(self, idx):
        # ---------------------- NIfTI backend ------------------------------
        if self.backend == "nifti" and not self.fake_mode:
            img = self.imgs[idx].astype(np.float32)
            lbl = self.lbls[idx].astype(np.int64)
            # if 3D-ish (H,W,1) squeeze
            if img.ndim == 3 and img.shape[-1] == 1:
                img = img[..., 0]
            if lbl.ndim == 3 and lbl.shape[-1] == 1:
                lbl = lbl[..., 0]

            if self.norm:
                img = (img - img.mean()) / (img.std() + 1e-8)

            img = np.expand_dims(img, 0)  # (1,H,W)
            lbl = self._remap_labels(lbl)
            return torch.from_numpy(img), torch.from_numpy(lbl)

        # ---------------------- fake backend -------------------------------
        if self.fake_mode:
            img = np.random.randn(1, 256, 256).astype(np.float32)
            mask = np.random.randint(0, self.num_classes, size=(256, 256), dtype=np.int64)
            return torch.from_numpy(img), torch.from_numpy(mask)

        # ---------------------- PNG backend --------------------------------
        img_path = self.imgs[idx]
        lbl_path = self.lbls[idx]

        img = Image.open(img_path).convert("L")
        lbl = Image.open(lbl_path).convert("L")

        img = np.array(img).astype(np.float32)
        lbl = np.array(lbl).astype(np.int64)

        if self.norm:
            img = (img - img.mean()) / (img.std() + 1e-8)

        img = np.expand_dims(img, 0)  # (1,H,W)
        lbl = self._remap_labels(lbl)

        return torch.from_numpy(img), torch.from_numpy(lbl)

    # ----------------------------------------------------------------------
    # EXTRA UTILITIES
    # ----------------------------------------------------------------------
    def img_show(self, start_idx: int = 0, num: int = 3):
        """
        Plot a few samples from the dataset:
          - original image
          - label / mask

        This is mainly for the README / markers.
        """
        import matplotlib.pyplot as plt

        end_idx = min(start_idx + num, len(self))
        for i in range(start_idx, end_idx):
            img, lbl = self[i]
            img = img.squeeze().numpy()
            lbl = lbl.numpy()

            plt.figure(figsize=(6, 3))
            plt.subplot(1, 2, 1)
            plt.imshow(img, cmap="gray")
            plt.title(f"image [{i}]")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(lbl, cmap="viridis", vmin=0, vmax=self.num_classes - 1)
            plt.title(f"label [{i}]")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

    def calculate_class_weights(self) -> torch.Tensor:
        """
        Iterate through the dataset and count how many pixels belong to each
        class, then return inverse-frequency weights suitable for
        nn.CrossEntropyLoss(weight=...).

        Returns
        -------
        torch.Tensor
            shape (num_classes,)
        """
        counts = np.zeros(self.num_classes, dtype=np.float64)

        for i in range(len(self)):
            _, lbl = self[i]
            lbl_np = lbl.numpy()
            for c in range(self.num_classes):
                counts[c] += (lbl_np == c).sum()

        # avoid div by zero
        counts = np.maximum(counts, 1.0)
        inv = 1.0 / counts
        weights = inv / inv.sum() * self.num_classes
        return torch.from_numpy(weights.astype(np.float32))

# --------------------------------------------------------------------------
# helpful: run this file directly to see shapes / plots
# --------------------------------------------------------------------------
if __name__ == "__main__":
    ds = OASIS2DSegmentation(
        root=guess_oasis_root(),
        split="train",
        norm=True,
        num_classes=4,
        backend="png",         # change to "nifti" once utils.py is in place
    )
    print("[main] root:", ds.root)
    print("[main] length:", len(ds))
    print("[main] fake_mode:", ds.fake_mode)

    if not ds.fake_mode:
        x, y = ds[0]
        print("[main] sample image shape:", x.shape)
        print("[main] sample label shape:", y.shape)

        # show a few samples
        ds.img_show(0, 3)

        # print class weights
        print("[main] class weights:", ds.calculate_class_weights())
    else:
        print("[main] dataset is in fake mode, please check your paths.")
