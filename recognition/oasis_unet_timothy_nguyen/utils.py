import torch
import torch.nn.functional as F

def one_hot_encode(mask, num_classes):
    """Convert integer mask (B, H, W) to one-hot (B, num_classes, H, W)."""
    return F.one_hot(mask.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
