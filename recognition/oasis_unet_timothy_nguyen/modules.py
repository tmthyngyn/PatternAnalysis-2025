"""
Minimal U-Net model components for 2D medical image segmentation.

Overview
--------
This module defines the convolutional blocks and U-Net architecture used for
segmenting 2D slices of MRI data from the OASIS dataset. The model here is 
intentionally simple and fully self-contained, making it suitable for 
quick experimentation on Colab or local systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------
# Basic building blocks for the U-Net architecture
# ---------------------------------------------------------------------
class DoubleConv(nn.Module):
    """
    A common U-Net block consisting of two convolutional layers,
    each followed by batch normalization and ReLU activation.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    out_ch : int
        Number of output channels.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            # First convolution: reduces aliasing and extracts features
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            # Second convolution: refines features for better localization
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Apply two conv-batchnorm-relu layers."""
        return self.net(x)

# ---------------------------------------------------------------------
# Downsampling and Upsampling helpers
# ---------------------------------------------------------------------
class Down(nn.Module):
    """Downscaling with maxpool followed by DoubleConv."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        # Max pooling halves spatial dimensions, DoubleConv increases feature depth
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        """Apply maxpool → double convolution."""
        return self.net(x)


class Up(nn.Module):
    """Upscaling followed by DoubleConv."""

    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        # Choose between bilinear interpolation or transposed convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        """
        Forward pass for the upsampling block.

        Parameters
        ----------
        x1 : torch.Tensor
            Decoder feature map (after upsampling).
        x2 : torch.Tensor
            Corresponding encoder feature map for skip connection.
        """
        x1 = self.up(x1)  # upscale by factor of 2
        # Handle possible size mismatches due to rounding in pooling
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        # Pad if upsampled map is smaller (center crop alignment)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # Concatenate encoder and decoder features (skip connection)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)  # refine fused features


class OutConv(nn.Module):
    """Final 1×1 convolution to map feature channels to class logits."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        """Return per-pixel class logits (no activation)."""
        return self.conv(x)

# ---------------------------------------------------------------------
# U-Net architecture
# ---------------------------------------------------------------------
class UNet(nn.Module):
    """
    Standard 2D U-Net architecture for semantic segmentation.

    Parameters
    ----------
    in_channels : int
        Number of channels in input image (e.g., 1 for grayscale MRI slices).
    out_channels : int
        Number of target segmentation classes.
    bilinear : bool
        Whether to use bilinear upsampling (True) or transposed convolutions (False).
    """

    def __init__(self, in_channels=1, out_channels=4, bilinear=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        # Encoder: progressively reduce spatial resolution while increasing depth
        self.inc = DoubleConv(in_channels, 64)       # input conv block
        self.down1 = Down(64, 128)                   # 1st downsample
        self.down2 = Down(128, 256)                  # 2nd downsample
        self.down3 = Down(256, 512)                  # 3rd downsample
        # If bilinear upsampling, reduce intermediate channels by half
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # Decoder: progressively upsample while fusing encoder features
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # Final output layer: maps to segmentation logits
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        """Forward pass through encoder, bottleneck, and decoder."""
        x1 = self.inc(x)             # encode level 1
        x2 = self.down1(x1)          # encode level 2
        x3 = self.down2(x2)          # encode level 3
        x4 = self.down3(x3)          # encode level 4
        x5 = self.down4(x4)          # bottleneck
        # Decode with skip connections (mirrors encoder)
        x = self.up1(x5, x4)         # combine bottleneck + encoder-4
        x = self.up2(x, x3)          # combine decoder + encoder-3
        x = self.up3(x, x2)          # combine decoder + encoder-2
        x = self.up4(x, x1)          # combine decoder + encoder-1
        logits = self.outc(x)        # produce final logits
        return logits                # no softmax here (applied in loss/inference)
