"""
3D U-Net Segmentation Model
Author: Prateek Gaur
Description: PyTorch implementation of 3D U-Net for volumetric segmentation
             of complex engineering geometries (e.g., battery modules, structural parts).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv3D(nn.Module):
    """Two consecutive 3D convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down3D(nn.Module):
    """Downsampling with MaxPool then DoubleConv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels),
        )

    def forward(self, x):
        return self.pool_conv(x)


class Up3D(nn.Module):
    """Upsampling then DoubleConv (with skip connection)."""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if sizes don't match
        diff_d = x2.size(2) - x1.size(2)
        diff_h = x2.size(3) - x1.size(3)
        diff_w = x2.size(4) - x1.size(4)
        x1 = F.pad(
            x1,
            [
                diff_w // 2, diff_w - diff_w // 2,
                diff_h // 2, diff_h - diff_h // 2,
                diff_d // 2, diff_d - diff_d // 2,
            ],
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    """
    Full 3D U-Net architecture for volumetric segmentation.

    Args:
        in_channels  : number of input channels (e.g. 1 for grayscale CT)
        out_channels : number of segmentation classes
        base_features: feature depth at the first encoder level (default 32)
        trilinear    : use trilinear upsampling (True) or transposed conv (False)
    """

    def __init__(self, in_channels=1, out_channels=2, base_features=32, trilinear=True):
        super().__init__()
        f = base_features
        self.inc   = DoubleConv3D(in_channels, f)
        self.down1 = Down3D(f,     f * 2)
        self.down2 = Down3D(f * 2, f * 4)
        self.down3 = Down3D(f * 4, f * 8)
        factor     = 2 if trilinear else 1
        self.down4 = Down3D(f * 8, f * 16 // factor)

        self.up1 = Up3D(f * 16, f * 8  // factor, trilinear)
        self.up2 = Up3D(f * 8,  f * 4  // factor, trilinear)
        self.up3 = Up3D(f * 4,  f * 2  // factor, trilinear)
        self.up4 = Up3D(f * 2,  f,                trilinear)
        self.outc = OutConv3D(f, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


if __name__ == "__main__":
    model = UNet3D(in_channels=1, out_channels=2, base_features=32)
    x = torch.randn(1, 1, 64, 64, 64)
    out = model(x)
    print(f"Input shape : {x.shape}")
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
