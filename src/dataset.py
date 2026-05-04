"""
Dataset utilities for 3D volumetric segmentation.
Supports NIfTI (.nii/.nii.gz) and NumPy (.npy) formats.
Author: Prateek Gaur
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Tuple, Callable


def normalize_volume(volume: np.ndarray, percentile: int = 99) -> np.ndarray:
    """Clip and normalize a 3D volume to [0, 1]."""
    p_low, p_high = np.percentile(volume, 1), np.percentile(volume, percentile)
    volume = np.clip(volume, p_low, p_high)
    volume = (volume - p_low) / (p_high - p_low + 1e-8)
    return volume.astype(np.float32)


def random_flip(volume: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Random flipping along all 3 axes."""
    for axis in range(3):
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=axis).copy()
            mask   = np.flip(mask,   axis=axis).copy()
    return volume, mask


def random_crop(
    volume: np.ndarray,
    mask: np.ndarray,
    patch_size: Tuple[int, int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly crop a patch of given size from volume and mask."""
    d, h, w = volume.shape
    pd, ph, pw = patch_size
    sd = np.random.randint(0, max(d - pd, 1))
    sh = np.random.randint(0, max(h - ph, 1))
    sw = np.random.randint(0, max(w - pw, 1))
    return (
        volume[sd:sd+pd, sh:sh+ph, sw:sw+pw],
        mask  [sd:sd+pd, sh:sh+ph, sw:sw+pw],
    )


class VolumetricDataset(Dataset):
    """
    Generic 3D segmentation dataset.

    Expects the data directory to have two sub-folders:
        images/   — 3D volumes as .npy files (float32)
        masks/    — corresponding segmentation masks as .npy files (int64)

    Args:
        data_dir   : root directory containing images/ and masks/
        patch_size : (D, H, W) patch to crop during training; None = full volume
        augment    : whether to apply random flips
        transform  : optional additional transform callable
    """

    def __init__(
        self,
        data_dir: str,
        patch_size: Optional[Tuple[int, int, int]] = (64, 64, 64),
        augment: bool = True,
        transform: Optional[Callable] = None,
    ):
        self.image_dir  = os.path.join(data_dir, "images")
        self.mask_dir   = os.path.join(data_dir, "masks")
        self.patch_size = patch_size
        self.augment    = augment
        self.transform  = transform

        self.ids = sorted([
            f for f in os.listdir(self.image_dir) if f.endswith(".npy")
        ])
        if not self.ids:
            raise FileNotFoundError(f"No .npy files found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        name   = self.ids[idx]
        volume = np.load(os.path.join(self.image_dir, name))
        mask   = np.load(os.path.join(self.mask_dir,  name))

        volume = normalize_volume(volume)

        if self.patch_size:
            volume, mask = random_crop(volume, mask, self.patch_size)

        if self.augment:
            volume, mask = random_flip(volume, mask)

        volume_t = torch.from_numpy(volume).unsqueeze(0)   # (1, D, H, W)
        mask_t   = torch.from_numpy(mask.astype(np.int64)) # (D, H, W)

        if self.transform:
            volume_t = self.transform(volume_t)

        return volume_t, mask_t


class SyntheticVolumetricDataset(Dataset):
    """
    Synthetic dataset for testing the pipeline without real scan data.
    Generates random volumes with simple spherical ground-truth masks.
    """

    def __init__(
        self,
        num_samples: int = 50,
        volume_size: Tuple[int, int, int] = (64, 64, 64),
        num_classes: int = 2,
    ):
        self.num_samples  = num_samples
        self.volume_size  = volume_size
        self.num_classes  = num_classes

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        d, h, w = self.volume_size
        volume = np.random.rand(d, h, w).astype(np.float32)

        # Synthetic spherical mask
        mask = np.zeros((d, h, w), dtype=np.int64)
        cx, cy, cz = d // 2, h // 2, w // 2
        radius = min(d, h, w) // 4
        for z in range(d):
            for y in range(h):
                for x in range(w):
                    if (z - cx)**2 + (y - cy)**2 + (x - cz)**2 < radius**2:
                        mask[z, y, x] = 1

        volume_t = torch.from_numpy(volume).unsqueeze(0)
        mask_t   = torch.from_numpy(mask)
        return volume_t, mask_t


if __name__ == "__main__":
    ds = SyntheticVolumetricDataset(num_samples=4)
    vol, msk = ds[0]
    print(f"Volume shape: {vol.shape}, Mask shape: {msk.shape}")
    print(f"Unique mask labels: {torch.unique(msk).tolist()}")
