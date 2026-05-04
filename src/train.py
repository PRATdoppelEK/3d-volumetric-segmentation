"""
Training script for 3D U-Net volumetric segmentation.
Author: Prateek Gaur
"""

import os
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from model   import UNet3D
from dataset import VolumetricDataset, SyntheticVolumetricDataset
from metrics import dice_score, iou_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── Loss ─────────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs   = torch.softmax(logits, dim=1)[:, 1]   # foreground channel
        targets = targets.float()
        inter   = (probs * targets).sum()
        denom   = probs.sum() + targets.sum()
        return 1 - (2 * inter + self.smooth) / (denom + self.smooth)


class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy composite loss."""
    def __init__(self, ce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.ce   = nn.CrossEntropyLoss()
        self.w    = ce_weight

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.w * self.ce(logits, targets)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for volumes, masks in loader:
        volumes, masks = volumes.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(volumes)
        loss   = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        total_loss += loss.item()
        total_dice += dice_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for volumes, masks in loader:
        volumes, masks = volumes.to(device), masks.to(device)
        logits = model(volumes)
        loss   = criterion(logits, masks)
        preds  = logits.argmax(dim=1)

        total_loss += loss.item()
        total_dice += dice_score(preds, masks)
        total_iou  += iou_score(preds, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train 3D U-Net for volumetric segmentation")
    p.add_argument("--data_dir",     default=None,  help="Path to dataset root (images/ + masks/)")
    p.add_argument("--synthetic",    action="store_true", help="Use synthetic data for testing")
    p.add_argument("--epochs",       type=int, default=50)
    p.add_argument("--batch_size",   type=int, default=2)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--base_features",type=int, default=32)
    p.add_argument("--patch_size",   type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--val_split",    type=float, default=0.2)
    p.add_argument("--save_dir",     default="./models")
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def main():
    args   = parse_args()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info(f"Using device: {device}")

    # Dataset
    if args.synthetic:
        logger.info("Using synthetic dataset for demonstration")
        full_ds = SyntheticVolumetricDataset(num_samples=40, volume_size=tuple(args.patch_size))
    else:
        full_ds = VolumetricDataset(
            data_dir=args.data_dir,
            patch_size=tuple(args.patch_size),
            augment=True,
        )

    val_size   = max(1, int(len(full_ds) * args.val_split))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    model     = UNet3D(in_channels=1, out_channels=2, base_features=args.base_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = CombinedLoss(ce_weight=0.5)

    os.makedirs(args.save_dir, exist_ok=True)
    best_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_dice,  val_iou  = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}  Dice: {train_dice:.4f} | "
            f"Val Loss: {val_loss:.4f}  Dice: {val_dice:.4f}  IoU: {val_iou:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_dice": val_dice}, path)
            logger.info(f"  ✓ Saved best model (Dice={best_dice:.4f}) → {path}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
