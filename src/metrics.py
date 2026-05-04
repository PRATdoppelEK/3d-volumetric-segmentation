"""
Segmentation metrics: Dice, IoU, Hausdorff distance.
Author: Prateek Gaur
"""

import torch
import numpy as np


def dice_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Binary Dice coefficient (foreground class = 1)."""
    p = (preds   == 1).float().view(-1)
    t = (targets == 1).float().view(-1)
    inter = (p * t).sum()
    return ((2 * inter + smooth) / (p.sum() + t.sum() + smooth)).item()


def iou_score(preds: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6) -> float:
    """Intersection over Union (foreground class = 1)."""
    p = (preds   == 1).float().view(-1)
    t = (targets == 1).float().view(-1)
    inter = (p * t).sum()
    union = p.sum() + t.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def precision_recall_f1(preds: torch.Tensor, targets: torch.Tensor):
    """Compute precision, recall, and F1 for foreground class."""
    tp = ((preds == 1) & (targets == 1)).sum().float()
    fp = ((preds == 1) & (targets == 0)).sum().float()
    fn = ((preds == 0) & (targets == 1)).sum().float()

    precision = (tp / (tp + fp + 1e-6)).item()
    recall    = (tp / (tp + fn + 1e-6)).item()
    f1        = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1


def volume_similarity(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Volume similarity metric (between -1 and 1)."""
    v_pred   = (preds   == 1).sum().float()
    v_target = (targets == 1).sum().float()
    return (1 - abs(v_pred - v_target) / (v_pred + v_target + 1e-6)).item()


def compute_all_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict:
    """Return a dict with all available metrics."""
    prec, rec, f1 = precision_recall_f1(preds, targets)
    return {
        "dice":              dice_score(preds, targets),
        "iou":               iou_score(preds, targets),
        "precision":         prec,
        "recall":            rec,
        "f1":                f1,
        "volume_similarity": volume_similarity(preds, targets),
    }
