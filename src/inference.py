"""
Inference pipeline with real-time ntfy push notifications.
Author: Prateek Gaur

Usage:
    python inference.py --model_path models/best_model.pth \
                        --input_path data/sample/volume.npy \
                        --output_path results/mask_pred.npy \
                        --ntfy_topic  your-ntfy-topic
"""

import os
import argparse
import logging
import time
import requests
import numpy as np
import torch

from model   import UNet3D
from dataset import normalize_volume
from metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ── ntfy notification ─────────────────────────────────────────────────────────

def send_ntfy(topic: str, title: str, message: str, priority: str = "default"):
    """Send a push notification via ntfy.sh."""
    if not topic:
        return
    try:
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "brain,white_check_mark"},
            timeout=5,
        )
        logger.info(f"ntfy notification sent (status={resp.status_code})")
    except Exception as e:
        logger.warning(f"ntfy notification failed: {e}")


# ── Inference ─────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device, base_features: int = 32) -> UNet3D:
    model     = UNet3D(in_channels=1, out_channels=2, base_features=base_features)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    logger.info(f"Model loaded from {model_path}  (trained Dice={checkpoint.get('val_dice', '?'):.4f})")
    return model


def sliding_window_inference(
    model: UNet3D,
    volume: torch.Tensor,
    patch_size: tuple = (64, 64, 64),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> np.ndarray:
    """
    Sliding-window inference for large volumes that do not fit in GPU memory.
    Returns a probability map (foreground class).
    """
    _, D, H, W = volume.shape
    pd, ph, pw = patch_size
    stride = tuple(max(1, int(p * (1 - overlap))) for p in patch_size)

    pred_map   = np.zeros((D, H, W), dtype=np.float32)
    count_map  = np.zeros((D, H, W), dtype=np.float32)

    for d in range(0, max(D - pd + 1, 1), stride[0]):
        for h in range(0, max(H - ph + 1, 1), stride[1]):
            for w in range(0, max(W - pw + 1, 1), stride[2]):
                ed = min(d + pd, D)
                eh = min(h + ph, H)
                ew = min(w + pw, W)
                patch = volume[:, d:ed, h:eh, w:ew].unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(patch)
                prob = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
                pred_map[d:ed, h:eh, w:ew] += prob
                count_map[d:ed, h:eh, w:ew] += 1

    count_map = np.maximum(count_map, 1)
    return pred_map / count_map


def run_inference(
    model_path: str,
    input_path: str,
    output_path: str,
    patch_size: tuple = (64, 64, 64),
    threshold: float  = 0.5,
    base_features: int = 32,
    ntfy_topic: str   = "",
    gt_mask_path: str = "",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    send_ntfy(ntfy_topic, "3D Segmentation Started", f"Processing: {os.path.basename(input_path)}")

    model  = load_model(model_path, device, base_features)

    volume = np.load(input_path)
    volume = normalize_volume(volume)
    volume_t = torch.from_numpy(volume).unsqueeze(0)  # (1, D, H, W)

    t0    = time.time()
    probs = sliding_window_inference(model, volume_t, patch_size, device=device)
    pred  = (probs >= threshold).astype(np.uint8)
    elapsed = time.time() - t0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.save(output_path, pred)
    logger.info(f"Prediction saved to {output_path}  (elapsed: {elapsed:.2f}s)")

    metrics_str = ""
    if gt_mask_path and os.path.exists(gt_mask_path):
        gt = np.load(gt_mask_path)
        m  = compute_all_metrics(
            torch.from_numpy(pred.astype(np.int64)),
            torch.from_numpy(gt.astype(np.int64)),
        )
        metrics_str = "  ".join(f"{k}={v:.4f}" for k, v in m.items())
        logger.info(f"Metrics: {metrics_str}")

    send_ntfy(
        ntfy_topic,
        "3D Segmentation Complete ✅",
        f"File: {os.path.basename(input_path)}\nTime: {elapsed:.2f}s\n{metrics_str}",
        priority="high",
    )
    return pred


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path",   required=True)
    p.add_argument("--input_path",   required=True)
    p.add_argument("--output_path",  default="results/prediction.npy")
    p.add_argument("--patch_size",   type=int, nargs=3, default=[64, 64, 64])
    p.add_argument("--threshold",    type=float, default=0.5)
    p.add_argument("--base_features",type=int, default=32)
    p.add_argument("--ntfy_topic",   default="")
    p.add_argument("--gt_mask_path", default="")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        model_path    = args.model_path,
        input_path    = args.input_path,
        output_path   = args.output_path,
        patch_size    = tuple(args.patch_size),
        threshold     = args.threshold,
        base_features = args.base_features,
        ntfy_topic    = args.ntfy_topic,
        gt_mask_path  = args.gt_mask_path,
    )
