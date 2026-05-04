# 3D Structural & Volumetric Analysis Pipeline

> Modular agentic pipeline for 3D volumetric image segmentation using a custom PyTorch 3D U-Net. Designed for extracting dimensional measurements from complex engineering geometries (e.g., battery modules, structural components). Includes real-time push notifications via ntfy API.

---

## Project overview

This project implements an end-to-end 3D segmentation system that:
- Takes raw 3D volumetric data (CT scans, engineering scans) as input
- Segments regions of interest using a custom **3D U-Net** (PyTorch)
- Runs efficient **sliding-window inference** for large volumes
- Sends **real-time notifications** via ntfy.sh upon completion
- Reports Dice, IoU, Precision, Recall, and Volume Similarity metrics

---

## Architecture

```
3d-volumetric-segmentation/
├── src/
│   ├── model.py        # 3D U-Net architecture (encoder–decoder + skip connections)
│   ├── dataset.py      # Dataset loader + 3D augmentations (flip, random crop)
│   ├── train.py        # Training loop (Dice + CE loss, cosine LR scheduling)
│   ├── inference.py    # Sliding-window inference + ntfy push notifications
│   └── metrics.py      # Dice, IoU, F1, Volume Similarity
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/PRATdoppelEK/3d-volumetric-segmentation.git
cd 3d-volumetric-segmentation
pip install -r requirements.txt
```

---

## Quickstart

### Train with synthetic data (no dataset needed — runs immediately)
```bash
python src/train.py --synthetic --epochs 20 --batch_size 2
```

### Train with your own volumetric data
```bash
# data_dir must contain:  images/*.npy  and  masks/*.npy
python src/train.py --data_dir ./data --epochs 50 --batch_size 2 --lr 1e-4
```

### Run inference with ntfy push notification
```bash
python src/inference.py \
  --model_path  models/best_model.pth \
  --input_path  data/sample/volume.npy \
  --output_path results/prediction.npy \
  --ntfy_topic  your-ntfy-topic-name
```

---

## Results (synthetic benchmark)

| Metric | Value |
|--------|-------|
| Dice Score | ~0.91 |
| IoU | ~0.84 |
| Inference time (64³ volume) | < 0.5s on CPU |

---

## Key technical highlights

- **3D U-Net**: Encoder–decoder with skip connections for precise spatial localisation
- **Combined loss**: Dice Loss + Cross-Entropy for handling class imbalance in volumetric data
- **Sliding-window inference**: Handles arbitrary volume sizes with configurable overlap
- **Data augmentation**: Random 3D flipping and patch cropping during training
- **ntfy integration**: Lightweight real-time push notification on inference completion

---

## Tech stack

`PyTorch` · `NumPy` · `scikit-image` · `NiBabel` · `ntfy API` · `Python 3.10+`

---

## Author

**Prateek Gaur** — ML Engineer | Battery & Engineering AI
[LinkedIn](https://www.linkedin.com/in/prateek-gaur-15a629b4) · [GitHub](https://github.com/PRATdoppelEK) · prateekgaur@gmx.de
