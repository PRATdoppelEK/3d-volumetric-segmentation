# 3D Structural & Volumetric Analysis Pipeline

> Modular agentic pipeline for 3D volumetric image segmentation using a custom PyTorch 3D U-Net. Designed for extracting dimensional measurements from complex engineering geometries (e.g., battery modules, structural components). Includes real-time push notifications via ntfy API.

---

> **📌 About this repo:** This is a showcase repo — write-up, architecture, and results only. The full implementation lives in a private core repo (kept private to protect the code and as a backup) and is available for a live walkthrough on request, e.g. in an interview.

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
│   ├── metrics.py      # Dice, IoU, F1, Volume Similarity
│   └── baseline.py     # Classical baseline: Gaussian blur + Otsu threshold
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

## 📊 Results (Synthetic Benchmark)

### Training output — 20 epochs, 3D U-Net on Apple M2 (MPS)

![Training output](screenshots/training_output.png)

> **Synthetic data corrected (Oct 2026).** The first synthetic generator produced pure-noise
> volumes with the *same* centred sphere as the mask every time, so the network could score high
> Dice without looking at the input; the Dice 0.98 reported from it is withdrawn. Each sample now has
> a visible ellipsoid (brighter than the background, with noise) at a random position and size.
> The screenshot above is from the earlier generator.

| Method | Val Dice | Val IoU |
|--------|----------|---------|
| Gaussian blur (σ = 1) + Otsu threshold — no training, no labels (`python src/baseline.py`) | 0.985 | 0.970 |
| 3D U-Net, best epoch (12 of 20) | 0.942 | 0.891 |
| 3D U-Net, final epoch | 0.933 | 0.875 |

*Apple M2 (MPS), 20 epochs, batch size 2, 64³ volumes, 40 synthetic volumes (32 train / 8 val,
random split per run). The baseline is averaged over all 40 volumes.*

**Finding:** on this benchmark the classical baseline wins. One bright ellipsoid on a uniform
background can be segmented from intensity alone after light smoothing, so the benchmark proves the
pipeline runs end to end but not what the U-Net adds. That needs data where intensity is not enough:
touching parts, distractors with the same density, intensity inhomogeneity — i.e. real CT scans or a
harder synthetic set. The best-epoch Dice is picked on the same 8 validation volumes it is reported
on, so it is optimistic.

### Key observations

- **3D U-Net vs 2D slice-by-slice**: 3D convolutions use spatial continuity along the Z-axis that 2D slice models lose (a 2D baseline comparison is a planned next step)
- **Sliding window inference** handles arbitrarily large volumes without memory overflow — tested up to 256³ on 16GB RAM
- **Combined Dice + Cross-Entropy loss** is used for class-imbalanced geometry data (a Dice-only ablation has not been run)
- **Engineering geometry application**: dimensional measurements (bounding box, volume, centroid) are extracted from the segmented regions
- **ntfy integration** reduces monitoring overhead — inference jobs run unattended with push notification on completion

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
