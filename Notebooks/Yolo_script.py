"""
YOLOv8-medium Training Script — Esophagitis Detection
Dataset: Roboflow YOLO format (train/valid/test split)

Usage:
    1. Install dependency:
           pip install ultralytics

    2. Place your dataset so the structure looks like:
           dataset/
               data.yaml
               train/images/  train/labels/
               valid/images/  valid/labels/
               test/images/   test/labels/   (optional)

    3. Set DATASET_ROOT below to the absolute path of that folder.

    4. Run:
           python train_yolov8.py
"""

import os
import shutil
from pathlib import Path

from ultralytics import YOLO

# ─────────────────────────────────────────────
# CONFIG — edit these before running
# ─────────────────────────────────────────────

# Absolute path to the folder that contains data.yaml
DATASET_ROOT = "D:\\AI endoscopy\\Dataset"   # <-- CHANGE THIS

# Training hyper-parameters
EPOCHS       = 100
IMG_SIZE     = 640
BATCH_SIZE   = 4           # REDUCED: 16 is too large for CPU, use 4 for CPU training
PATIENCE     = 20          # early stopping: stop after N epochs with no improvement
LR0          = 0.01        # initial learning rate
WORKERS      = 0           # Must be 0 for Windows (dataloader single-threaded)
DEVICE       = "cpu"       # Explicitly use CPU (change to "0" if you have CUDA GPU)

# Where to save runs
PROJECT_DIR  = "runs/esophagitis"
RUN_NAME     = "yolov8m_v1"

# Pretrained weights (downloaded automatically on first run)
WEIGHTS      = "yolov8m.pt"

# ─────────────────────────────────────────────
# VALIDATE DATASET PATH
# ─────────────────────────────────────────────

dataset_root = Path(DATASET_ROOT).resolve()
data_yaml    = dataset_root / "data.yaml"

if not data_yaml.exists():
    raise FileNotFoundError(
        f"data.yaml not found at: {data_yaml}\n"
        f"Please set DATASET_ROOT to the folder that contains data.yaml."
    )

# Fix relative paths inside data.yaml so they resolve correctly
# regardless of where you run the script from.
import yaml

with open(data_yaml) as f:
    cfg = yaml.safe_load(f)

updated = False
for key in ("train", "val", "test"):
    if key in cfg and cfg[key] is not None:
        p = Path(cfg[key])
        if not p.is_absolute():
            # Resolve relative to data.yaml's directory
            resolved = (dataset_root / p).resolve()
            if resolved.exists():
                cfg[key] = str(resolved)
                updated = True
            else:
                # Try resolving relative to dataset_root directly
                direct = (dataset_root / key / "images").resolve()
                if direct.exists():
                    cfg[key] = str(direct)
                    updated = True

# Write a patched copy next to the original so we don't mutate the user's file
patched_yaml = dataset_root / "data_abs.yaml"
with open(patched_yaml, "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)

print(f"Dataset root : {dataset_root}")
print(f"Config used  : {patched_yaml}")
print(f"Classes      : {cfg.get('names')} (nc={cfg.get('nc')})")
print()

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        print(f"Loading model from: {WEIGHTS}")
        model = YOLO(WEIGHTS)   # loads pretrained COCO weights
        print(f"Model loaded successfully. Starting training with:")
        print(f"  - Batch size: {BATCH_SIZE}")
        print(f"  - Workers: {WORKERS}")
        print(f"  - Device: {DEVICE}")
        print(f"  - Epochs: {EPOCHS}")

        results = model.train(
            data      = str(patched_yaml),
            epochs    = EPOCHS,
            imgsz     = IMG_SIZE,
            batch     = BATCH_SIZE,
            patience  = PATIENCE,
            lr0       = LR0,
            workers   = WORKERS,
            device    = DEVICE,
            project   = PROJECT_DIR,
            name      = RUN_NAME,

            # Augmentation (sensible defaults for medical imaging)
            hsv_h     = 0.015,   # hue jitter
            hsv_s     = 0.4,     # saturation jitter
            hsv_v     = 0.4,     # brightness jitter
            flipud    = 0.5,     # vertical flip (endoscopy images have no fixed orientation)
            fliplr    = 0.5,     # horizontal flip
            mosaic    = 1.0,     # mosaic augmentation
            mixup     = 0.1,     # mixup augmentation

            # Logging & saving
            save        = True,
            save_period = 10,    # checkpoint every N epochs (0 = only best/last)
            plots       = True,  # save training plots
            verbose     = True,
        )

        # ─────────────────────────────────────────────
        # VALIDATE ON VALIDATION SET
        # ─────────────────────────────────────────────

        print("\n" + "="*60)
        print("VALIDATION ON BEST CHECKPOINT")
        print("="*60)

        best_weights = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
        best_model   = YOLO(str(best_weights))

        val_results = best_model.val(
            data    = str(patched_yaml),
            imgsz   = IMG_SIZE,
            batch   = BATCH_SIZE,
            device  = DEVICE,
            split   = "val",
        )

        print(f"\nmAP50      : {val_results.box.map50:.4f}")
        print(f"mAP50-95   : {val_results.box.map:.4f}")
        print(f"Precision  : {val_results.box.mp:.4f}")
        print(f"Recall     : {val_results.box.mr:.4f}")

        # ─────────────────────────────────────────────
        # EXPORT SAVED MODEL INFO
        # ─────────────────────────────────────────────

        print("\n" + "="*60)
        print("SAVED MODEL LOCATIONS")
        print("="*60)
        run_dir = Path(PROJECT_DIR) / RUN_NAME
        print(f"Best weights : {run_dir / 'weights' / 'best.pt'}")
        print(f"Last weights : {run_dir / 'weights' / 'last.pt'}")
        print(f"Results dir  : {run_dir}")

        # ─────────────────────────────────────────────
        # OPTIONAL: EXPORT TO ONNX FOR DEPLOYMENT
        # ─────────────────────────────────────────────

        EXPORT_ONNX = False   # set True if you want an ONNX model for deployment

        if EXPORT_ONNX:
            print("\nExporting to ONNX...")
            best_model.export(format="onnx", imgsz=IMG_SIZE, dynamic=True)
            print(f"ONNX model saved to: {run_dir / 'weights' / 'best.onnx'}")

    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("TRAINING FAILED — FULL ERROR BELOW")
        print("="*60)
        traceback.print_exc()
        print("\n" + "="*60)
        input("Press Enter to exit...")   # keeps the PowerShell window open