"""
=============================================================
Retail Shop Behavior Detection — YOLOv8 Training Script
=============================================================
Usage:
    python train.py

Before running:
  1. Download datasets (see model/dataset.yaml for links)
  2. Organize images into:
       data/images/train/<class>/
       data/images/val/<class>/
       data/labels/train/<class>/   (YOLO format .txt annotations)
       data/labels/val/<class>/
  3. Adjust CONFIG below as needed
=============================================================
"""

from ultralytics import YOLO
import os
import yaml

# ─────────────────────────────────────────
# CONFIG — adjust before training
# ─────────────────────────────────────────
CONFIG = {
    "model":        "yolov8n.pt",       # Base model: yolov8n/s/m/l/x (n=fastest, x=most accurate)
    "data":         "model/dataset.yaml",
    "epochs":       100,
    "imgsz":        640,
    "batch":        16,
    "patience":     20,                 # Early stopping patience
    "device":       "0",                # GPU device (use 'cpu' if no GPU)
    "workers":      4,
    "project":      "runs/train",
    "name":         "retail_behavior",
    "pretrained":   True,
    "optimizer":    "AdamW",
    "lr0":          0.001,
    "lrf":          0.01,
    "momentum":     0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "augment":      True,
    "mosaic":       1.0,
    "mixup":        0.1,
    "copy_paste":   0.1,
    "degrees":      10.0,
    "flipud":       0.0,
    "fliplr":       0.5,
    "scale":        0.5,
    "shear":        2.0,
    "perspective":  0.0001,
    "hsv_h":        0.015,
    "hsv_s":        0.7,
    "hsv_v":        0.4,
    "conf":         0.25,
    "iou":          0.45,
    "save":         True,
    "save_period":  10,
    "plots":        True,
    "verbose":      True,
}


def train():
    print("=" * 60)
    print("  Retail Behavior Detection — YOLOv8 Training")
    print("=" * 60)

    # Validate dataset config exists
    if not os.path.exists(CONFIG["data"]):
        raise FileNotFoundError(
            f"Dataset config not found: {CONFIG['data']}\n"
            "Make sure model/dataset.yaml exists and data paths are correct."
        )

    # Load YOLO model
    print(f"\n[INFO] Loading base model: {CONFIG['model']}")
    model = YOLO(CONFIG["model"])

    # Start training
    print(f"[INFO] Starting training for {CONFIG['epochs']} epochs...")
    results = model.train(
        data=CONFIG["data"],
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        patience=CONFIG["patience"],
        device=CONFIG["device"],
        workers=CONFIG["workers"],
        project=CONFIG["project"],
        name=CONFIG["name"],
        pretrained=CONFIG["pretrained"],
        optimizer=CONFIG["optimizer"],
        lr0=CONFIG["lr0"],
        lrf=CONFIG["lrf"],
        momentum=CONFIG["momentum"],
        weight_decay=CONFIG["weight_decay"],
        warmup_epochs=CONFIG["warmup_epochs"],
        augment=CONFIG["augment"],
        mosaic=CONFIG["mosaic"],
        mixup=CONFIG["mixup"],
        copy_paste=CONFIG["copy_paste"],
        degrees=CONFIG["degrees"],
        flipud=CONFIG["flipud"],
        fliplr=CONFIG["fliplr"],
        scale=CONFIG["scale"],
        shear=CONFIG["shear"],
        perspective=CONFIG["perspective"],
        hsv_h=CONFIG["hsv_h"],
        hsv_s=CONFIG["hsv_s"],
        hsv_v=CONFIG["hsv_v"],
        conf=CONFIG["conf"],
        iou=CONFIG["iou"],
        save=CONFIG["save"],
        save_period=CONFIG["save_period"],
        plots=CONFIG["plots"],
        verbose=CONFIG["verbose"],
    )

    # Copy best weights to model/ folder
    best_weights = f"{CONFIG['project']}/{CONFIG['name']}/weights/best.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy(best_weights, "model/best.pt")
        print(f"\n[✓] Best model saved to: model/best.pt")

    print("\n[✓] Training complete!")
    print(f"    Results saved to: {CONFIG['project']}/{CONFIG['name']}/")
    print(f"    Best weights:     model/best.pt")

    # Validation
    print("\n[INFO] Running validation on best model...")
    model_best = YOLO("model/best.pt")
    val_results = model_best.val(
        data=CONFIG["data"],
        imgsz=CONFIG["imgsz"],
        conf=CONFIG["conf"],
        iou=CONFIG["iou"],
    )
    print(f"\n[✓] Validation mAP@0.5: {val_results.box.map50:.4f}")
    print(f"[✓] Validation mAP@0.5:0.95: {val_results.box.map:.4f}")


if __name__ == "__main__":
    train()
