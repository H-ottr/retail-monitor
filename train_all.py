"""
=============================================================
RetailGuard AI — Combined 5-Behavior Training Script
=============================================================
Datasets:
  1. Normal          — person-3ese7
  2. Running         — running-man
  3. Loitering       — who-s-loitering
  4. Suspicious Shelf— shoplifting-uofcd
  5. Crowd Formation — crowd-detection-i75bl

Run: python train_all.py
=============================================================
"""

from ultralytics import YOLO
from roboflow import Roboflow
import os, shutil, yaml, torch
from pathlib import Path

API_KEY = "fjJROslW0URyOW34pwOE"


def download_datasets():
    rf = Roboflow(api_key=API_KEY)
    datasets = []

    print("=" * 60)
    print("  RetailGuard AI — Downloading All 5 Datasets")
    print("=" * 60)

    # 1. Normal behavior
    print("\n[1/5] Downloading Normal (Person Walking) dataset...")
    try:
        ds1 = rf.workspace("person-j3bp5").project("person-3ese7").version(1).download("yolov8")
        datasets.append({"path": ds1.location, "behavior": "normal", "new_class_id": 0})
        print("      Done!")
    except Exception as e:
        print(f"      Warning: {e}")

    # 2. Running
    print("\n[2/5] Downloading Running dataset...")
    try:
        ds2 = rf.workspace("tracksport").project("running-man").version(1).download("yolov8")
        datasets.append({"path": ds2.location, "behavior": "running", "new_class_id": 1})
        print("      Done!")
    except Exception as e:
        print(f"      Warning: {e}")

    # 3. Loitering
    print("\n[3/5] Downloading Loitering dataset...")
    try:
        ds3 = rf.workspace("seeing-algotrial-2").project("who-s-loitering").version(2).download("yolov8")
        datasets.append({"path": ds3.location, "behavior": "loitering", "new_class_id": 2})
        print("      Done!")
    except Exception as e:
        print(f"      Warning: {e}")

    # 4. Suspicious Shelf Interaction (Shoplifting)
    print("\n[4/5] Downloading Shoplifting dataset...")
    try:
        ds4 = rf.workspace("cpm").project("shoplifting-uofcd").version(9).download("yolov8")
        datasets.append({"path": ds4.location, "behavior": "suspicious", "new_class_id": 3})
        print("      Done!")
    except Exception as e:
        print(f"      Warning: {e}")

    # 5. Crowd Formation
    print("\n[5/5] Downloading Crowd Formation dataset...")
    try:
        ds5 = rf.workspace("institut-teknologi-nasional-bandung-mxgtc").project("crowd-detection-i75bl").version(2).download("yolov8")
        datasets.append({"path": ds5.location, "behavior": "crowd", "new_class_id": 4})
        print("      Done!")
    except Exception as e:
        print(f"      Warning: {e}")

    print(f"\n✓ Successfully downloaded {len(datasets)}/5 datasets")
    return datasets


def remap_labels(src_label_dir, dst_label_dir, new_class_id, prefix):
    src = Path(src_label_dir)
    if not src.exists():
        return 0
    count = 0
    for txt_file in src.glob("*.txt"):
        new_lines = []
        with open(txt_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    parts[0] = str(new_class_id)
                    new_lines.append(" ".join(parts))
        if new_lines:
            dst_file = Path(dst_label_dir) / f"{prefix}_{txt_file.name}"
            with open(dst_file, "w") as f:
                f.write("\n".join(new_lines))
            count += 1
    return count


def merge_datasets(datasets):
    print("\n" + "=" * 60)
    print("  Merging All Datasets...")
    print("=" * 60)

    MERGED_PATH = "data/merged_dataset"
    for split in ["train", "val"]:
        os.makedirs(f"{MERGED_PATH}/images/{split}", exist_ok=True)
        os.makedirs(f"{MERGED_PATH}/labels/{split}", exist_ok=True)

    total_train = 0
    total_val   = 0

    for ds in datasets:
        path     = ds["path"]
        behavior = ds["behavior"]
        cls_id   = ds["new_class_id"]
        prefix   = behavior

        print(f"\n  Processing [{behavior.upper()}] from {path}")

        for split_name in ["train", "valid", "val", "test"]:
            img_src = Path(f"{path}/{split_name}/images")
            lbl_src = Path(f"{path}/{split_name}/labels")

            if not img_src.exists():
                continue

            split_dst = "train" if split_name == "train" else "val"
            img_dst   = Path(f"{MERGED_PATH}/images/{split_dst}")
            copied    = 0

            for img in img_src.glob("*.*"):
                if img.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    shutil.copy(img, img_dst / f"{prefix}_{img.name}")
                    copied += 1

            lbl_dst   = f"{MERGED_PATH}/labels/{split_dst}"
            lbl_count = remap_labels(lbl_src, lbl_dst, cls_id, prefix)

            if split_dst == "train":
                total_train += copied
            else:
                total_val   += copied

            print(f"    {split_name} → {split_dst}: {copied} images, {lbl_count} labels")

    print(f"\n{'='*60}")
    print(f"  Merge Complete!")
    print(f"  Total train images : {total_train}")
    print(f"  Total val images   : {total_val}")
    print(f"{'='*60}")
    return MERGED_PATH, total_train, total_val


def create_yaml(merged_path):
    combined_yaml = {
        "path": os.path.abspath(merged_path),
        "train": "images/train",
        "val":   "images/val",
        "nc": 5,
        "names": [
            "Normal",
            "Running",
            "Loitering",
            "SuspiciousShelfInteraction",
            "CrowdFormation"
        ]
    }
    os.makedirs("data", exist_ok=True)
    yaml_path = "data/retail_combined.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(combined_yaml, f, default_flow_style=False)
    print("\n✓ Combined dataset.yaml created!")
    print(f"  Classes : {combined_yaml['names']}")
    return yaml_path


def check_gpu():
    print("\n" + "=" * 60)
    print("  GPU Check")
    print("=" * 60)
    if torch.cuda.is_available():
        gpu  = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU  : {gpu}")
        print(f"  VRAM : {vram:.1f} GB")
        return "0", 8
    else:
        print("  No GPU found — using CPU")
        return "cpu", 4


def train_model(yaml_path, device, batch, total_train, total_val):
    print("\n" + "=" * 60)
    print("  Starting YOLOv8 Training — All 5 Behaviors")
    print("=" * 60)
    print(f"  Device : {'GTX 1650 GPU' if device == '0' else 'CPU'}")
    print(f"  Batch  : {batch}")
    print(f"  Epochs : 100")
    print(f"  Images : {total_train} train / {total_val} val")
    print("  This may take 2-4 hours on GTX 1650")
    print("=" * 60 + "\n")

    model = YOLO("yolov8s.pt")

    results = model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=batch,
        device=device,
        workers=0,       # Windows fix — must be 0
        project="model/runs",
        name="retail_all_behaviors",
        pretrained=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        fliplr=0.5,
        degrees=10.0,
        scale=0.5,
        patience=20,
        save=True,
        save_period=10,
        plots=True,
        verbose=True,
        amp=False,       # Disable AMP — fixes GTX 1650 NaN issue
    )

    print("\n✓ Training Complete!")

    best_src = "model/runs/retail_all_behaviors/weights/best.pt"
    best_dst = "model/best.pt"
    if os.path.exists(best_src):
        shutil.copy(best_src, best_dst)
        print(f"✓ best.pt saved to: {best_dst}")
        print("\n" + "=" * 60)
        print("  YOUR MODEL IS READY!")
        print("  Run the app : cd app && python app.py")
        print("  Open browser: http://localhost:5000")
        print("=" * 60)
    else:
        print(f"Warning: best.pt not found at {best_src}")


# ─────────────────────────────────────────────────────────
# Windows requires if __name__ == '__main__'
# ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    datasets                         = download_datasets()
    merged_path, total_train, total_val = merge_datasets(datasets)
    yaml_path                        = create_yaml(merged_path)
    device, batch                    = check_gpu()
    train_model(yaml_path, device, batch, total_train, total_val)
