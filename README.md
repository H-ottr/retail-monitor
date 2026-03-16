# RetailGuard AI — Retail Shop Monitoring System

An AI-powered retail shop monitoring web application built with **YOLOv8 + Flask**.
Detects 5 behavior categories in real time with visual & sound alerts.

---

## 📁 Project Structure

```
retail-monitor/
├── app/
│   ├── app.py                  ← Flask backend (main entry point)
│   ├── templates/
│   │   ├── base.html           ← Shared layout (sidebar, header)
│   │   ├── login.html          ← Login page
│   │   ├── dashboard.html      ← Overview dashboard
│   │   ├── live.html           ← Live camera monitoring
│   │   ├── video_analysis.html ← Offline video processing
│   │   ├── alerts.html         ← Incidents table
│   │   ├── analytics.html      ← Charts & trends
│   │   ├── logs.html           ← Activity log
│   │   └── settings.html       ← System configuration
│   └── static/
│       ├── uploads/            ← Uploaded videos + incident frames
│       └── processed/          ← Processed output videos
├── model/
│   ├── train.py                ← YOLOv8 training script
│   ├── dataset.yaml            ← Dataset configuration
│   └── best.pt                 ← Trained model weights (after training)
├── data/
│   ├── images/
│   │   ├── train/              ← Training images
│   │   └── val/                ← Validation images
│   └── labels/
│       ├── train/              ← YOLO format labels (.txt)
│       └── val/
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application (without training)

```bash
cd app
python app.py
```

Visit **http://localhost:5000**

Default credentials:
- Admin: `admin` / `admin123`
- Staff: `staff` / `staff123`

> The app runs with YOLOv8n (pretrained person detection) until you train your custom model.

---

## 🤖 Training Your Custom Model

### Step 1: Download Datasets

| Behavior | Recommended Dataset |
|---|---|
| Normal | [DCSASS (Kaggle)](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset) |
| Running | [UCF-Crime (Kaggle)](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) |
| Loitering | [DCSASS](https://www.kaggle.com/datasets/mateohervas/dcsass-dataset) + [Roboflow](https://universe.roboflow.com) |
| Suspicious Shelf Interaction | [PoseLift (GitHub)](https://github.com/TeCSAR-UNCC/PoseLift) |
| Crowd Formation | [UCF-Crime](https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset) |

### Step 2: Organize Data

```
data/
  images/
    train/   ← put training images here (.jpg/.png)
    val/     ← put validation images here
  labels/
    train/   ← YOLO format .txt files (one per image)
    val/
```

**YOLO label format** (one line per detected object in image):
```
<class_id> <x_center> <y_center> <width> <height>
```

Class IDs: `0=Normal, 1=Running, 2=Loitering, 3=SuspiciousShelfInteraction, 4=CrowdFormation`

Example label file:
```
1 0.523 0.441 0.134 0.562
0 0.201 0.380 0.110 0.420
```

### Step 3: Annotate Videos

If using video datasets, extract frames and annotate using:
- **Roboflow** (recommended, free tier available): https://roboflow.com
- **LabelImg**: `pip install labelImg && labelImg`
- **CVAT**: https://cvat.org (open source)

### Step 4: Train the Model

```bash
# From project root
python model/train.py
```

Training config in `model/train.py`:
```python
CONFIG = {
    "model":   "yolov8n.pt",   # Start with nano, upgrade to yolov8s/m for better accuracy
    "epochs":  100,
    "imgsz":   640,
    "batch":   16,             # Reduce if GPU memory error
    "device":  "0",            # Use "cpu" if no GPU
}
```

Best weights automatically saved to `model/best.pt`.

### Step 5: Verify Model Works

```python
from ultralytics import YOLO
model = YOLO("model/best.pt")
results = model("path/to/test_image.jpg")
results[0].show()
```

---

## 🎨 Design Theme

The application uses the theme specified in the project brief:
- **Background**: White (`#ffffff`) and light grey (`#f0f4f8`)
- **Accent**: Blue (`#0ea5e9`) and Teal (`#14b8a6`) for headers, buttons, navigation
- **Alerts**: Red (`#ef4444`) and Orange (`#f97316`) for warnings and anomalies
- **Success**: Green (`#22c55e`) for normal behavior

---

## 📊 Behavior Classes

| ID | Class | Description |
|---|---|---|
| 0 | Normal | Walking, browsing, standard shopping |
| 1 | Running | Unusually fast movement inside store |
| 2 | Loitering | Staying in one area too long |
| 3 | Suspicious Shelf Interaction | Unusual handling of products/shelves |
| 4 | Crowd Formation | Multiple people gathering unexpectedly |

---

## 🔧 Technical Stack

| Component | Technology |
|---|---|
| AI Model | YOLOv8 (Ultralytics) |
| Video Processing | OpenCV |
| Backend | Flask + Flask-SocketIO |
| Database | SQLite (via SQLAlchemy) |
| Frontend | HTML, CSS, JavaScript |
| Charts | Chart.js |
| Real-time | WebSockets (Socket.IO) |
| Auth | Flask sessions + Werkzeug hashing |

---

## ⚙️ Configuration

Edit `app/app.py` to change:
- `MODEL_PATH` — path to your trained `.pt` file
- `ANOMALY_CLASSES` — which class IDs trigger alerts
- `BEHAVIOR_COLORS` — bounding box colors per class
- Database URI — switch from SQLite to MySQL for production

---

## 📝 Notes for Production

1. Change default passwords in `app.py` `init_db()` function
2. Set a strong `app.secret_key`
3. Use `gunicorn` + `nginx` instead of Flask dev server
4. Switch SQLite to MySQL/PostgreSQL for scale
5. Configure HTTPS for secure camera streaming
