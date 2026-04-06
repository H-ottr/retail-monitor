# 🛒 RetailGuard AI — Setup Guide

A step-by-step guide to get the retail monitoring platform running on your local machine after cloning the repository.

---

## 📋 Prerequisites

Before you begin, make sure the following are installed on your system:

| Requirement | Version | Notes |
|---|---|---|
| **Python** | 3.9 or higher | [Download here](https://www.python.org/downloads/) |
| **pip** | Latest | Comes with Python |
| **Git** | Any | [Download here](https://git-scm.com/) |
| **Webcam** *(optional)* | — | Only needed for live camera monitoring |

> **Windows users:** Make sure Python is added to your `PATH` during installation.

---

## 🚀 Step-by-Step Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd retail-monitor
```

---

### 2. Create a Virtual Environment

It is strongly recommended to use a virtual environment to avoid package conflicts.

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

> You should see `(.venv)` appear at the start of your terminal prompt — this means the virtual environment is active.

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including Flask, OpenCV, Ultralytics (YOLOv8), and SocketIO.

---

### 4. Download Model Weights

The `.gitignore` excludes `.pt` model files from the repository (they are too large to commit). You need to download them manually.

#### 4a. YOLOv8 Base Model (`yolov8n.pt`)

Place `yolov8n.pt` in the **root** of the project directory:

```
retail-monitor/
├── yolov8n.pt          ← Place here
├── app/
├── model/
└── ...
```

You can download it automatically by running:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```
This will auto-download `yolov8n.pt` from the Ultralytics servers on first run.

#### 4b. Custom Suspicious Shelf Model (`model/best.pt`)

This is a custom-trained model for detecting suspicious shelf interactions. Place it here:

```
retail-monitor/
└── model/
    └── best.pt         ← Place here
```

> ⚠️ **Note:** If `model/best.pt` is not present, the system will still run. Suspicious shelf interaction detection will fall back to a basic bounding-box overlap method. All other detections (Running, Loitering, Crowd Formation) will work normally.

---

### 5. Create Required Directories

The app needs some empty directories to store uploaded videos and incident snapshots. Run the following commands from the project root:

**Windows (PowerShell):**
```powershell
New-Item -ItemType Directory -Force -Path "app\static\uploads\incidents"
New-Item -ItemType Directory -Force -Path "app\static\processed"
```

**macOS / Linux:**
```bash
mkdir -p app/static/uploads/incidents
mkdir -p app/static/processed
```

---

### 6. Run the Application

Navigate into the `app` directory and start the server:

**Windows:**
```powershell
cd app
python app.py
```

**macOS / Linux:**
```bash
cd app
python3 app.py
```

Alternatively, from the project root on Linux/macOS you can use the quick-start script:
```bash
bash run.sh
```

---

### 7. Open the Platform

Once the server starts, open your browser and go to:

```
http://localhost:5000
```

---

## 🔑 Default Login Credentials

| Role | Username | Password |
|---|---|---|
| **Admin** | `admin` | `admin123` |
| **Staff** | `staff` | `staff123` |

> ⚠️ **Security Notice:** Change these default passwords immediately if you plan to deploy this on a network or expose it publicly.

---

## 🖥️ Platform Features

Once logged in, you can access the following pages:

| Page | URL | Description |
|---|---|---|
| **Dashboard** | `/dashboard` | Overview stats, incident chart, live camera status |
| **Live Monitor** | `/live` | Real-time webcam feed with AI behavior detection |
| **Video Analysis** | `/video-analysis` | Upload & analyze recorded video files |
| **Alerts** | `/alerts` | History of all detected incidents |
| **Analytics** | `/analytics` | Behavior trend charts |
| **Logs** | `/logs` | System activity logs |
| **Settings** | `/settings` | Application settings |

---

## 🧠 How Detection Works

The system uses two AI models in a hybrid pipeline:

- **`yolov8n.pt`** — General-purpose person tracking. Detects **Running**, **Loitering**, and **Crowd Formation** using math-based rules (velocity, dwell time, proximity).
- **`model/best.pt`** *(custom)* — Detects **Suspicious Shelf Interactions** (e.g., concealing items). Uses a 30-frame temporal voting window to reduce false positives.

---

## 🗃️ Project Structure

```
retail-monitor/
├── app/
│   ├── app.py                  # Main Flask application
│   ├── retail_monitor.db       # SQLite database (auto-created on first run)
│   ├── static/
│   │   ├── uploads/            # Uploaded videos & incident snapshots
│   │   └── processed/          # AI-processed output videos
│   └── templates/              # HTML templates (Jinja2)
├── model/
│   └── best.pt                 # Custom shelf-interaction model (not in repo)
├── yolov8n.pt                  # Base YOLOv8 model (not in repo)
├── requirements.txt            # Python dependencies
├── run.sh                      # Quick-start script (Linux/macOS)
└── setup.md                    # This file
```

---

## 🛠️ Troubleshooting

### `ModuleNotFoundError: No module named 'ultralytics'`
You forgot to activate your virtual environment or install dependencies.
```bash
# Activate venv first, then:
pip install -r requirements.txt
```

### Camera not opening / `Cannot open camera`
- Make sure no other app is using your webcam.
- On Windows, the app uses `cv2.CAP_DSHOW` for faster camera initialization. If the camera still fails, check your webcam drivers.

### `yolov8n.pt not found`
Run this once to auto-download it:
```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Port 5000 already in use
Kill the existing process or change the port in `app.py` at the bottom of the file:
```python
socketio.run(app, host="0.0.0.0", port=5000, ...)
# Change 5000 to e.g. 5001
```

### Database errors on first run
The SQLite database is created automatically. If you see schema errors, delete the old database and restart:
```bash
del app\retail_monitor.db     # Windows
# OR
rm app/retail_monitor.db      # macOS/Linux
```

---

## 📦 Dependencies Summary

| Package | Version | Purpose |
|---|---|---|
| `flask` | 3.0.0 | Web framework |
| `flask-socketio` | 5.3.6 | Real-time WebSocket alerts |
| `flask-login` | 0.6.3 | User session management |
| `flask-sqlalchemy` | 3.1.1 | Database ORM |
| `ultralytics` | 8.1.0 | YOLOv8 AI inference |
| `opencv-python` | 4.9.0.80 | Video capture & frame processing |
| `numpy` | 1.26.4 | Numerical operations |
| `eventlet` | 0.35.1 | Async server for SocketIO |
| `python-dotenv` | 1.0.1 | Environment variable support |

---

*Built with Flask + YOLOv8 + OpenCV. For questions or issues, open a GitHub Issue.*
