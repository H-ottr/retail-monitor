# 🛒 RetailGuard AI — Smart Retail Monitoring Platform

[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-yellow.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-blue.svg)](https://ultralytics.com)
[![Flask](https://img.shields.io/badge/Web-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)

**RetailGuard AI** is a state-of-the-art retail monitoring system that uses a hybrid computer vision pipeline to detect suspicious activities and ensure shop floor safety. Built on a zero-lag architecture, it provides real-time insights, instant alerts, and detailed historical analytics.

---

## 🧠 Core Intelligence: The Hybrid CV Engine

Unlike traditional systems that rely on a single model, **RetailGuard AI** employs a specialized dual-model inference pipeline:

1.  **Primary Model (`yolov8n.pt`)**: Optimized for high-speed person tracking. It calculates real-time math for:
    *   **🏃 Running**: Velocity-based detection from tracking history.
    *   **🚶 Loitering**: Dwell-time monitoring in sensitive zones.
    *   **👥 Crowd Formation**: Proximity-based analysis (distance calculation between tracked persons).

2.  **Specialized Model (`model/best.pt`)**: A custom-trained YOLOv8 model dedicated to:
    *   **🛡️ Suspicious Shelf Interaction**: Identifying unusual handling or concealment of items.

### 🛡️ Advanced Temporal Logic
To virtually eliminate false positives (e.g., a customer just reaching for a high shelf), we implement:
*   **Temporal Voting Window**: Suspicious behavior must persist for a **30-frame sliding window** to trigger a confirmed alert.
*   **60-Frame Alert Decay**: Once confirmed, alerts remain active briefly to ensure operators catch the incident even if the behavior pauses.

---

## 🖥️ Platform Modules

### 📹 Zero-Lag Live Monitoring
Our dedicated **Dual-Threaded Camera Engine** uses one thread to strip the camera buffer (preventing backlog) and another for AI inference. This ensures you see the freshest frame possible with 0.1s latency.

### 📊 Intelligence Dashboard
*   **Real-time Stats**: Hourly incident trends and behavior distribution.
*   **Live Event Feed**: Instant notification stream powered by **Socket.IO**.
*   **UI Aesthetic**: Professional dark-themed sidebar with a clean white/teal monitoring workspace.

### 📼 Video Analysis Lab
Need to review past footage? Upload any `.mp4`, `.avi`, or `.mov` file to our analysis lab. It will process the video frame-by-frame, applying the same hybrid AI logic, and generate a downloadable annotated report video.

### 📜 Incident & Audit Logs
Every alert captures a high-resolution snapshot stored in our `static/uploads/incidents` folder. Complete audit logs track system activity, logins, and camera operations.

---

## 🚀 Quick Start

### 1. Installation
Ensure you have the virtual environment ready and dependencies installed.
```bash
pip install -r requirements.txt
```

### 2. Models
The system looks for `yolov8n.pt` in the root and `model/best.pt` in the model folder.
> *See [setup.md](setup.md) for detailed download instructions.*

### 3. Run
```bash
cd app
python app.py
```
Visit: **http://localhost:5000**

**Default Credentials:**
*   **Admin**: `admin` / `admin123`
*   **Staff**: `staff` / `staff123`

---

## 🛠️ Technical Architecture

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Backend** | Flask | Core Web Logic & API |
| **AI Runtime** | Ultralytics YOLOv8 | Inference & Tracking |
| **Computer Vision** | OpenCV | Frame Manipulation & Stream Engine |
| **Persistence** | SQLAlchemy + SQLite | Incident Logging & User Auth |
| **Real-time** | Socket.IO | Push Alerts to Browser |
| **Frontend** | Chart.js & Vanilla CSS | Dashboard & Professional UI |

---

## 📂 Project Organization

```text
retail-monitor/
├── app/
│   ├── app.py              # Main Server & Detection Pipeline
│   ├── retail_monitor.db   # SQLite DB (Generated)
│   ├── templates/          # HTML Layouts
│   └── static/             # UI Assets, Uploads & Snapshots
├── model/
│   └── best.pt             # Custom AI Weights
├── yolov8n.pt              # Base Tracking Weights
├── setup.md                # Comprehensive Setup Guide
└── requirements.txt        # System Dependencies
```

---

> [!TIP]
> This system is designed for NVIDIA GPUs but runs exceptionally well on modern CPUs thanks to the `yolov8n` (nano) lightweight architecture.

*© 2024 RetailGuard AI — Secure. Proactive. Intelligent.*

---

## 📝 Notes for Production

1. Change default passwords in `app.py` `init_db()` function
2. Set a strong `app.secret_key`
3. Use `gunicorn` + `nginx` instead of Flask dev server
4. Switch SQLite to MySQL/PostgreSQL for scale
5. Configure HTTPS for secure camera streaming
