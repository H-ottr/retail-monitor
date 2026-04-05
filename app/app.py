"""
=============================================================
Retail Shop Monitoring — Flask Backend
=============================================================
"""

import os
import cv2
import json
import time
import threading

import torch
original_load = torch.load
torch.load = lambda *a, **kw: original_load(*a, **dict(kw, weights_only=False))

import numpy as np
from datetime import datetime, timedelta
from flask import (Flask, render_template, request, redirect,
                   url_for, session, jsonify, Response, send_from_directory)
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# ─────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────
# Base directory paths
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

app = Flask(__name__)
app.secret_key = "retail_monitor_secret_2024"
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "retail_monitor.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"]    = os.path.join(BASE_DIR, "static", "uploads")
app.config["PROCESSED_FOLDER"] = os.path.join(BASE_DIR, "static", "processed")
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ─────────────────────────────────────────
# YOLOv8 Model Loading
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(PROJECT_ROOT, "yolov8n.pt")
SUSPICIOUS_MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "best.pt")
model = None
suspicious_model = None
model_lock = threading.Lock()

BEHAVIOR_CLASSES = {
    0: "Normal",
    1: "Running",
    2: "Loitering",
    3: "Suspicious Shelf Interaction",
    4: "Crowd Formation",
}

ANOMALY_CLASSES = {1, 2, 3, 4}  # Non-normal behaviors

BEHAVIOR_COLORS = {
    "Normal":                      (34, 197, 94),    # Green
    "Running":                     (239, 68, 68),    # Red
    "Loitering":                   (249, 115, 22),   # Orange
    "Suspicious Shelf Interaction": (234, 179, 8),   # Yellow
    "Crowd Formation":             (168, 85, 247),   # Purple
}


def load_model():
    global model, suspicious_model
    try:
        from ultralytics import YOLO
        if os.path.exists(MODEL_PATH):
            with model_lock:
                model = YOLO(MODEL_PATH)
            print(f"[✓] Model loaded: {MODEL_PATH}")
        else:
            print(f"[!] Model not found at {MODEL_PATH}. Using yolov8n.pt pretrained for maximum speed.")
            with model_lock:
                model = YOLO("yolov8n.pt")
                
        if os.path.exists(SUSPICIOUS_MODEL_PATH):
            with model_lock:
                suspicious_model = YOLO(SUSPICIOUS_MODEL_PATH)
            print(f"[✓] Suspicious model loaded: {SUSPICIOUS_MODEL_PATH}")
        else:
            print(f"[!] Suspicious model not found at {SUSPICIOUS_MODEL_PATH}.")
            suspicious_model = None
            
    except Exception as e:
        print(f"[✗] Model load error: {e}")
        model = None
        suspicious_model = None


# ─────────────────────────────────────────
# Database Models
# ─────────────────────────────────────────
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="staff")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Incident(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    behavior = db.Column(db.String(50), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    camera_source = db.Column(db.String(50), default="Webcam")
    frame_path = db.Column(db.String(200))
    is_anomaly = db.Column(db.Boolean, default=True)


class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    action = db.Column(db.String(200), nullable=False)
    user = db.Column(db.String(80))
    category = db.Column(db.String(50))


# ─────────────────────────────────────────
# Camera State
# ─────────────────────────────────────────
camera_state = {
    "active": False,
    "paused": False,
    "cap": None,
    "thread": None,
    "frame": None,
    "lock": threading.Lock(),
    "alerts": [],
    "detection_count": 0,
}

output_frame = None
latest_raw_frame = None
output_lock = threading.Lock()

# Tracking history state map {track_id: {"history": [(x,y,time), ...], "start_time": time.time()}}
track_history = {}


def log_activity(action, user="System", category="System"):
    with app.app_context():
        log = ActivityLog(action=action, user=user, category=category)
        db.session.add(log)
        db.session.commit()


def save_incident(behavior, confidence, frame=None):
    with app.app_context():
        frame_path = None
        if frame is not None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            fname = f"incident_{ts}.jpg"
            fpath = os.path.join(BASE_DIR, "static", "uploads", "incidents", fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            cv2.imwrite(fpath, frame)
            frame_path = f"uploads/incidents/{fname}"

        incident = Incident(
            behavior=behavior,
            confidence=round(float(confidence), 3),
            is_anomaly=(behavior != "Normal"),
            frame_path=frame_path,
        )
        db.session.add(incident)
        db.session.commit()


def process_frame(frame, custom_track_history=None, current_time_override=None):
    """Run YOLOv8 tracking on a frame, calculate behaviors, and draw results."""
    global model, suspicious_model, track_history
    detections = []

    if model is None:
        return frame, detections
        
    # Use provided history or default to global
    history_dict = custom_track_history if custom_track_history is not None else track_history

    try:
        # We track person=0, backpack=24, handbag=26, suitcase=27
        with model_lock:
            results = model.track(frame, persist=True, conf=0.4, iou=0.45, classes=[0, 24, 26, 27], verbose=False)

        current_time = current_time_override if current_time_override is not None else time.time()
        active_ids = set()
        person_centroids = [] # For crowd detection

        for result in results:
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
                
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else -1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                if cls_id == 0 and track_id != -1: # Person logic
                    active_ids.add(track_id)
                    person_centroids.append((cx, cy))
                    
                    if track_id not in history_dict:
                        history_dict[track_id] = {
                            "history": [], 
                            "start_time": current_time,
                            "suspicious_buffer": [],
                            "suspicious_cooldown": 0
                        }
                    
                    history = history_dict[track_id]["history"]
                    history.append((cx, cy, current_time))
                    history_dict[track_id]["suspicious_buffer"].append(0)
                    
                    if len(history) > 30: # keep last 30 frames
                        history.pop(0)
                    if len(history_dict[track_id]["suspicious_buffer"]) > 30:
                        history_dict[track_id]["suspicious_buffer"].pop(0)
                        
                    # Calculate velocity (Running)
                    velocity = 0
                    if len(history) >= 10:
                        old_cx, old_cy, old_time = history[-10]
                        dist = np.sqrt((cx - old_cx)**2 + (cy - old_cy)**2)
                        time_diff = current_time - old_time
                        velocity = dist / time_diff if time_diff > 0 else 0
                        
                    # Calculate loitering
                    total_time_present = current_time - history_dict[track_id]["start_time"]
                    
                    # Compute max distance moved over last 5 seconds if available
                    recent_dist = 0
                    if len(history) >= 20:
                        first_cx, first_cy, _ = history[0]
                        recent_dist = np.sqrt((cx - first_cx)**2 + (cy - first_cy)**2)
                    
                    # Behavior rules
                    behavior = "Normal"
                    if velocity > 150: # Threshold for running (pixels per second)
                        behavior = "Running"
                    elif total_time_present > 5.0 and len(history) >= 20 and recent_dist < 50:
                        behavior = "Loitering"
                        
                    is_anomaly = (behavior != "Normal")
                    color = BEHAVIOR_COLORS.get(behavior, (34, 197, 94))
                    
                    # Draw bounding box for person
                    thickness = 3 if is_anomaly else 2
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

                    label = f"{behavior} id:{track_id} {conf:.0%}"
                    (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, y1 - lh - bl - 6), (x1 + lw + 6, y1), color, -1)
                    cv2.putText(frame, label, (x1 + 3, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

                    detections.append({
                        "id": track_id,
                        "cx": cx,
                        "cy": cy,
                        "bbox": [x1, y1, x2, y2],
                        "behavior": behavior,
                        "confidence": round(conf, 3),
                        "is_anomaly": is_anomaly,
                    })
                
                elif cls_id in [24, 26, 27]: # Bags (Suspicious items)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

        # Post-process for Crowd Formation
        crowd_flags = [False] * len(person_centroids)
        for i in range(len(person_centroids)):
            close_count = 0
            for j in range(len(person_centroids)):
                if i != j:
                    dist = np.sqrt((person_centroids[i][0] - person_centroids[j][0])**2 + (person_centroids[i][1] - person_centroids[j][1])**2)
                    if dist < 120: # Pixel distance threshold for crowd
                        close_count += 1
            if close_count >= 2: # 3+ people close = crowd
                crowd_flags[i] = True
                
        # Update behavior if part of crowd
        person_detections = [d for d in detections if 'cx' in d]
        for i, det in enumerate(person_detections):
            if crowd_flags[i]:
                det["behavior"] = "Crowd Formation"
                det["is_anomaly"] = True
                x1, y1, x2, y2 = det["bbox"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), BEHAVIOR_COLORS["Crowd Formation"], 3)
                label = f"Crowd Formation id:{det['id']} {det['confidence']:.0%}"
                (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(frame, (x1, y1 - lh - bl - 6), (x1 + lw + 6, y1), BEHAVIOR_COLORS["Crowd Formation"], -1)
                cv2.putText(frame, label, (x1 + 3, y1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Check for Suspicious Shelf Interaction using the custom model, if available
        if suspicious_model is not None and len(person_detections) > 0:
            if not hasattr(process_frame, "counter"):
                process_frame.counter = 0
            process_frame.counter += 1

            # Only run heavy suspicious model 1 out of every 3 frames (~10fps) to prevent lag
            if process_frame.counter % 3 == 0 or not hasattr(process_frame, "last_suspicious"):
                with model_lock:
                    suspicious_results = suspicious_model.predict(frame, conf=0.4, verbose=False)
                process_frame.last_suspicious = suspicious_results
            else:
                suspicious_results = process_frame.last_suspicious
            
            for result in suspicious_results:
                if result.boxes is None: continue
                for box in result.boxes:
                    conf = float(box.conf[0])
                    sx1, sy1, sx2, sy2 = map(int, box.xyxy[0])
                    
                    best_iou = 0
                    associated_det = None
                    
                    # Associate bounding box with nearest person
                    for det in person_detections:
                        px1, py1, px2, py2 = det["bbox"]
                        ix1 = max(sx1, px1)
                        iy1 = max(sy1, py1)
                        ix2 = min(sx2, px2)
                        iy2 = min(sy2, py2)
                        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        if inter_area > best_iou:
                            best_iou = inter_area
                            associated_det = det
                                
                    if associated_det:
                        tid = associated_det["id"]
                        if tid in history_dict and len(history_dict[tid]["suspicious_buffer"]) > 0:
                            history_dict[tid]["suspicious_buffer"][-1] = 1
                    else:
                        # Draw standalone suspicious behavior
                        cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), BEHAVIOR_COLORS["Suspicious Shelf Interaction"], 3)
                        label = f"Suspicious Activity {conf:.0%}"
                        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (sx1, sy1 - lh - bl - 6), (sx1 + lw + 6, sy1), BEHAVIOR_COLORS["Suspicious Shelf Interaction"], -1)
                        cv2.putText(frame, label, (sx1 + 3, sy1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                        detections.append({
                            "id": -1,
                            "bbox": [sx1, sy1, sx2, sy2],
                            "behavior": "Suspicious Shelf Interaction",
                            "confidence": round(conf, 3),
                            "is_anomaly": True,
                        })
        else:
            # Fallback to very basic check for Suspicious Shelf Interaction (Bag overlap with a Person who is Loitering)
            # Assuming bags are not explicitly tracked, we just use bounding box overlap
            for det in detections:
                if det["behavior"] == "Loitering":
                    px1, py1, px2, py2 = det["bbox"]
                    for b_box in (b.xyxy[0] for r in results for b in r.boxes if int(b.cls[0]) in [24,26,27]):
                        bx1, by1, bx2, by2 = map(int, b_box)
                        if px2 > bx1 and px1 < bx2 and py2 > by1 and py1 < by2:
                            tid = det.get("id", -1)
                            if tid in history_dict and len(history_dict[tid]["suspicious_buffer"]) > 0:
                                history_dict[tid]["suspicious_buffer"][-1] = 1
                            break

        # Process Suspicious Buffer Temporal Voting and Decay
        for det in detections:
            tid = det.get("id", -1)
            if tid in history_dict:
                buf = history_dict[tid]["suspicious_buffer"]
                if len(buf) > 0:
                    ratio = sum(buf) / len(buf)
                    
                    if ratio >= 0.5:
                        history_dict[tid]["suspicious_cooldown"] = 60
                        
                    if history_dict[tid].get("suspicious_cooldown", 0) > 0:
                        det["behavior"] = "Suspicious Shelf Interaction"
                        det["is_anomaly"] = True
                        if ratio < 0.5:
                            history_dict[tid]["suspicious_cooldown"] -= 1
                            
                        # Overwrite existing person box with Suspicious Activity styling
                        ax1, ay1, ax2, ay2 = det["bbox"]
                        cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), BEHAVIOR_COLORS["Suspicious Shelf Interaction"], 3)
                        label = f"Suspicious Activity id:{tid} {det['confidence']:.0%}"
                        (lw, lh), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                        cv2.rectangle(frame, (ax1, ay1 - lh - bl - 6), (ax1 + lw + 6, ay1), BEHAVIOR_COLORS["Suspicious Shelf Interaction"], -1)
                        cv2.putText(frame, label, (ax1 + 3, ay1 - bl - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        # Remove stale tracks
        for tid in list(history_dict.keys()):
            if tid not in active_ids:
                if len(history_dict[tid]["history"]) == 0 or (current_time - history_dict[tid]["history"][-1][2] > 2.0):
                    del history_dict[tid]

    except Exception as e:
        print(f"[Detection error] {e}")

    return frame, detections


def frame_grabber_thread():
    """Tight loop to constantly empty the camera buffer and grab the absolute freshest frame."""
    global latest_raw_frame
    while camera_state["active"]:
        with camera_state["lock"]:
            cap = camera_state["cap"]
            if cap is None or not cap.isOpened():
                break
            ret, frame = cap.read()
            if ret:
                latest_raw_frame = frame
        time.sleep(0.01) # Small sleep to prevent 100% CPU lock


def camera_thread():
    global output_frame, camera_state, latest_raw_frame
    last_save = {}
    frame_count = 0

    while camera_state["active"]:
        if camera_state["paused"] or latest_raw_frame is None:
            time.sleep(0.05)
            continue

        frame_count += 1
        # Grab the absolute freshest frame (bypassing any backlog)
        frame_to_process = latest_raw_frame.copy()
        
        annotated, detections = process_frame(frame_to_process)

        # Overlay timestamp
        ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        cv2.putText(annotated, ts, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Handle detections
        alerts = []
        for det in detections:
            if det["is_anomaly"]:
                alerts.append(det)
                beh = det["behavior"]
                now = time.time()
                # Save incident every 5 seconds per behavior type
                if now - last_save.get(beh, 0) > 5:
                    last_save[beh] = now
                    save_incident(beh, det["confidence"], frame_to_process)
                    camera_state["detection_count"] += 1

        # Save normal occasionally
        if detections and frame_count % 30 == 0:
            normals = [d for d in detections if not d["is_anomaly"]]
            if normals:
                save_incident("Normal", normals[0]["confidence"])

        # Emit alerts to browser
        if alerts:
            socketio.emit("alert", {"detections": alerts})

        # Update global frame
        with output_lock:
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            output_frame = buffer.tobytes()

        # Let CPU breathe, YOLO takes time so we don't need artificial sleep, but just in case
        time.sleep(0.01)


def gen_frames():
    global output_frame
    while True:
        with output_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            frame = output_frame
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.03)


# ─────────────────────────────────────────
# Auth helpers
# ─────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ─────────────────────────────────────────
# Routes — Auth
# ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            session["user_id"] = user.id
            session["username"] = user.username
            session["role"] = user.role
            log_activity(f"User '{username}' logged in", user=username, category="Auth")
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    username = session.get("username", "Unknown")
    log_activity(f"User '{username}' logged out", user=username, category="Auth")
    session.clear()
    return redirect(url_for("login"))


# ─────────────────────────────────────────
# Routes — Pages
# ─────────────────────────────────────────
@app.route("/dashboard")
@login_required
def dashboard():
    today = datetime.utcnow().date()
    total = Incident.query.filter(db.func.date(Incident.timestamp) == today).count()
    suspicious = Incident.query.filter(
        db.func.date(Incident.timestamp) == today,
        Incident.is_anomaly == True
    ).count()
    normal = total - suspicious
    return render_template("dashboard.html",
                           username=session.get("username"),
                           total=total, suspicious=suspicious, normal=normal,
                           camera_active=camera_state["active"])


@app.route("/live")
@login_required
def live():
    return render_template("live.html", username=session.get("username"))


@app.route("/video-analysis")
@login_required
def video_analysis():
    return render_template("video_analysis.html", username=session.get("username"))


@app.route("/alerts")
@login_required
def alerts():
    incidents = Incident.query.order_by(Incident.timestamp.desc()).limit(200).all()
    return render_template("alerts.html", username=session.get("username"), incidents=incidents)


@app.route("/analytics")
@login_required
def analytics():
    return render_template("analytics.html", username=session.get("username"))


@app.route("/logs")
@login_required
def logs():
    logs_list = ActivityLog.query.order_by(ActivityLog.timestamp.desc()).limit(200).all()
    return render_template("logs.html", username=session.get("username"), logs=logs_list)


@app.route("/settings")
@login_required
def settings():
    return render_template("settings.html", username=session.get("username"))


# ─────────────────────────────────────────
# Routes — API
# ─────────────────────────────────────────
@app.route("/api/camera/start", methods=["POST"])
@login_required
def camera_start():
    if camera_state["active"]:
        return jsonify({"status": "already_running"})
    
    # Use cv2.CAP_DSHOW for INSTANT camera connection on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # Decrease resolution and set buffer to 1 to drastically speed up processing and prevent frame buildup / lag
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Cannot open camera"}), 500
    camera_state["active"] = True
    camera_state["paused"] = False
    camera_state["cap"] = cap
    
    # Start both threads to completely eliminate lag
    t_grab = threading.Thread(target=frame_grabber_thread, daemon=True)
    t_process = threading.Thread(target=camera_thread, daemon=True)
    
    camera_state["thread"] = t_process
    t_grab.start()
    t_process.start()
    
    log_activity("Camera started", user=session.get("username"), category="Camera")
    return jsonify({"status": "started"})


@app.route("/api/camera/stop", methods=["POST"])
@login_required
def camera_stop():
    camera_state["active"] = False
    camera_state["paused"] = False
    with camera_state["lock"]:
        if camera_state["cap"]:
            camera_state["cap"].release()
            camera_state["cap"] = None
    global output_frame
    with output_lock:
        output_frame = None
    log_activity("Camera stopped", user=session.get("username"), category="Camera")
    return jsonify({"status": "stopped"})


@app.route("/api/camera/pause", methods=["POST"])
@login_required
def camera_pause():
    camera_state["paused"] = not camera_state["paused"]
    state = "paused" if camera_state["paused"] else "resumed"
    log_activity(f"Camera {state}", user=session.get("username"), category="Camera")
    return jsonify({"status": state})


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/dashboard/stats")
@login_required
def dashboard_stats():
    today = datetime.utcnow().date()
    # Hourly bar chart data (last 12 hours)
    hourly = []
    for h in range(12):
        dt = datetime.utcnow() - timedelta(hours=11 - h)
        count = Incident.query.filter(
            Incident.timestamp >= dt.replace(minute=0, second=0),
            Incident.timestamp < dt.replace(minute=59, second=59)
        ).count()
        hourly.append({"hour": dt.strftime("%H:00"), "count": count})

    # Pie chart — behavior distribution
    behaviors = {}
    for cls in BEHAVIOR_CLASSES.values():
        cnt = Incident.query.filter(
            db.func.date(Incident.timestamp) == today,
            Incident.behavior == cls
        ).count()
        if cnt > 0:
            behaviors[cls] = cnt

    return jsonify({
        "hourly": hourly,
        "behaviors": behaviors,
        "camera_active": camera_state["active"],
    })


@app.route("/api/alerts/recent")
@login_required
def recent_alerts():
    incidents = Incident.query.filter_by(is_anomaly=True)\
        .order_by(Incident.timestamp.desc()).limit(10).all()
    return jsonify([{
        "id": i.id,
        "timestamp": i.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "behavior": i.behavior,
        "confidence": i.confidence,
        "camera": i.camera_source,
    } for i in incidents])


@app.route("/api/upload-video", methods=["POST"])
@login_required
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    allowed = {"mp4", "avi", "mov", "mkv"}
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(f"upload_{int(time.time())}.{ext}")
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(upload_path)
    log_activity(f"Video uploaded: {filename}", user=session.get("username"), category="Video")

    # Process video in background thread
    out_name = f"processed_{filename}"
    out_path = os.path.join(app.config["PROCESSED_FOLDER"], out_name)
    threading.Thread(
        target=process_video_file,
        args=(upload_path, out_path, out_name),
        daemon=True
    ).start()

    return jsonify({"status": "processing", "output": out_name})


def process_video_file(input_path, output_path, out_name):
    try:
        import subprocess

        # Verify input file exists and is readable
        if not os.path.exists(input_path):
            raise Exception(f"Input file not found: {input_path}")

        print(f"[Video] Processing: {input_path}")
        print(f"[Video] Output: {output_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"OpenCV cannot open video: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] {w}x{h} @ {fps}fps, {total} frames")

        # Resize video to a max of 640 width to drastically speed up AI and encoding
        target_w = 640
        if w > target_w:
            target_h = int(h * (target_w / w))
        else:
            target_w, target_h = w, h

        # Write annotated frames to temp file
        temp_path = output_path.replace(".mp4", "_tmp.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(temp_path, fourcc, fps, (target_w, target_h))

        if not out.isOpened():
            raise Exception(f"VideoWriter failed to open: {temp_path}")

        frame_count = 0
        last_save = {}
        video_incidents = []
        video_track_history = {} # Isolated tracking state
        
        # We will process 1 frame and repeat it for performance (2x speed jump)
        skip_factor = 2 if fps >= 24 else 1
        last_annotated = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (target_w, target_h))
            
            # Calculate simulated video time for accurate behavior detection
            simulated_time = frame_count / fps

            if frame_count % skip_factor == 0:
                annotated, detections = process_frame(frame, custom_track_history=video_track_history, current_time_override=simulated_time)
                last_annotated = annotated
            else:
                # Add tiny motion blur or just reuse to avoid heavy YOLO calculation
                annotated = last_annotated if last_annotated is not None else frame

            out.write(annotated)
            frame_count += 1

            # Save incidents detected in video to database
            for det in detections:
                beh = det["behavior"]
                now = time.time()
                if now - last_save.get(beh, 0) > 3:
                    last_save[beh] = now
                    video_incidents.append(det)
                    with app.app_context():
                        inc = Incident(
                            behavior=beh,
                            confidence=round(float(det["confidence"]), 3),
                            is_anomaly=det["is_anomaly"],
                            camera_source=f"Video:{out_name}",
                        )
                        db.session.add(inc)
                        db.session.commit()

            if frame_count % 100 == 0:
                print(f"[Video] Processed {frame_count}/{total} frames, {len(video_incidents)} incidents")

        cap.release()
        out.release()
        print(f"[Video] Written {frame_count} frames, {len(video_incidents)} incidents saved")

        if frame_count == 0:
            raise Exception("No frames were processed — video may be corrupted or codec unsupported")

        # Re-encode to H264 using FFmpeg
        ffmpeg_path = r"C:fmpegfmpeg-8.0.1-essentials_buildinfmpeg.EXE"
        if not os.path.exists(ffmpeg_path):
            ffmpeg_path = "ffmpeg"  # fallback to PATH

        ffmpeg_cmd = [
            ffmpeg_path, "-y",
            "-i", temp_path,
            "-vcodec", "libx264",
            "-crf", "23",
            "-preset", "fast",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]
        print(f"[Video] Re-encoding with FFmpeg...")
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.remove(temp_path)
            print(f"[✓] Video ready: {out_name}")
        else:
            import shutil
            shutil.move(temp_path, output_path)
            print(f"[!] FFmpeg failed: {result.stderr[:300]}")

        socketio.emit("video_ready", {"filename": out_name})
        log_activity(f"Video processed: {out_name}", category="Video")

    except Exception as e:
        print(f"[Video processing error] {e}")
        socketio.emit("video_error", {"error": str(e)})


@app.route("/api/analytics/data")
@login_required
def analytics_data():
    # Last 7 days trend
    trend = []
    for d in range(6, -1, -1):
        date = (datetime.utcnow() - timedelta(days=d)).date()
        anomalies = Incident.query.filter(
            db.func.date(Incident.timestamp) == date,
            Incident.is_anomaly == True
        ).count()
        normals = Incident.query.filter(
            db.func.date(Incident.timestamp) == date,
            Incident.is_anomaly == False
        ).count()
        trend.append({
            "date": date.strftime("%b %d"),
            "anomalies": anomalies,
            "normal": normals
        })

    # Behavior breakdown all time
    breakdown = {}
    for cls in BEHAVIOR_CLASSES.values():
        cnt = Incident.query.filter_by(behavior=cls).count()
        breakdown[cls] = cnt

    return jsonify({"trend": trend, "breakdown": breakdown})


@app.route("/api/settings/save", methods=["POST"])
@login_required
def save_settings():
    data = request.json
    log_activity("Settings updated", user=session.get("username"), category="Settings")
    return jsonify({"status": "saved"})


# ─────────────────────────────────────────
# Init DB & Default User
# ─────────────────────────────────────────
def init_db():
    with app.app_context():
        db.create_all()
        if not User.query.filter_by(username="admin").first():
            admin = User(username="admin", role="admin")
            admin.set_password("admin123")
            db.session.add(admin)
        if not User.query.filter_by(username="staff").first():
            staff = User(username="staff", role="staff")
            staff.set_password("staff123")
            db.session.add(staff)
        db.session.commit()
        print("[✓] Database initialized")
        print("[✓] Default users: admin/admin123 | staff/staff123")


# ─────────────────────────────────────────
# NEW FEATURES
# ─────────────────────────────────────────

# 1. List all processed videos
@app.route("/api/processed-videos")
@login_required
def list_processed_videos():
    processed_dir = app.config["PROCESSED_FOLDER"]
    videos = []
    if os.path.exists(processed_dir):
        for f in sorted(os.listdir(processed_dir), reverse=True):
            if f.endswith(".mp4") and not f.endswith("_tmp.mp4"):
                fpath = os.path.join(processed_dir, f)
                size  = os.path.getsize(fpath)
                mtime = os.path.getmtime(fpath)
                videos.append({
                    "filename": f,
                    "url": f"/static/processed/{f}",
                    "size_mb": round(size / 1024 / 1024, 1),
                    "date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M"),
                })
    return jsonify({"videos": videos})


# 2. Delete a processed video
@app.route("/api/processed-videos/<filename>", methods=["DELETE"])
@login_required
def delete_processed_video(filename):
    safe = secure_filename(filename)
    fpath = os.path.join(app.config["PROCESSED_FOLDER"], safe)
    if os.path.exists(fpath):
        os.remove(fpath)
        log_activity(f"Deleted video: {safe}", user=session.get("username"), category="Video")
        return jsonify({"status": "deleted"})
    return jsonify({"error": "File not found"}), 404


# 3. Reset / clear all data
@app.route("/api/reset-data", methods=["POST"])
@login_required
def reset_data():
    data = request.json or {}
    what = data.get("what", "all")  # all | incidents | logs | videos

    if what in ("all", "incidents"):
        Incident.query.delete()
        db.session.commit()

    if what in ("all", "logs"):
        ActivityLog.query.delete()
        db.session.commit()

    if what in ("all", "videos"):
        proc_dir = app.config["PROCESSED_FOLDER"]
        upl_dir  = app.config["UPLOAD_FOLDER"]
        for d in [proc_dir, upl_dir]:
            if os.path.exists(d):
                for f in os.listdir(d):
                    fp = os.path.join(d, f)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                        except Exception:
                            pass

    log_activity(f"Data reset: {what}", user=session.get("username"), category="System")
    return jsonify({"status": "reset", "what": what})


# 4. Get video incidents (for a specific processed video)
@app.route("/api/video-incidents/<filename>")
@login_required
def video_incidents(filename):
    safe = secure_filename(filename)
    incidents = Incident.query.filter(
        Incident.camera_source == f"Video:{safe}"
    ).order_by(Incident.timestamp.desc()).all()
    return jsonify([{
        "id": i.id,
        "timestamp": i.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "behavior": i.behavior,
        "confidence": i.confidence,
        "is_anomaly": i.is_anomaly,
    } for i in incidents])


# ─────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "static", "uploads", "incidents"), exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "static", "processed"), exist_ok=True)
    init_db()
    load_model()
    print("\n[✓] Starting Retail Monitor on http://localhost:5000\n")
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)


