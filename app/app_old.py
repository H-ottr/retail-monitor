"""
ShopGuard AI — Flask Backend
Real-time shoplifting detection via YOLOv8 + webcam + video upload
"""

import cv2, json, csv, time, threading, os, random, subprocess, shutil, sys
from datetime import datetime
from pathlib import Path
from collections import deque
from flask import (Flask, Response, render_template, jsonify,
                   request, send_from_directory)
from werkzeug.utils import secure_filename

# ── FFmpeg binary detection ───────────────────────────────────────────────────
def ensure_ffmpeg():
    # 1. Check system PATH
    ff = shutil.which("ffmpeg")
    if ff:
        print(f"[✓] FFmpeg found in PATH: {ff}")
        return ff
    # 2. imageio-ffmpeg bundles its own binary
    try:
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[✓] FFmpeg via imageio-ffmpeg: {ff}")
        return ff
    except Exception:
        pass
    # 3. Common Windows paths
    for c in [r"C:\ffmpeg\bin\ffmpeg.exe",
              r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe",
              r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"]:
        if os.path.exists(c):
            return c
    # 4. Auto-install imageio-ffmpeg
    print("[*] Installing imageio-ffmpeg...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install",
                               "imageio-ffmpeg", "-q"])
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        print(f"[✓] FFmpeg ready: {ff}")
        return ff
    except Exception as e:
        print(f"[!] FFmpeg unavailable: {e}")
        return None

FFMPEG = ensure_ffmpeg()

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

BASE       = Path(__file__).parent
OUTPUT     = BASE / "output"
LOGS       = BASE / "logs"
UPLOADS    = BASE / "uploads"
MODEL_PATH = BASE / "model" / "best.pt"

for d in (OUTPUT, LOGS, UPLOADS):
    d.mkdir(exist_ok=True)

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# ── Global state ─────────────────────────────────────────────────────────────
model          = None
camera         = None
camera_lock    = threading.Lock()
is_streaming   = False
detection_log  = []
recent_events  = deque(maxlen=100)
processing_job = {"active": False, "progress": 0, "output": None, "error": None}
session_stats  = {"detections": 0, "frames": 0, "start": None}


# ── FFmpeg H.264 conversion ───────────────────────────────────────────────────
def convert_to_web(input_path: Path) -> Path:
    """Re-encode raw mp4v video to H.264 so all browsers can play it."""
    if not FFMPEG:
        return input_path
    out_path = input_path.with_name(
        input_path.name.replace("_raw.mp4", "_web.mp4")
        if "_raw.mp4" in input_path.name
        else input_path.stem + "_web.mp4"
    )
    cmd = [FFMPEG, "-y", "-i", str(input_path),
           "-vcodec", "libx264", "-acodec", "aac",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(out_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if out_path.exists() and out_path.stat().st_size > 0:
            try: input_path.unlink()
            except: pass
            print(f"[✓] Converted to H.264: {out_path.name}")
            return out_path
        else:
            print(f"[!] FFmpeg failed: {result.stderr.decode()[-300:]}")
    except Exception as e:
        print(f"[!] Conversion error: {e}")
    return input_path   # fallback: return original


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    global model
    try:
        from ultralytics import YOLO
        if MODEL_PATH.exists():
            model = YOLO(str(MODEL_PATH))
            print(f"[✓] YOLOv8 model loaded from {MODEL_PATH}")
        else:
            print(f"[!] best.pt not found → DEMO mode")
    except ImportError:
        print("[!] ultralytics not installed → DEMO mode")


def detect(frame):
    if model:
        results = model(frame, verbose=False)
        out = []
        for r in results:
            for box in r.boxes:
                conf = float(box.conf[0])
                if conf < 0.35: continue
                cls = int(box.cls[0])
                lbl = model.names[cls]
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                out.append({"bbox":[x1,y1,x2,y2],"label":lbl,
                            "confidence":round(conf,2)})
        return out
    if random.random() < 0.07:
        h, w = frame.shape[:2]
        x1 = random.randint(60, w//2)
        y1 = random.randint(40, h//2)
        x2 = min(x1+random.randint(90,220), w-1)
        y2 = min(y1+random.randint(120,280), h-1)
        lbl = random.choice(["Shoplifting","Concealing Item","Suspicious Activity"])
        return [{"bbox":[x1,y1,x2,y2],"label":lbl,
                 "confidence":round(random.uniform(.55,.93),2)}]
    return []


def draw_boxes(frame, dets):
    for d in dets:
        x1,y1,x2,y2 = d["bbox"]
        lbl, conf = d["label"], d["confidence"]
        ov = frame.copy()
        cv2.rectangle(ov,(x1,y1),(x2,y2),(0,0,200),-1)
        frame = cv2.addWeighted(ov,.13,frame,.87,0)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,40,255),2)
        txt = f"{lbl}  {conf:.0%}"
        (tw,th),_ = cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,.52,1)
        cv2.rectangle(frame,(x1,y1-th-10),(x1+tw+8,y1),(0,40,255),-1)
        cv2.putText(frame,txt,(x1+4,y1-5),cv2.FONT_HERSHEY_SIMPLEX,
                    .52,(255,255,255),1,cv2.LINE_AA)
        for cx,cy,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(cx,cy),(cx+sx*14,cy),(0,200,255),2)
            cv2.line(frame,(cx,cy),(cx,cy+sy*14),(0,200,255),2)
    return frame


def add_hud(frame, dets):
    h, w = frame.shape[:2]
    cv2.rectangle(frame,(0,0),(w,30),(10,10,18),-1)
    ts = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame,f"ShopGuard AI  |  {ts}",(10,20),
                cv2.FONT_HERSHEY_SIMPLEX,.48,(0,200,255),1,cv2.LINE_AA)
    mode = "DEMO" if not model else "LIVE"
    col  = (0,140,255) if not model else (0,255,80)
    cv2.putText(frame,mode,(w-70,20),cv2.FONT_HERSHEY_SIMPLEX,
                .48,col,1,cv2.LINE_AA)
    cv2.rectangle(frame,(0,h-28),(w,h),(10,10,18),-1)
    if dets:
        cv2.putText(frame,f"ALERT  {len(dets)} DETECTION(S)",(10,h-8),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(80,80,255),1,cv2.LINE_AA)
    else:
        cv2.putText(frame,"MONITORING",(10,h-8),
                    cv2.FONT_HERSHEY_SIMPLEX,.5,(60,180,60),1,cv2.LINE_AA)
    return frame


def log_detection(dets):
    ts = datetime.now().isoformat()
    for d in dets:
        entry = {"timestamp":ts,"label":d["label"],
                 "confidence":d["confidence"],"bbox":d["bbox"]}
        detection_log.append(entry)
        recent_events.appendleft(entry)
        session_stats["detections"] += 1
        csv_path = LOGS/"detections.csv"
        hdr = not csv_path.exists()
        with open(csv_path,"a",newline="") as f:
            w = csv.DictWriter(f,fieldnames=["timestamp","label","confidence","bbox"])
            if hdr: w.writeheader()
            w.writerow(entry)
        with open(LOGS/"detections.json","w") as f:
            json.dump(detection_log,f,indent=2)


# ── Live stream ───────────────────────────────────────────────────────────────
def gen_frames():
    global camera, is_streaming, session_stats
    with camera_lock:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
    if not camera.isOpened():
        return
    session_stats = {"detections":0,"frames":0,"start":datetime.now().isoformat()}
    raw_path = OUTPUT / f"live_{int(time.time())}_raw.mp4"
    vw = cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*"mp4v"),20,(1280,720))
    try:
        while is_streaming:
            ok, frame = camera.read()
            if not ok: break
            session_stats["frames"] += 1
            dets = detect(frame)
            if dets: log_detection(dets)
            frame = draw_boxes(frame,dets)
            frame = add_hud(frame,dets)
            vw.write(frame)
            _, buf = cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,82])
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"+buf.tobytes()+b"\r\n"
    finally:
        vw.release()
        camera.release()
        camera = None
        threading.Thread(target=convert_to_web,args=(raw_path,),daemon=True).start()


# ── Uploaded video processing ─────────────────────────────────────────────────
def process_video_file(input_path: Path, stem: str):
    global processing_job
    cap   = cv2.VideoCapture(str(input_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_path = OUTPUT / f"{stem}_raw.mp4"
    vw = cv2.VideoWriter(str(raw_path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(W,H))
    frame_n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok: break
            frame_n += 1
            dets = detect(frame)
            if dets: log_detection(dets)
            frame = draw_boxes(frame,dets)
            frame = add_hud(frame,dets)
            vw.write(frame)
            processing_job["progress"] = round(frame_n/total*93)
    except Exception as e:
        processing_job["error"] = str(e)
    finally:
        cap.release()
        vw.release()

    processing_job["progress"] = 95
    final = convert_to_web(raw_path)
    processing_job["active"]   = False
    processing_job["progress"] = 100
    processing_job["output"]   = final.name


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    if not is_streaming:
        return Response(status=204)
    return Response(gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
def start():
    global is_streaming
    is_streaming = True
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    global is_streaming
    is_streaming = False
    return jsonify({"ok": True})

@app.route("/stats")
def stats():
    return jsonify({**session_stats,"streaming":is_streaming,
                    "model_loaded":model is not None,
                    "log_count":len(detection_log)})

@app.route("/events")
def events():
    return jsonify(list(recent_events)[:30])

@app.route("/clear_log", methods=["POST"])
def clear_log():
    global detection_log
    detection_log.clear(); recent_events.clear()
    session_stats["detections"] = 0
    return jsonify({"ok": True})

@app.route("/upload", methods=["POST"])
def upload():
    global processing_job
    if processing_job["active"]:
        return jsonify({"error":"Already processing a video."}), 409
    f = request.files.get("video")
    if not f:
        return jsonify({"error":"No file received."}), 400
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error":f"Unsupported format: {ext}"}), 400
    fname = secure_filename(f.filename)
    inp   = UPLOADS / fname
    f.save(inp)
    stem = f"processed_{int(time.time())}"
    processing_job = {"active":True,"progress":0,"output":None,"error":None}
    threading.Thread(target=process_video_file,args=(inp,stem),daemon=True).start()
    return jsonify({"ok": True})

@app.route("/processing_status")
def processing_status():
    return jsonify(processing_job)

@app.route("/output/<path:fname>")
def serve_output(fname):
    return send_from_directory(str(OUTPUT), fname)

@app.route("/output_list")
def output_list():
    files = sorted(OUTPUT.glob("*.mp4"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    files = [f for f in files if "_raw" not in f.name]
    return jsonify([f.name for f in files])

if __name__ == "__main__":
    load_model()
    print("\n  ShopGuard AI  →  http://127.0.0.1:5000\n")
    app.run(debug=False, threaded=True)
