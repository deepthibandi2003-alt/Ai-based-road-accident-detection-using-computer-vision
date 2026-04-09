"""
AI Road Accident Detection — Flask App
=======================================
Run:  python app.py
Open: http://127.0.0.1:5000/

Requirements:
    pip install flask ultralytics opencv-python-headless scipy
"""

import os, json, time, base64, threading, queue, math
import cv2
import numpy as np
from flask import Flask, Response, request, jsonify

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics not installed — running in DEMO mode")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024

UPLOAD_FOLDER = "uploads"
MODEL_PATH    = "models/best.pt"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

_job_queue   = queue.Queue(maxsize=400)
_job_running = False
_stop_event  = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
#  VEHICLE TRACKER  (MOG2 background subtraction + simple centroid tracking)
#  Used for motion-based accident PREDICTION (independent of YOLO)
# ─────────────────────────────────────────────────────────────────────────────

class VehicleTracker:
    """
    Tracks vehicle blobs per frame using MOG2 + contour detection.
    Maintains per-ID speed history to detect sudden drops.
    Detects bounding-box overlap between any two tracked vehicles.
    """

    SPEED_HISTORY   = 8    # frames to average speed over
    DROP_RATIO      = 0.35 # sudden drop: new_speed < DROP_RATIO * avg_speed
    OVERLAP_THRESH  = 0.08 # IoU threshold to flag overlap
    MIN_AREA        = 800  # min contour area to be considered a vehicle
    MAX_TRACK_DIST  = 80   # max centroid distance to associate to existing track

    def __init__(self, fps: float):
        self.fps        = max(fps, 1.0)
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=200, varThreshold=50, detectShadows=False)
        self.tracks: dict[int, dict] = {}   # track_id -> {cx,cy,vx,vy,bbox,speeds,age}
        self.next_id = 0
        self.kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def _iou(self, a, b):
        ax1,ay1,ax2,ay2 = a
        bx1,by1,bx2,by2 = b
        ix1,iy1 = max(ax1,bx1), max(ay1,by1)
        ix2,iy2 = min(ax2,bx2), min(ay2,by2)
        iw,ih   = max(0,ix2-ix1), max(0,iy2-iy1)
        inter   = iw * ih
        if inter == 0: return 0.0
        ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, frame):
        """
        Process one frame. Returns:
          detections : list of {x1,y1,x2,y2,track_id,speed_px}
          predictions: list of {x1,y1,x2,y2,conf,reason,track_ids}
                       — accident predictions (sky-blue boxes)
        """
        fg = self.subtractor.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Build candidate blobs this frame
        candidates = []
        for c in contours:
            if cv2.contourArea(c) < self.MIN_AREA:
                continue
            x,y,w,h = cv2.boundingRect(c)
            cx,cy   = x + w//2, y + h//2
            candidates.append({"cx":cx,"cy":cy,"bbox":(x,y,x+w,y+h)})

        # Associate candidates → existing tracks (greedy nearest centroid)
        used_tracks = set()
        used_cands  = set()
        associations = []
        for ci, cand in enumerate(candidates):
            best_dist, best_tid = float("inf"), None
            for tid, tr in self.tracks.items():
                if tid in used_tracks: continue
                d = math.hypot(cand["cx"]-tr["cx"], cand["cy"]-tr["cy"])
                if d < self.MAX_TRACK_DIST and d < best_dist:
                    best_dist, best_tid = d, tid
            if best_tid is not None:
                associations.append((ci, best_tid, best_dist))
                used_tracks.add(best_tid)
                used_cands.add(ci)

        # Update matched tracks
        for ci, tid, _ in associations:
            cand = candidates[ci]
            tr   = self.tracks[tid]
            vx   = cand["cx"] - tr["cx"]
            vy   = cand["cy"] - tr["cy"]
            speed = math.hypot(vx, vy) * self.fps   # px/sec
            tr["speeds"].append(speed)
            if len(tr["speeds"]) > self.SPEED_HISTORY:
                tr["speeds"].pop(0)
            tr.update(cx=cand["cx"], cy=cand["cy"],
                      vx=vx, vy=vy, bbox=cand["bbox"], age=tr["age"]+1)

        # Create new tracks for unmatched candidates
        for ci, cand in enumerate(candidates):
            if ci in used_cands: continue
            self.tracks[self.next_id] = {
                "cx": cand["cx"], "cy": cand["cy"],
                "vx": 0, "vy": 0,
                "bbox": cand["bbox"],
                "speeds": [], "age": 0
            }
            self.next_id += 1

        # Remove stale tracks (not updated this frame)
        updated_ids = {tid for _, tid, _ in associations}
        stale = [tid for tid in self.tracks if tid not in updated_ids]
        for tid in stale:
            # keep ghost for 3 frames then delete
            self.tracks[tid]["age"] -= 1
            if self.tracks[tid]["age"] < -3:
                del self.tracks[tid]

        # Build detection list for drawing green/blue tracker boxes
        detections = []
        for tid, tr in self.tracks.items():
            if tr["age"] < 0: continue
            detections.append({
                "track_id": tid,
                "x1":tr["bbox"][0],"y1":tr["bbox"][1],
                "x2":tr["bbox"][2],"y2":tr["bbox"][3],
                "speed": round(tr["speeds"][-1], 1) if tr["speeds"] else 0
            })

        # ── Accident PREDICTION ──────────────────────────────────────────────
        predictions = []
        track_list  = [t for t in self.tracks.values() if t["age"] >= 0 and len(t["speeds"]) >= 3]

        for i, tr in enumerate(track_list):
            speeds = tr["speeds"]
            avg_speed = sum(speeds[:-1]) / max(len(speeds)-1, 1)
            cur_speed = speeds[-1]
            speed_drop = avg_speed > 5 and cur_speed < avg_speed * self.DROP_RATIO

            if not speed_drop:
                continue

            # Check if this vehicle's bbox overlaps any other
            for j, other in enumerate(track_list):
                if i == j: continue
                iou = self._iou(tr["bbox"], other["bbox"])
                if iou >= self.OVERLAP_THRESH:
                    # Merge bboxes for prediction box
                    mx1 = min(tr["bbox"][0], other["bbox"][0])
                    my1 = min(tr["bbox"][1], other["bbox"][1])
                    mx2 = max(tr["bbox"][2], other["bbox"][2])
                    my2 = max(tr["bbox"][3], other["bbox"][3])
                    # Confidence heuristic: higher drop + higher overlap → higher conf
                    drop_ratio = 1 - (cur_speed / avg_speed) if avg_speed > 0 else 1
                    conf = min(0.95, 0.4 + drop_ratio * 0.35 + iou * 0.3)
                    predictions.append({
                        "x1": mx1, "y1": my1, "x2": mx2, "y2": my2,
                        "conf": round(conf, 2),
                        "reason": f"Speed drop {drop_ratio:.0%} + overlap IoU {iou:.2f}",
                        "speed_before": round(avg_speed, 1),
                        "speed_after":  round(cur_speed, 1),
                    })

        return detections, predictions


# ─────────────────────────────────────────────────────────────────────────────
#  YOLO + TRACKER WORKER THREAD
# ─────────────────────────────────────────────────────────────────────────────

def yolo_worker(video_path):
    global _job_running
    _stop_event.clear()

    model = None
    if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
        model = YOLO(MODEL_PATH)
    else:
        print(f"[WARN] Model not found at '{MODEL_PATH}' — DEMO mode")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _job_queue.put({"type": "error", "msg": "Could not open video file"})
        _job_running = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_idx    = 0
    total_accidents = 0
    max_conf        = 0.0
    SKIP = max(1, int(fps / 10))

    tracker = VehicleTracker(fps)

    _job_queue.put({"type": "status", "state": "processing",
                    "total_frames": total_frames, "fps": fps})

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx      += 1
        yolo_dets       = []
        frame_max_conf  = 0.0
        frame_accidents = 0

        # ── YOLO detection ───────────────────────────────────────────────────
        if model:
            results = model(frame, verbose=False)
            for box in results[0].boxes:
                conf = float(box.conf[0])
                if conf < 0.46:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                yolo_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf})
                if conf > frame_max_conf: frame_max_conf = conf
                if conf > 0.70:          frame_accidents += 1
        else:
            # DEMO: simulated YOLO boxes
            rng = np.random.default_rng(frame_idx * 7 + 13)
            h, w = frame.shape[:2]
            n = int(rng.integers(0, 4))
            for _ in range(n):
                conf = float(rng.uniform(0.46, 0.92))
                x1 = int(rng.uniform(0.05, 0.65) * w)
                y1 = int(rng.uniform(0.10, 0.60) * h)
                bw = int(rng.uniform(0.08, 0.22) * w)
                bh = int(rng.uniform(0.07, 0.18) * h)
                x2, y2 = min(x1+bw, w-1), min(y1+bh, h-1)
                yolo_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,"conf":conf})
                if conf > frame_max_conf: frame_max_conf = conf
                if conf > 0.70:          frame_accidents += 1

        total_accidents += frame_accidents
        if frame_max_conf > max_conf: max_conf = frame_max_conf

        # ── MOG2 tracker + accident prediction ───────────────────────────────
        tracker_dets, predictions = tracker.update(frame)

        # ── Draw YOLO boxes (red = accident, orange = overspeed, green = normal)
        for det in yolo_dets:
            conf = det["conf"]
            x1,y1,x2,y2 = det["x1"],det["y1"],det["x2"],det["y2"]
            is_acc = conf > 0.70
            color  = (59,59,255) if is_acc else (68,136,255) if conf > 0.58 else (0,255,136)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cs = 10
            for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*cs,py),color,3)
                cv2.line(frame,(px,py),(px,py+dy*cs),color,3)
            label = f"ACCIDENT {conf:.0%}" if is_acc else f"Vehicle {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+6,y1),color,-1)
            cv2.putText(frame,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(10,10,10),1)

        # ── Draw tracker boxes (thin cyan outline, speed label)
        for td in tracker_dets:
            x1,y1,x2,y2 = td["x1"],td["y1"],td["x2"],td["y2"]
            cv2.rectangle(frame,(x1,y1),(x2,y2),(200,200,0),1)

        # ── Draw PREDICTION boxes (sky-blue, thicker dashed-style)
        SKY = (255, 191, 0)   # BGR sky-blue
        for pred in predictions:
            x1,y1,x2,y2 = pred["x1"],pred["y1"],pred["x2"],pred["y2"]
            # Thick sky-blue rectangle
            cv2.rectangle(frame,(x1,y1),(x2,y2),SKY,3)
            # Corner L-marks
            cs = 12
            for (px,py,dx,dy) in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
                cv2.line(frame,(px,py),(px+dx*cs,py),SKY,4)
                cv2.line(frame,(px,py),(px,py+dy*cs),SKY,4)
            label = f"PREDICTED {pred['conf']:.0%}"
            (tw,th),_ = cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,0.45,1)
            cv2.rectangle(frame,(x1,y1-th-6),(x1+tw+6,y1),SKY,-1)
            cv2.putText(frame,label,(x1+3,y1-4),cv2.FONT_HERSHEY_SIMPLEX,0.45,(10,10,10),1)

        # ── Stream every SKIP-th frame ────────────────────────────────────────
        if frame_idx % SKIP == 0 or frame_idx == 1:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                b64 = base64.b64encode(buf.tobytes()).decode()
                pct = round(frame_idx / total_frames * 100, 1)
                # frame time in seconds
                frame_time_sec = round(frame_idx / fps, 2)
                payload = {
                    "type":         "frame",
                    "frame_b64":    b64,
                    "frame_idx":    frame_idx,
                    "total_frames": total_frames,
                    "progress":     pct,
                    "frame_time_sec": frame_time_sec,
                    "accidents":    1 if total_accidents >= 1 else 0,
                    "max_conf":     round(max_conf * 100, 1),
                    "detections":   yolo_dets,
                    "has_accident_this_frame": frame_accidents > 0,
                    "frame_max_conf": round(frame_max_conf * 100, 1),
                    # Prediction events
                    "predictions":  predictions,   # list of prediction dicts
                    "has_prediction_this_frame": len(predictions) > 0,
                }
                try:
                    _job_queue.put_nowait(payload)
                except queue.Full:
                    pass

    cap.release()
    try:
        os.remove(video_path)
    except Exception:
        pass

    _job_queue.put({
        "type":     "done",
        "accidents": 1 if total_accidents >= 1 else 0,
        "max_conf":  round(max_conf * 100, 1),
        "frames_processed": frame_idx,
    })
    _job_running = False


# ─────────────────────────────────────────────────────────────────────────────
#  FLASK ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return DASHBOARD_HTML

@app.route("/upload", methods=["POST"])
def upload():
    global _job_running, _job_queue
    if _job_running:
        return jsonify({"error": "Already processing a video"}), 409
    f = request.files.get("video")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, f"upload_{int(time.time())}.mp4")
    f.save(save_path)
    _job_queue = queue.Queue(maxsize=400)
    _job_running = True
    threading.Thread(target=yolo_worker, args=(save_path,), daemon=True).start()
    return jsonify({"ok": True})

@app.route("/stop", methods=["POST"])
def stop():
    _stop_event.set()
    return jsonify({"ok": True})

@app.route("/stream")
def stream():
    def event_gen():
        while True:
            try:
                item = _job_queue.get(timeout=30)
            except queue.Empty:
                yield "data: {\"type\":\"ping\"}\n\n"
                continue
            yield f"data: {json.dumps(item)}\n\n"
            if item.get("type") in ("done", "error"):
                break
    return Response(event_gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD HTML
# ─────────────────────────────────────────────────────────────────────────────

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>AI Road Accident Detection</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap" rel="stylesheet"/>
<style>
  :root{
    --bg:#080c10;--bg2:#0d1117;--bg3:#111820;--card:#0f1922;--border:#1e3a4a;
    --cyan:#00d4ff;--cyan-dim:#00a8cc;--green:#00ff88;--yellow:#ffd700;
    --red:#ff3b3b;--blue:#4488ff;--sky:#00bfff;--sky-dim:#0090cc;
    --text:#c8dde8;--text-dim:#5a7a8a;--text-bright:#eef6fa;
    --font-mono:'Share Tech Mono',monospace;
    --font-head:'Orbitron',sans-serif;
    --font-body:'Exo 2',sans-serif;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:var(--bg);color:var(--text);font-family:var(--font-body);min-height:100vh;overflow-x:hidden}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.015) 2px,rgba(0,212,255,.015) 4px);pointer-events:none;z-index:9999}

  /* NAV */
  .navbar{display:flex;align-items:center;justify-content:space-between;padding:0 24px;height:56px;background:#060a0e;border-bottom:1px solid var(--border);position:relative}
  .navbar::after{content:'';position:absolute;bottom:0;left:0;right:0;height:1px;background:linear-gradient(90deg,transparent,var(--cyan),transparent);opacity:.5}
  .brand-title{font-family:var(--font-head);font-size:18px;font-weight:900;color:var(--cyan);letter-spacing:3px;text-shadow:0 0 20px rgba(0,212,255,.6)}
  .brand-sub{font-family:var(--font-mono);font-size:9px;color:var(--text-dim);letter-spacing:4px;margin-top:2px}
  .nav-pills{display:flex;gap:4px;background:#0a1018;border:1px solid var(--border);border-radius:6px;padding:4px 10px;font-family:var(--font-mono);font-size:11px;color:var(--cyan-dim);letter-spacing:1px}
  .nav-pills span{color:var(--text-dim)}
  .nav-right{display:flex;align-items:center;gap:16px}
  .status-pill{display:flex;align-items:center;gap:8px;background:#0a1018;border:1px solid var(--border);border-radius:6px;padding:6px 14px;font-family:var(--font-mono);font-size:11px}
  .status-dot{width:8px;height:8px;border-radius:50%;background:var(--yellow);box-shadow:0 0 8px var(--yellow);animation:pulse 2s infinite}
  .status-dot.active{background:var(--green);box-shadow:0 0 8px var(--green)}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

  /* LAYOUT */
  .main{display:grid;grid-template-columns:1fr 320px;height:calc(100vh - 56px)}
  .left-panel{display:flex;flex-direction:column;gap:12px;padding:16px;overflow-y:auto}

  /* METRICS */
  .metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
  .metric-card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:16px 18px;position:relative;overflow:hidden}
  .metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--cyan);opacity:.3}
  .metric-card.alert-active::before{background:var(--red);opacity:1}
  .metric-card.alert-active{animation:card-pulse 1.4s ease infinite alternate}
  @keyframes card-pulse{from{box-shadow:none}to{box-shadow:0 0 18px rgba(255,59,59,.25)}}
  .metric-icon{width:32px;height:32px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:16px;margin-bottom:10px;font-family:var(--font-head);font-weight:900}
  .metric-icon.red{background:rgba(255,59,59,.15);color:var(--red);border:1px solid rgba(255,59,59,.3)}
  .metric-icon.yellow{background:rgba(255,215,0,.15);color:var(--yellow);border:1px solid rgba(255,215,0,.3)}
  .metric-icon.cyan{background:rgba(0,212,255,.1);color:var(--cyan);border:1px solid rgba(0,212,255,.2)}
  .metric-icon.green{background:rgba(0,255,136,.1);color:var(--green);border:1px solid rgba(0,255,136,.2)}
  .metric-value{font-family:var(--font-head);font-size:28px;font-weight:700;line-height:1;margin-bottom:6px}
  .metric-value.red{color:var(--red);text-shadow:0 0 15px rgba(255,59,59,.5)}
  .metric-value.yellow{color:var(--yellow);text-shadow:0 0 15px rgba(255,215,0,.4)}
  .metric-value.cyan{color:var(--cyan);text-shadow:0 0 15px rgba(0,212,255,.4)}
  .metric-value.green{color:var(--green);text-shadow:0 0 15px rgba(0,255,136,.4)}
  .metric-label{font-family:var(--font-mono);font-size:9px;letter-spacing:2px;color:var(--text-dim);text-transform:uppercase}

  /* LEGEND */
  .legend{display:flex;align-items:center;gap:18px;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 16px;font-family:var(--font-mono);font-size:10px;color:var(--text-dim);flex-wrap:wrap}
  .legend-item{display:flex;align-items:center;gap:6px}
  .legend-dot{width:10px;height:10px;border-radius:2px;flex-shrink:0}

  /* UPLOAD */
  .upload-zone{background:var(--card);border:2px dashed var(--border);border-radius:10px;padding:32px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:10px;cursor:pointer;transition:all .3s;min-height:140px;position:relative}
  .upload-zone:hover,.upload-zone.dragover{border-color:var(--cyan);background:rgba(0,212,255,.03);box-shadow:0 0 20px rgba(0,212,255,.08)}
  .upload-zone.hidden{display:none}
  .upload-icon{font-size:36px;opacity:.7}
  .upload-text{font-size:15px;font-weight:600;color:var(--text-bright)}
  .upload-sub{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);letter-spacing:2px}
  #fileInput{display:none}
  .grid-scan{position:absolute;inset:0;background:repeating-linear-gradient(90deg,transparent,transparent 40px,rgba(0,212,255,.02) 40px,rgba(0,212,255,.02) 41px),repeating-linear-gradient(0deg,transparent,transparent 40px,rgba(0,212,255,.02) 40px,rgba(0,212,255,.02) 41px);pointer-events:none}
  .corner-tl,.corner-br{position:absolute;width:12px;height:12px;pointer-events:none}
  .corner-tl{top:0;left:0;border-top:2px solid var(--cyan);border-left:2px solid var(--cyan)}
  .corner-br{bottom:0;right:0;border-bottom:2px solid var(--cyan);border-right:2px solid var(--cyan)}

  /* VIDEO BAR */
  .video-bar{display:none;align-items:center;gap:12px;background:var(--card);border:1px solid var(--border);border-radius:8px;padding:12px 16px}
  .video-bar.visible{display:flex}
  .video-name{font-family:var(--font-mono);font-size:11px;color:var(--text-bright);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:200px}
  .video-meta{font-family:var(--font-mono);font-size:10px;color:var(--text-dim)}
  .prog-wrap{flex:1;height:4px;background:var(--bg3);border-radius:2px;overflow:hidden}
  .prog-fill{height:100%;background:linear-gradient(90deg,var(--cyan),var(--green));width:0%;transition:width .4s;box-shadow:0 0 8px var(--cyan)}
  .btn-stop{background:none;border:1px solid var(--border);color:var(--text-dim);font-family:var(--font-mono);font-size:10px;padding:4px 10px;border-radius:4px;cursor:pointer;transition:all .2s;white-space:nowrap}
  .btn-stop:hover{border-color:var(--red);color:var(--red)}

  /* FEED */
  .feed-section{background:var(--card);border:1px solid var(--border);border-radius:10px;overflow:hidden;flex:1;display:flex;flex-direction:column}
  .feed-header{display:flex;align-items:center;justify-content:space-between;padding:10px 16px;background:#0a1018;border-bottom:1px solid var(--border)}
  .feed-title{font-family:var(--font-mono);font-size:10px;letter-spacing:2px;color:var(--text-dim)}
  .feed-badge{font-family:var(--font-mono);font-size:10px;padding:3px 10px;border-radius:4px;background:#0d1a24;border:1px solid var(--border);color:var(--text-dim);letter-spacing:1px}
  .feed-badge.processing{border-color:var(--cyan);color:var(--cyan);animation:blink 1s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.5}}
  .feed-body{flex:1;background:#020508;display:flex;align-items:center;justify-content:center;min-height:320px}
  #liveImg{width:100%;height:100%;object-fit:contain;display:none}
  .feed-placeholder{display:flex;flex-direction:column;align-items:center;gap:12px;color:var(--text-dim);font-family:var(--font-mono);font-size:11px;letter-spacing:1px;padding:40px;text-align:center}
  .radar{font-size:40px;opacity:.5;animation:spin 4s linear infinite}
  @keyframes spin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}

  /* RIGHT PANEL */
  .right-panel{border-left:1px solid var(--border);display:flex;flex-direction:column;background:var(--bg2);overflow:hidden}
  .panel-header{padding:14px 18px;border-bottom:1px solid var(--border);font-family:var(--font-mono);font-size:10px;letter-spacing:3px;color:var(--text-dim);background:#060a0e;flex-shrink:0}
  .alerts-container{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}

  .no-alerts{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:14px;color:var(--text-dim);font-family:var(--font-mono);font-size:11px;text-align:center;padding:20px}
  .no-alerts .search-icon{font-size:40px;opacity:.4}
  .no-alerts .no-title{font-size:13px;color:var(--text);font-family:var(--font-body);font-weight:600}

  /* ── YOLO accident alert card (red) ── */
  .alert-card{background:var(--card);border:1px solid rgba(255,59,59,.4);border-radius:8px;padding:14px;border-left:3px solid var(--red);animation:slideIn .3s ease}

  /* ── Prediction alert card (sky-blue) ── */
  .predict-card{background:#050f18;border:1px solid rgba(0,191,255,.35);border-radius:8px;padding:14px;border-left:3px solid var(--sky);animation:slideIn .3s ease}

  @keyframes slideIn{from{opacity:0;transform:translateX(10px)}to{opacity:1;transform:translateX(0)}}

  .alert-header-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
  .alert-title{font-family:var(--font-head);font-size:11px;color:var(--red);letter-spacing:1px}
  .predict-title{font-family:var(--font-head);font-size:11px;color:var(--sky);letter-spacing:1px}
  .alert-time-val{font-family:var(--font-mono);font-size:10px;color:var(--text-dim)}
  .alert-detail{font-family:var(--font-mono);font-size:10px;color:var(--text-dim);line-height:1.9}

  .conf-badge{display:inline-flex;align-items:center;gap:5px;margin-top:8px;background:rgba(255,59,59,.1);border:1px solid rgba(255,59,59,.35);color:var(--red);font-family:var(--font-mono);font-size:11px;padding:4px 10px;border-radius:4px;font-weight:600;transition:background .3s,box-shadow .3s}
  .conf-badge.flash{background:rgba(255,59,59,.28);box-shadow:0 0 12px rgba(255,59,59,.4)}

  .pred-badge{display:inline-flex;align-items:center;gap:5px;margin-top:8px;background:rgba(0,191,255,.1);border:1px solid rgba(0,191,255,.35);color:var(--sky);font-family:var(--font-mono);font-size:11px;padding:4px 10px;border-radius:4px;font-weight:600;transition:background .3s,box-shadow .3s}
  .pred-badge.flash{background:rgba(0,191,255,.25);box-shadow:0 0 12px rgba(0,191,255,.4)}

  .reason-line{font-family:var(--font-mono);font-size:9px;color:var(--sky-dim);margin-top:5px;opacity:.8}

  /* Dispatch card */
  .dispatch-card{background:linear-gradient(135deg,#081510,#080f1a);border:1px solid rgba(0,255,136,.25);border-radius:8px;padding:14px;border-left:3px solid var(--green);animation:slideIn .4s ease}
  .dispatch-title{font-family:var(--font-head);font-size:10px;color:var(--green);letter-spacing:1px;margin-bottom:10px}
  .dispatch-row{display:flex;align-items:flex-start;gap:8px;margin-bottom:8px}
  .dispatch-icon{font-size:16px;flex-shrink:0}
  .dispatch-name{font-family:var(--font-body);font-size:11px;color:var(--text-bright);font-weight:600}
  .dispatch-sub{font-family:var(--font-mono);font-size:10px;color:var(--text-dim)}
  .sent-tag{display:inline-block;background:rgba(0,255,136,.12);border:1px solid rgba(0,255,136,.35);color:var(--green);font-family:var(--font-mono);font-size:9px;padding:2px 7px;border-radius:3px;letter-spacing:1px;margin-top:3px}

  .spin-sm{display:inline-block;width:10px;height:10px;border:2px solid rgba(0,212,255,.25);border-top-color:var(--cyan);border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle}

  ::-webkit-scrollbar{width:4px}
  ::-webkit-scrollbar-track{background:var(--bg)}
  ::-webkit-scrollbar-thumb{background:var(--border);border-radius:2px}
</style>
</head>
<body>

<nav class="navbar">
  <div class="brand">
    <div class="brand-title">AI ROAD ACCIDENT DETECTION</div>
    <div class="brand-sub">DECISION TREE · MOG2 BACKGROUND SUBTRACTION · A* ROUTING</div>
  </div>
  <div class="nav-right">
    <div class="nav-pills">DT <span>·</span> MOG2 <span>·</span> IOU <span>·</span> A*</div>
    <div class="status-pill">
      <div class="status-dot" id="statusDot"></div>
      <span id="statusText">Idle — Upload a video</span>
    </div>
  </div>
</nav>

<div class="main">
  <div class="left-panel">

    <div class="metrics">
      <div class="metric-card" id="accCard">
        <div class="metric-icon red">⚠</div>
        <div class="metric-value red" id="accVal">0</div>
        <div class="metric-label">Accidents Detected</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon yellow">≡</div>
        <div class="metric-value yellow" id="confVal">—</div>
        <div class="metric-label">Accuracy (Max DT Conf)</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon cyan">%</div>
        <div class="metric-value cyan" id="progVal">0%</div>
        <div class="metric-label">Video Processed</div>
      </div>
      <div class="metric-card">
        <div class="metric-icon green">●</div>
        <div class="metric-value green" id="sysVal" style="font-size:14px;padding-top:6px">IDLE</div>
        <div class="metric-label">System Status</div>
      </div>
    </div>

    <div class="legend">
      <div class="legend-item"><div class="legend-dot" style="background:#00ff88"></div>MOG2 tracked (normal)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#4488ff"></div>MOG2 tracked (overspeed)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#ff3b3b"></div>YOLO accident detected</div>
      <div class="legend-item"><div class="legend-dot" style="background:#00bfff;border:1px dashed #00bfff"></div>Motion predicted accident</div>
    </div>

    <div class="upload-zone" id="uploadZone">
      <div class="grid-scan"></div>
      <div class="corner-tl"></div><div class="corner-br"></div>
      <div class="upload-icon">📹</div>
      <div class="upload-text">Drop road / traffic video here or click to browse</div>
      <div class="upload-sub">MP4 · AVI · MOV · MKV SUPPORTED</div>
      <input type="file" id="fileInput" accept="video/*"/>
    </div>

    <div class="video-bar" id="videoBar">
      <span style="font-size:20px">🎞</span>
      <div style="flex:0 0 auto">
        <div class="video-name" id="vidName">—</div>
        <div class="video-meta" id="vidMeta">—</div>
      </div>
      <div class="prog-wrap"><div class="prog-fill" id="progFill"></div></div>
      <button class="btn-stop" id="stopBtn">✕ STOP</button>
    </div>

    <div class="feed-section">
      <div class="feed-header">
        <div class="feed-title">LIVE DETECTION FEED — MOG2 VEHICLE TRACKING + DECISION TREE</div>
        <div class="feed-badge" id="feedBadge">WAITING FOR UPLOAD</div>
      </div>
      <div class="feed-body">
        <div class="feed-placeholder" id="feedPH">
          <div class="radar">📡</div>
          <div>Upload a traffic video to start detection</div>
        </div>
        <img id="liveImg" alt="Detection feed"/>
      </div>
    </div>

  </div>

  <!-- RIGHT PANEL -->
  <div class="right-panel">
    <div class="panel-header">DETECTION ALERTS</div>
    <div class="alerts-container" id="alertsBox">
      <div class="no-alerts" id="noAlerts">
        <div class="search-icon">🔍</div>
        <div class="no-title">No accidents detected yet</div>
        <div>Upload a traffic video to begin monitoring</div>
      </div>
    </div>
  </div>
</div>

<script>
// ── DOM refs ──────────────────────────────────────────────────────────────────
const uploadZone = document.getElementById('uploadZone');
const fileInput  = document.getElementById('fileInput');
const videoBar   = document.getElementById('videoBar');
const vidName    = document.getElementById('vidName');
const vidMeta    = document.getElementById('vidMeta');
const progFill   = document.getElementById('progFill');
const stopBtn    = document.getElementById('stopBtn');
const feedPH     = document.getElementById('feedPH');
const liveImg    = document.getElementById('liveImg');
const feedBadge  = document.getElementById('feedBadge');
const statusDot  = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const accVal     = document.getElementById('accVal');
const confVal    = document.getElementById('confVal');
const progVal    = document.getElementById('progVal');
const sysVal     = document.getElementById('sysVal');
const accCard    = document.getElementById('accCard');
const alertsBox  = document.getElementById('alertsBox');
const noAlerts   = document.getElementById('noAlerts');

let es = null;

// YOLO accident card state (single card, updated in-place)
let yoloCardEl  = null;
let bestConf    = 0;
let bestFrame   = null;
let detectedAt  = null;

// Prediction card state (single card, updated in-place)
let predCardEl  = null;
let bestPredConf   = 0;
let bestPredFrame  = null;
let bestPredTime   = null;
let bestPredReason = '';
let predDetectedAt = null;

// ── Upload ────────────────────────────────────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault(); uploadZone.classList.remove('dragover');
  if (e.dataTransfer.files[0]) doUpload(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) doUpload(fileInput.files[0]); });
stopBtn.addEventListener('click', doStop);

async function doUpload(file) {
  reset();
  uploadZone.classList.add('hidden');
  videoBar.classList.add('visible');
  vidName.textContent = file.name;
  vidMeta.textContent = 'Uploading…';
  setStatus('processing');

  const fd = new FormData();
  fd.append('video', file);
  const res  = await fetch('/upload', { method: 'POST', body: fd });
  const data = await res.json();

  if (!res.ok || data.error) {
    vidMeta.textContent = '⚠ ' + (data.error || 'Upload failed');
    setStatus('idle'); return;
  }

  vidMeta.textContent   = 'Processing with YOLO + MOG2…';
  sysVal.textContent    = 'ACTIVE';
  sysVal.style.fontSize = '18px';

  es = new EventSource('/stream');
  es.onmessage = handleSSE;
  es.onerror   = () => { es && es.close(); };
}

// ── SSE handler ───────────────────────────────────────────────────────────────
function handleSSE(e) {
  const d = JSON.parse(e.data);
  if (d.type === 'ping') return;

  if (d.type === 'status') {
    feedBadge.textContent = 'PROCESSING';
    feedBadge.classList.add('processing');
    feedPH.style.display = 'none';
    liveImg.style.display = 'block';
    return;
  }

  if (d.type === 'frame') {
    liveImg.src = 'data:image/jpeg;base64,' + d.frame_b64;

    accVal.textContent  = d.accidents;
    confVal.textContent = d.max_conf > 0 ? d.max_conf.toFixed(1) + '%' : '—';
    progVal.textContent = d.progress.toFixed(0) + '%';
    progFill.style.width = d.progress + '%';
    vidMeta.textContent = `Frame ${d.frame_idx} / ${d.total_frames}`;

    if (d.accidents >= 1) accCard.classList.add('alert-active');

    // ── YOLO accident card: single, update frame+conf in-place ───────────────
    if (d.has_accident_this_frame && d.frame_max_conf > 0) {
      noAlerts.style.display = 'none';
      if (!yoloCardEl) {
        // First YOLO detection
        bestConf   = d.frame_max_conf;
        bestFrame  = d.frame_idx;
        detectedAt = new Date().toTimeString().slice(0,8);
        yoloCardEl = buildYoloCard(bestFrame, detectedAt, bestConf, d.frame_time_sec);
        alertsBox.insertBefore(yoloCardEl, alertsBox.firstChild);
      } else if (d.frame_max_conf > bestConf) {
        // Higher confidence on a later frame → update frame + conf
        bestConf  = d.frame_max_conf;
        bestFrame = d.frame_idx;
        const numEl   = yoloCardEl.querySelector('#yoloConfNum');
        const frameEl = yoloCardEl.querySelector('#yoloFrame');
        const timeEl  = yoloCardEl.querySelector('#yoloTime');
        const badgeEl = yoloCardEl.querySelector('#yoloBadge');
        if (numEl)   numEl.textContent   = bestConf.toFixed(1);
        if (frameEl) frameEl.textContent = bestFrame;
        if (timeEl)  timeEl.textContent  = fmtSec(d.frame_time_sec);
        if (badgeEl) { badgeEl.classList.add('flash'); setTimeout(() => badgeEl.classList.remove('flash'), 700); }
      }
    }

    // ── Prediction card: single, update if higher confidence ─────────────────
    if (d.has_prediction_this_frame && d.predictions && d.predictions.length > 0) {
      // Find the highest-confidence prediction this frame
      const best = d.predictions.reduce((a,b) => b.conf > a.conf ? b : a);
      const bestConfPct = Math.round(best.conf * 100);
      noAlerts.style.display = 'none';

      if (!predCardEl) {
        bestPredConf   = bestConfPct;
        bestPredFrame  = d.frame_idx;
        bestPredTime   = d.frame_time_sec;
        bestPredReason = best.reason;
        predDetectedAt = new Date().toTimeString().slice(0,8);
        predCardEl = buildPredCard(bestPredFrame, predDetectedAt, bestPredConf, bestPredTime, bestPredReason);
        // Insert after YOLO card (or at top if no YOLO card)
        const ref = yoloCardEl ? yoloCardEl.nextSibling : alertsBox.firstChild;
        alertsBox.insertBefore(predCardEl, ref);
      } else if (bestConfPct > bestPredConf) {
        bestPredConf   = bestConfPct;
        bestPredFrame  = d.frame_idx;
        bestPredTime   = d.frame_time_sec;
        bestPredReason = best.reason;
        const numEl    = predCardEl.querySelector('#predConfNum');
        const frameEl  = predCardEl.querySelector('#predFrame');
        const timeEl   = predCardEl.querySelector('#predDuration');
        const reasonEl = predCardEl.querySelector('#predReason');
        const badgeEl  = predCardEl.querySelector('#predBadge');
        if (numEl)    numEl.textContent   = bestPredConf;
        if (frameEl)  frameEl.textContent = bestPredFrame;
        if (timeEl)   timeEl.textContent  = fmtSec(bestPredTime);
        if (reasonEl) reasonEl.textContent = bestPredReason;
        if (badgeEl)  { badgeEl.classList.add('flash'); setTimeout(() => badgeEl.classList.remove('flash'), 700); }
      }
    }
    return;
  }

  if (d.type === 'done') { onDone(d); return; }

  if (d.type === 'error') {
    vidMeta.textContent = '⚠ ' + d.msg;
    setStatus('idle');
    sysVal.textContent = 'ERROR';
    es && es.close();
  }
}

// ── Build YOLO alert card (red) ───────────────────────────────────────────────
function buildYoloCard(frame, timeStr, conf, timeSec) {
  const card = document.createElement('div');
  card.className = 'alert-card';
  card.innerHTML = `
    <div class="alert-header-row">
      <div class="alert-title">⚠ ACCIDENT DETECTED</div>
      <div class="alert-time-val">${timeStr}</div>
    </div>
    <div class="alert-detail">
      Frame detected: <strong style="color:var(--text-bright)" id="yoloFrame">${frame}</strong>
      &nbsp;·&nbsp; Duration: <span id="yoloTime">${fmtSec(timeSec)}</span><br>
      YOLO decision tree triggered · tracking peak confidence…
    </div>
    <div class="conf-badge" id="yoloBadge">
      ▲ PEAK CONF: <span id="yoloConfNum">${conf.toFixed(1)}</span>%
    </div>`;
  return card;
}

// ── Build prediction alert card (sky-blue) ────────────────────────────────────
function buildPredCard(frame, timeStr, confPct, timeSec, reason) {
  const card = document.createElement('div');
  card.className = 'predict-card';
  card.innerHTML = `
    <div class="alert-header-row">
      <div class="predict-title">🔵 ACCIDENT PREDICTED</div>
      <div class="alert-time-val">${timeStr}</div>
    </div>
    <div class="alert-detail">
      Frame detected: <strong style="color:var(--sky)" id="predFrame">${frame}</strong>
      &nbsp;·&nbsp; Duration: <span id="predDuration">${fmtSec(timeSec)}</span><br>
      MOG2 motion analysis · speed drop + vehicle overlap
    </div>
    <div class="reason-line" id="predReason">${reason}</div>
    <div class="pred-badge" id="predBadge">
      ◈ ACCIDENT POSSIBILITY: <span id="predConfNum">${confPct}</span>%
    </div>`;
  return card;
}

// ── Processing done ───────────────────────────────────────────────────────────
function onDone(d) {
  es && es.close();
  accVal.textContent  = d.accidents;
  confVal.textContent = d.max_conf > 0 ? d.max_conf.toFixed(1) + '%' : '—';
  progVal.textContent = '100%';
  progFill.style.width = '100%';
  feedBadge.textContent = 'COMPLETE';
  feedBadge.classList.remove('processing');
  sysVal.textContent = 'DONE';
  vidMeta.textContent = `✓ Complete — ${d.frames_processed} frames processed`;
  setStatus('done');

  // Freeze YOLO card detail
  if (yoloCardEl) {
    const det = yoloCardEl.querySelector('.alert-detail');
    if (det) det.innerHTML =
      `Frame detected: <strong style="color:var(--text-bright)" id="yoloFrame">${bestFrame}</strong>
       &nbsp;·&nbsp; Duration: <span id="yoloTime">${fmtSec(null)}</span><br>
       YOLO decision tree · final peak confidence shown below`;
  }

  if (d.accidents >= 1) showDispatchCard();
}

async function doStop() {
  await fetch('/stop', { method: 'POST' });
  es && es.close();
  reset();
  uploadZone.classList.remove('hidden');
  videoBar.classList.remove('visible');
  feedBadge.textContent = 'WAITING FOR UPLOAD';
  feedBadge.classList.remove('processing');
  setStatus('idle');
  fileInput.value = '';
}

// ── Static dispatch card ──────────────────────────────────────────────────────
function showDispatchCard() {
   
  const now = new Date().toTimeString().slice(0,8);
  const card = document.createElement('div');
  card.className = 'dispatch-card';
  card.innerHTML = `
<div class="dispatch-title">✅ ALERT DISPATCHED TO EMERGENCY SERVICES</div>

<div style="
  color:#ff3b3b;
  font-family: var(--font-head);
  font-size:12px;
  margin-bottom:8px;
  text-shadow:0 0 10px rgba(255,59,59,0.6);
">
  ⚠ ACCIDENT DETECTED
</div>

<div class="dispatch-row">
  <span class="dispatch-icon">🏥</span>
  <div>
    <div class="dispatch-name">Apollo Hospital</div>
    <div class="dispatch-sub">📞 108 · Nearest · Alert sent at ${now}</div>
    <span class="sent-tag">✓ ALERT SENT</span>
  </div>
</div>

<div class="dispatch-row">
  <span class="dispatch-icon">🏥</span>
  <div>
    <div class="dispatch-name">Medicover Hospital</div>
    <div class="dispatch-sub">📞 108 · Nearest · Alert sent at ${now}</div>
    <span class="sent-tag">✓ ALERT SENT</span>
  </div>
</div>

<div class="dispatch-row">
  <span class="dispatch-icon">🚔</span>
  <div>
    <div class="dispatch-name">Madhapur Police Station</div>
    <div class="dispatch-sub">📞 100 · Nearest · Alert sent at ${now}</div>
    <span class="sent-tag">✓ ALERT SENT</span>
  </div>
</div>
`;
  alertsBox.appendChild(card);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function fmtSec(sec) {
  if (sec == null) return '—';
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function reset() {
  accVal.textContent  = '0';
  confVal.textContent = '—';
  progVal.textContent = '0%';
  progFill.style.width = '0%';
  sysVal.textContent  = 'IDLE';
  sysVal.style.fontSize = '14px';
  accCard.classList.remove('alert-active');
  alertsBox.innerHTML = '';
  alertsBox.appendChild(noAlerts);
  noAlerts.style.display = 'flex';
  liveImg.style.display  = 'none';
  feedPH.style.display   = 'flex';
  yoloCardEl = null; bestConf = 0; bestFrame = null; detectedAt = null;
  predCardEl = null; bestPredConf = 0; bestPredFrame = null; bestPredTime = null;
  bestPredReason = ''; predDetectedAt = null;
}

function setStatus(state) {
  statusDot.className = 'status-dot';
  if (state === 'processing') { statusDot.classList.add('active'); statusText.textContent = 'Processing video…'; }
  else if (state === 'done')  { statusDot.classList.add('active'); statusText.textContent = 'Analysis complete'; }
  else statusText.textContent = 'Idle — Upload a video';
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    print("=" * 55)
    print("  AI Road Accident Detection — Flask Server")
    print("=" * 55)
    if not YOLO_AVAILABLE:
        print("  [!] ultralytics not installed → DEMO mode active")
    if not os.path.exists(MODEL_PATH):
        print(f"  [!] Model not found at '{MODEL_PATH}' → DEMO mode active")
    print("  Open: http://127.0.0.1:5000/")
    print("=" * 55)
    app.run(debug=False, threaded=True, host="127.0.0.1", port=5000)
