# detector.py — BirdWatch
# Motion detection via OpenCV MOG2 + burst capture + SQLite + classification

import cv2
import sqlite3
import os
import time
import subprocess
import numpy as np
from datetime import datetime
from classifier import BirdClassifier
from alerts import send_alert

# --- Config ---
DB_PATH        = os.path.expanduser("~/BirdWatch/data/birdwatch.db")
DETECTIONS_DIR = os.path.expanduser("~/BirdWatch/detections/")

MIN_CONTOUR_AREA  = 1200   # px — minimum single contour to count
MIN_TOTAL_AREA    = 2000   # px — total motion area threshold
SAVE_COOLDOWN_SEC = 8      # seconds before a new "visit" can start

# Burst config — collect frames while motion persists, then classify all
BURST_MAX_FRAMES   = 8     # max frames to collect per visit
BURST_TIMEOUT_SEC  = 3.0   # stop burst if no motion for this long
CROP_BOTTOM        = 0.82  # ignore bottom 18% of frame (cars below)
WARMUP_FRAMES      = 30

os.makedirs(DETECTIONS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

# --- Database ---
def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp         TEXT,
            image_path        TEXT,
            species           TEXT,
            confidence        REAL,
            classifier_source TEXT,
            burst_frames      INTEGER DEFAULT 1
        )
    """)
    conn.commit()

# --- Camera ---
def open_camera():
    cmd = [
        "rpicam-vid",
        "--codec", "mjpeg",
        "--width", "1640",
        "--height", "1232",
        "--framerate", "15",
        "--timeout", "0",
        "--nopreview",
        "-o", "-"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    print("Camera opened via rpicam-vid pipe")
    return ["pipe", proc, b""]

def read_frame(camera):
    _, proc, _ = camera
    while True:
        chunk = proc.stdout.read(65536)
        if not chunk:
            return False, None
        camera[2] += chunk
        buf = camera[2]
        start = buf.find(b'\xff\xd8')
        end   = buf.find(b'\xff\xd9', start) if start != -1 else -1
        if start != -1 and end != -1:
            jpg = buf[start:end+2]
            camera[2] = buf[end+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                h = frame.shape[0]
                frame = frame[:int(h * CROP_BOTTOM), :]
                return True, frame
        if len(camera[2]) > 500000:
            camera[2] = b""

def release_camera(camera):
    _, proc, _ = camera
    proc.terminate()
    proc.wait()

# --- Motion check ---
def detect_motion(frame, bg_sub):
    fg = bg_sub.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    total_area = sum(cv2.contourArea(c) for c in large)
    motion = len(large) > 0 and total_area > MIN_TOTAL_AREA
    return motion, large, fg

# --- Burst classify: run inference on all burst frames, pick best ---
def classify_burst(frames, classifier):
    """
    Run classifier on every frame in the burst.
    Returns the species with the highest single-frame confidence,
    plus the frame that produced it (best photo).
    """
    best_species    = "Unknown"
    best_confidence = 0.0
    best_source     = "unknown"
    best_frame      = frames[0]

    for frame in frames:
        species, confidence, source = classifier.classify(frame)
        if confidence > best_confidence:
            best_confidence = confidence
            best_species    = species
            best_source     = source
            best_frame      = frame

    return best_species, best_confidence, best_source, best_frame

# --- Save best frame + log burst result ---
def save_detection(best_frame, burst_frames, conn, species, confidence, source):
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{species.replace(' ', '_')}_{int(confidence * 100)}.jpg"
    fpath = os.path.join(DETECTIONS_DIR, fname)
    cv2.imwrite(fpath, best_frame)
    conn.execute(
        """INSERT INTO detections
           (timestamp, image_path, species, confidence, classifier_source, burst_frames)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (datetime.now().isoformat(), fpath, species, confidence, source, burst_frames)
    )
    conn.commit()
    print(f"  >> {species} ({confidence:.1%} via {source}) — {burst_frames} burst frames")
    if species not in ("Unknown", "background"):
        send_alert(species, confidence, source, fpath)
    return fpath

def main():
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    camera = open_camera()
    classifier = BirdClassifier()

    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=25, detectShadows=False
    )

    print(f"Warming up ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        ret, frame = read_frame(camera)
        if ret:
            bg_sub.apply(frame)

    last_save_time = 0
    print("BirdWatch running. Press Ctrl+C to stop.")

    try:
        while True:
            ret, frame = read_frame(camera)
            if not ret:
                print("Camera read failed — restarting pipe...")
                release_camera(camera)
                time.sleep(2)
                camera = open_camera()
                continue

            motion, large_contours, _ = detect_motion(frame, bg_sub)

            now = time.time()
            if motion and (now - last_save_time) > SAVE_COOLDOWN_SEC:
                print(f"Motion detected — collecting burst...")

                # --- Burst collection ---
                burst = [frame.copy()]
                last_motion_time = time.time()

                while len(burst) < BURST_MAX_FRAMES:
                    ret2, f2 = read_frame(camera)
                    if not ret2:
                        break
                    has_motion, _, _ = detect_motion(f2, bg_sub)
                    if has_motion:
                        burst.append(f2.copy())
                        last_motion_time = time.time()
                    else:
                        if time.time() - last_motion_time > BURST_TIMEOUT_SEC:
                            break

                print(f"  Burst: {len(burst)} frames")

                # --- Classify all burst frames, keep best ---
                species, confidence, source, best_frame = classify_burst(burst, classifier)

                # Annotate best frame
                for c in large_contours:
                    if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                        x, y, w, h = cv2.boundingRect(c)
                        cv2.rectangle(best_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                save_detection(best_frame, len(burst), conn,
                               species, confidence, source)
                last_save_time = time.time()

    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        release_camera(camera)
        conn.close()

if __name__ == "__main__":
    main()
