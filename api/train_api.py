from flask import Flask, jsonify
from ultralytics import YOLO
import cv2
import time
import threading
import os

app = Flask(__name__)

MODEL_PATH = "../runs/detect/train/weights/best.pt"
VIDEO_PATH = "../test_media/videos/train.mp4"   # no webcam needed

model = None

latest = {
    "train_detected": False,
    "confidence": 0.0,
    "approaching": False,
    "eta_seconds": None,
    "last_updated": None
}

def estimate_eta(prev_area, area, dt):
    """
    Simple academic ETA:
    If bounding-box area increases over time => approaching.
    Growth rate -> ETA buckets.
    """
    if prev_area is None or area is None:
        return None

    growth = (area - prev_area) / max(dt, 0.001)
    if growth <= 0:
        return None

    if growth > 8000:
        return 10
    elif growth > 3000:
        return 20
    elif growth > 1000:
        return 40
    else:
        return 60

def detection_loop():
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found: {MODEL_PATH}")
        print("Train first, then run the API.")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video not found: {VIDEO_PATH}")
        print("Put train.mp4 inside test_media/videos/")
        return

    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)

    prev_area = None
    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # loop the video for continuous demo
            cap.release()
            cap = cv2.VideoCapture(VIDEO_PATH)
            prev_area = None
            prev_t = time.time()
            continue

        now = time.time()
        dt = now - prev_t

        results = model(frame, verbose=False)
        boxes = results[0].boxes

        best_conf = 0.0
        best_area = None

        for b in boxes:
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = b.xyxy[0].tolist()
                best_area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

        detected = best_conf >= 0.50
        eta = estimate_eta(prev_area, best_area if detected else None, dt)
        approaching = eta is not None

        latest["train_detected"] = bool(detected)
        latest["confidence"] = round(best_conf, 3)
        latest["approaching"] = bool(approaching)
        latest["eta_seconds"] = eta
        latest["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")

        prev_area = best_area if detected else None
        prev_t = now

        time.sleep(0.05)

@app.get("/train_status")
def train_status():
    return jsonify(latest)

if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=5000)
