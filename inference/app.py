import cv2
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn
import logging

# Mute third-party noise
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from yolo_detector import YOLODetector
from resnet_lstm_classifier import ResNetLSTMClassifier

# Configurations
DEVICE_MODE = os.getenv("DEVICE_MODE", "server")
PROCESSING_DURATION = float(os.getenv("PROCESSING_DURATION", "2.0"))
SLEEP_DURATION = float(os.getenv("SLEEP_DURATION", "3.0"))
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "5000"))
TARGET_FPS = int(os.getenv("TARGET_FPS", "8"))
LSTM_TRAIN_FPS = int(os.getenv("LSTM_TRAIN_FPS", "30"))

IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "0")
DEFAULT_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/resnet_lstm_best.pth"))
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", DEFAULT_MODEL_PATH)
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "16"))

# Visual styles
COLOR_NORMAL = (0, 200, 80)
COLOR_ANOMALY = (0, 60, 255)
COLOR_BUFFERING = (0, 200, 220)
COLOR_SLEEPING = (180, 130, 50)

print(f"[App] DEVICE_MODE={DEVICE_MODE} | TARGET_FPS={TARGET_FPS} | PROCESSING={PROCESSING_DURATION}s | SLEEP={SLEEP_DURATION}s")

if TARGET_FPS < LSTM_TRAIN_FPS:
    print(f"[App] WARNING: Target FPS ({TARGET_FPS}) is lower than training FPS ({LSTM_TRAIN_FPS}). "
          f"Classification performance might be degraded.")

print("[App] Initializing YOLOv8 + ByteTrack...")
yolo = YOLODetector('yolov8n.pt', device_mode=DEVICE_MODE)

print("[App] Initializing ResNet18 + LSTM classifier...")
classifier = None
if os.path.exists(LSTM_MODEL_PATH):
    classifier = ResNetLSTMClassifier(
        model_path=LSTM_MODEL_PATH,
        sequence_length=SEQUENCE_LENGTH
    )
else:
    print(f"[App] WARNING: LSTM model not found at: {LSTM_MODEL_PATH}")
    print("[App] Running in detection-only mode.")

app = FastAPI(title="Suspicious Behavior Detection API")

def _draw_label(frame, x1: int, y1: int, text: str, color: tuple) -> None:
    b, g, r = color
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv2.LINE_AA)

def _draw_pipeline_hud(frame, frame_idx: int, num_tracks: int, state: str) -> None:
    hud = f"Frame #{frame_idx:05d} | Tracks: {num_tracks} | {state} | ResNet18+LSTM"
    cv2.putText(frame, hud, (11, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def generate_frames():
    source = int(IP_CAMERA_URL) if IP_CAMERA_URL.isdigit() else IP_CAMERA_URL
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[App] ERROR: Cannot open stream {IP_CAMERA_URL}")
        return

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or native_fps > 120:
        native_fps = 30

    if DEVICE_MODE == "server" and TARGET_FPS < native_fps:
        frame_skip = max(1, round(native_fps / TARGET_FPS)) - 1
        print(f"[App] FPS Throttling active: native={native_fps:.0f} -> target={TARGET_FPS} (skip={frame_skip})")
    else:
        frame_skip = 0

    state = "IS_PROCESSING"
    state_start_time = time.monotonic()
    prev_gray = None

    frame_idx = 0
    cached_labels: dict[int, tuple] = {}

    while True:
        success, frame = cap.read()
        if not success:
            if isinstance(source, str) and os.path.exists(source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
                if not success:
                    break
            else:
                break

        if frame_skip > 0:
            for _ in range(frame_skip):
                cap.read()

        frame_idx += 1

        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        elapsed = time.monotonic() - state_start_time

        if state == "IS_PROCESSING":
            results = yolo.track(frame)
            crops = yolo.get_tracked_crops(frame, results)
            print(f"Frame #{frame_idx}: Detected {len(crops)} person(s)")
            active_ids = set()

            for item in crops:
                x1, y1, x2, y2 = item['bbox']
                track_id = item['track_id']
                active_ids.add(track_id)

                if classifier:
                    label, confidence, buf_len = classifier.predict(item['image'], track_id)

                    if label == 'Anomaly':
                        color = COLOR_ANOMALY
                        bbox_thick = 3
                        display_text = f"ID:{track_id} Human - ANOMALY {confidence:.0%}"
                    elif label == 'Buffering...':
                        color = COLOR_BUFFERING
                        bbox_thick = 1
                        display_text = f"ID:{track_id} Human - [{buf_len}/{SEQUENCE_LENGTH}]"
                    else:
                        color = COLOR_NORMAL
                        bbox_thick = 2
                        display_text = f"ID:{track_id} Human - Normal {confidence:.0%}"

                    cached_labels[track_id] = (color, bbox_thick, display_text)

                    if label != 'Buffering...':
                        print(f"[LSTM] Track {track_id}: {label} ({confidence:.1%})")

                else:
                    color = (128, 128, 128)
                    bbox_thick = 1
                    display_text = f"ID:{track_id} Human - [no model]"
                    cached_labels[track_id] = (color, bbox_thick, display_text)

                if track_id in cached_labels:
                    color, bbox_thick, display_text = cached_labels[track_id]
                else:
                    color = COLOR_BUFFERING
                    bbox_thick = 1
                    display_text = f"ID:{track_id} Human - Detecting..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thick)
                _draw_label(frame, x1, y1, display_text, color)

            if classifier:
                classifier.cleanup_tracks(active_ids)
            
            stale_labels = [tid for tid in cached_labels if tid not in active_ids]
            for tid in stale_labels:
                del cached_labels[tid]

            num_tracks = len(crops)

            if elapsed >= PROCESSING_DURATION:
                state = "SLEEPING"
                state_start_time = time.monotonic()
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(f"[StateMachine] Entering SLEEPING mode.")

        elif state == "SLEEPING":
            num_tracks = 0
            watchdog_triggered = False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_score = int(np.sum(diff > 25))

                if motion_score > MOTION_THRESHOLD:
                    watchdog_triggered = True
                    print(f"[Watchdog] Motion detected ({motion_score}) -> waking up.")

            prev_gray = gray

            overlay_text = "SLEEPING - Watchdog Active"
            cv2.putText(frame, overlay_text, (11, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, overlay_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SLEEPING, 1, cv2.LINE_AA)

            if watchdog_triggered or elapsed >= SLEEP_DURATION:
                reason = "motion" if watchdog_triggered else "timeout"
                state = "IS_PROCESSING"
                state_start_time = time.monotonic()
                cached_labels.clear()
                if classifier:
                    classifier.clear_buffer()
                print(f"[StateMachine] Waking up due to {reason}.")

        _draw_pipeline_hud(frame, frame_idx, num_tracks, state)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()

@app.get("/")
def index():
    return {
        "status": "running",
        "device_mode": DEVICE_MODE,
        "config": {
            "processing_duration": PROCESSING_DURATION,
            "sleep_duration": SLEEP_DURATION,
            "motion_threshold": MOTION_THRESHOLD,
            "target_fps": TARGET_FPS,
        },
        "model_loaded": classifier is not None,
    }

@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
