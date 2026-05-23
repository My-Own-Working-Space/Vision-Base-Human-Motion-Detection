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

# ── Logging configuration ─────────────────────────────────────────────────────
# Silent ONLY the HTTP request access logs (blocks the GET /video frame spam)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
# Silent ALL ultralytics YOLOv8/ByteTrack console output spam
logging.getLogger("ultralytics").setLevel(logging.WARNING)

from yolo_detector import YOLODetector
from resnet_lstm_classifier import ResNetLSTMClassifier

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG BLOCK — All tunable parameters in one place
# ══════════════════════════════════════════════════════════════════════════════

# Device mode: "embedded" (Jetson/RPi) or "server" (desktop/cloud)
DEVICE_MODE        = os.getenv("DEVICE_MODE", "server")

# State machine timing (seconds)
PROCESSING_DURATION = float(os.getenv("PROCESSING_DURATION", "2.0"))   # AI active burst
SLEEP_DURATION      = float(os.getenv("SLEEP_DURATION", "3.0"))        # Rest period

# Watchdog: pixel diff threshold to wake from SLEEPING early
MOTION_THRESHOLD    = int(os.getenv("MOTION_THRESHOLD", "5000"))

# FPS throttling (server mode only)
TARGET_FPS          = int(os.getenv("TARGET_FPS", "8"))

# Training reference FPS — used to warn if inference FPS differs significantly
LSTM_TRAIN_FPS      = int(os.getenv("LSTM_TRAIN_FPS", "30"))

# Stream / model paths
IP_CAMERA_URL       = os.getenv("IP_CAMERA_URL", "0")
DEFAULT_MODEL_PATH  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../models/resnet_lstm_best.pth"))
LSTM_MODEL_PATH     = os.getenv("LSTM_MODEL_PATH", DEFAULT_MODEL_PATH)
SEQUENCE_LENGTH     = int(os.getenv("SEQUENCE_LENGTH", "16"))

# ── Visualization color palette ───────────────────────────────────────────────
COLOR_NORMAL    = (0, 200, 80)     # Green  — Normal behavior
COLOR_ANOMALY   = (0, 60, 255)     # Red    — Anomaly detected
COLOR_BUFFERING = (0, 200, 220)    # Cyan   — Accumulating frames
COLOR_SLEEPING  = (180, 130, 50)   # Blue-gray — Sleeping/watchdog mode

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

print(f"[App] DEVICE_MODE={DEVICE_MODE} | TARGET_FPS={TARGET_FPS} | PROCESSING={PROCESSING_DURATION}s | SLEEP={SLEEP_DURATION}s")

if TARGET_FPS < LSTM_TRAIN_FPS:
    print(f"[App] ⚠ WARNING — TARGET_FPS ({TARGET_FPS}) < LSTM_TRAIN_FPS ({LSTM_TRAIN_FPS}). "
          f"Temporal model was trained at {LSTM_TRAIN_FPS} FPS; inference at lower FPS may "
          f"reduce classification accuracy for fast-moving anomalies.")

print("[App] Initializing YOLOv8 + ByteTrack detector...")
yolo = YOLODetector('yolov8n.pt', device_mode=DEVICE_MODE)

print("[App] Initializing ResNet18 + LSTM classifier...")
classifier = None
if os.path.exists(LSTM_MODEL_PATH):
    classifier = ResNetLSTMClassifier(
        model_path=LSTM_MODEL_PATH,
        sequence_length=SEQUENCE_LENGTH
    )
else:
    print(f"[App] WARNING — LSTM model not found at: {LSTM_MODEL_PATH}")
    print("[App] Running in detection-only mode (no behavior classification).")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Human Suspicious Behavior Detection API",
    description="Real-time pipeline: YOLOv8 + ByteTrack → ResNet18 + LSTM | State Machine: PROCESSING ↔ SLEEPING"
)


# ══════════════════════════════════════════════════════════════════════════════
# DRAWING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _draw_label(frame, x1: int, y1: int, text: str, color: tuple) -> None:
    """Draw a filled label box with high-contrast text based on background luminance."""
    b, g, r = color
    # Calculate luminance: if bright, use black text; otherwise use white text
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv2.LINE_AA)


def _draw_pipeline_hud(frame, frame_idx: int, num_tracks: int, state: str) -> None:
    """Overlay pipeline status text in the top-left corner."""
    hud = f"Frame #{frame_idx:05d} | Tracks: {num_tracks} | {state} | ResNet18+LSTM"
    # Shadow
    cv2.putText(frame, hud, (11, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   2, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FRAME GENERATOR — State Machine
# ══════════════════════════════════════════════════════════════════════════════

def generate_frames():
    """Main generator: reads frames, runs the full pipeline with state machine, yields MJPEG bytes."""
    source = int(IP_CAMERA_URL) if IP_CAMERA_URL.isdigit() else IP_CAMERA_URL
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[App] ERROR — Cannot open stream: {IP_CAMERA_URL}")
        return

    # Detect native FPS for server-mode throttling
    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if native_fps <= 0 or native_fps > 120:
        native_fps = 30  # sensible default for IP cameras that don't report FPS

    # Server mode: compute how many frames to skip per read to match TARGET_FPS
    if DEVICE_MODE == "server" and TARGET_FPS < native_fps:
        frame_skip = max(1, round(native_fps / TARGET_FPS)) - 1
        print(f"[App] Server FPS Throttling: native={native_fps:.0f} → target={TARGET_FPS} (skip {frame_skip} per read)")
    else:
        frame_skip = 0

    # ── State machine variables ───────────────────────────────────────────────
    STATE_PROCESSING = "IS_PROCESSING"
    STATE_SLEEPING   = "SLEEPING"

    state = STATE_PROCESSING
    state_start_time = time.monotonic()
    prev_gray = None   # For SLEEPING watchdog frame differencing

    frame_idx = 0
    cached_labels: dict[int, tuple] = {}  # track_id → (color, bbox_thick, label_text)

    print(f"[App] State Machine initialized → {state}")

    while True:
        # ── Read frame (with optional FPS throttling) ─────────────────────────
        success, frame = cap.read()
        if not success:
            # If it's a video file, loop back to the beginning to stream endlessly
            if isinstance(source, str) and os.path.exists(source):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame = cap.read()
                if not success:
                    break
            else:
                break

        # Server mode throttling: skip frames to match TARGET_FPS
        if frame_skip > 0:
            for _ in range(frame_skip):
                cap.read()  # discard frames

        frame_idx += 1

        # Resize large camera frames to max 640px wide for speed
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            frame = cv2.resize(frame, (640, int(h * scale)))

        elapsed = time.monotonic() - state_start_time

        # ══════════════════════════════════════════════════════════════════════
        # STATE: IS_PROCESSING — Full AI pipeline active
        # ══════════════════════════════════════════════════════════════════════
        if state == STATE_PROCESSING:
            # ── Stage 1 & 2: YOLOv8 + ByteTrack ─────────────────────────────
            results = yolo.track(frame)
            crops   = yolo.get_tracked_crops(frame, results)
            active_ids: set[int] = set()

            for item in crops:
                x1, y1, x2, y2 = item['bbox']
                track_id        = item['track_id']
                active_ids.add(track_id)

                if classifier:
                    # ── Stage 3–6: Feature extraction + Temporal inference ────
                    label, confidence, buf_len = classifier.predict(item['image'], track_id)

                    if label == 'Anomaly':
                        color        = COLOR_ANOMALY
                        bbox_thick   = 3
                        display_text = f"ID:{track_id} Human - ANOMALY {confidence:.0%}"
                    elif label == 'Buffering...':
                        color        = COLOR_BUFFERING
                        bbox_thick   = 1
                        display_text = f"ID:{track_id} Human - [{buf_len}/{SEQUENCE_LENGTH}]"
                    else:  # Normal
                        color        = COLOR_NORMAL
                        bbox_thick   = 2
                        display_text = f"ID:{track_id} Human - Normal {confidence:.0%}"

                    cached_labels[track_id] = (color, bbox_thick, display_text)

                    # Log LSTM classification result after each inference
                    if label != 'Buffering...':
                        print(f"[LSTM] Track {track_id}: {label} ({confidence:.1%})")

                elif not classifier:
                    color        = (128, 128, 128)
                    bbox_thick   = 1
                    display_text = f"ID:{track_id} Human - [no model]"
                    cached_labels[track_id] = (color, bbox_thick, display_text)

                # ── Stage 7: Visualization ────────────────────────────────────
                if track_id in cached_labels:
                    color, bbox_thick, display_text = cached_labels[track_id]
                else:
                    color        = COLOR_BUFFERING
                    bbox_thick   = 1
                    display_text = f"ID:{track_id} Human - Detecting..."

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thick)
                _draw_label(frame, x1, y1, display_text, color)

            # Evict stale per-track buffers from memory
            if classifier:
                classifier.cleanup_tracks(active_ids)
            stale_labels = [tid for tid in cached_labels if tid not in active_ids]
            for tid in stale_labels:
                del cached_labels[tid]

            num_tracks = len(crops)

            # ── Transition: PROCESSING → SLEEPING ────────────────────────────
            if elapsed >= PROCESSING_DURATION:
                state = STATE_SLEEPING
                state_start_time = time.monotonic()
                prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(f"[StateMachine] {STATE_PROCESSING} → {STATE_SLEEPING} (after {elapsed:.1f}s)")

        # ══════════════════════════════════════════════════════════════════════
        # STATE: SLEEPING — Lightweight watchdog only, NO AI inference
        # ══════════════════════════════════════════════════════════════════════
        elif state == STATE_SLEEPING:
            num_tracks = 0
            watchdog_triggered = False

            # Simple frame differencing as motion watchdog
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion_score = int(np.sum(diff > 25))  # count pixels with significant change

                if motion_score > MOTION_THRESHOLD:
                    watchdog_triggered = True
                    print(f"[Watchdog] Motion detected! score={motion_score} > threshold={MOTION_THRESHOLD} → waking up")

            prev_gray = gray

            # Draw sleeping overlay
            overlay_text = "SLEEPING - Watchdog Active"
            cv2.putText(frame, overlay_text, (11, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, overlay_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SLEEPING, 1, cv2.LINE_AA)

            # ── Transition: SLEEPING → PROCESSING ────────────────────────────
            if watchdog_triggered or elapsed >= SLEEP_DURATION:
                reason = "watchdog trigger" if watchdog_triggered else f"timer ({elapsed:.1f}s)"
                state = STATE_PROCESSING
                state_start_time = time.monotonic()
                cached_labels.clear()
                if classifier:
                    classifier.clear_buffer()
                print(f"[StateMachine] {STATE_SLEEPING} → {STATE_PROCESSING} ({reason})")

        # HUD overlay (pipeline info + state)
        _draw_pipeline_hud(frame, frame_idx, num_tracks, state)

        # ── Stage 8: Encode & stream ──────────────────────────────────────────
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()


# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/", summary="Pipeline overview")
def index():
    return {
        "status": "running",
        "device_mode": DEVICE_MODE,
        "config": {
            "processing_duration_s": PROCESSING_DURATION,
            "sleep_duration_s": SLEEP_DURATION,
            "motion_threshold": MOTION_THRESHOLD,
            "target_fps": TARGET_FPS,
            "lstm_train_fps": LSTM_TRAIN_FPS,
        },
        "pipeline": [
            "1. Camera / Video Stream",
            "2. YOLOv8 (person detection) + ByteTrack (multi-object tracking)",
            "3. OpenCV crop → PIL Image per tracked person",
            f"4. Sliding Window Sequence Builder ({SEQUENCE_LENGTH} frames per track)",
            "5. ResNet18 Feature Extractor (512-dim spatial features)",
            "6. LSTM Temporal Classifier",
            "7. Binary Classification: Normal / Anomaly",
            "8. Visualization & MJPEG Stream",
        ],
        "state_machine": "IS_PROCESSING ↔ SLEEPING (with motion watchdog)",
        "endpoints": {
            "video_stream": "/video",
            "pipeline_info": "/"
        },
        "model_loaded": classifier is not None,
    }


@app.get("/video", summary="Live MJPEG video stream with behavior annotations")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
