"""
Human Suspicious Behavior Detection — Real-time Inference API
=============================================================
Pipeline:
  Camera/Video Stream
        ↓
  YOLOv8 (person detection) + ByteTrack (multi-object tracking)
        ↓
  OpenCV crop → PIL image per tracked person
        ↓
  Sliding Window Sequence Builder  (16 frames per track_id)
        ↓
  ResNet18 Feature Extractor       (512-dim spatial features)
        ↓
  Bi-GRU + Temporal Attention      (analyzes motion over time)
        ↓
  Binary Classification: Normal / Anomaly
        ↓
  Visualization & MJPEG Stream → FastAPI /video
"""

import cv2
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import uvicorn

from yolo_detector import YOLODetector
from resnet_lstm_classifier import ResNetLSTMClassifier

# ── Environment configuration ─────────────────────────────────────────────────
IP_CAMERA_URL  = os.getenv("IP_CAMERA_URL",  "0")
LSTM_MODEL_PATH = os.getenv("LSTM_MODEL_PATH", "../models/resnet_lstm_best.pth")
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "16"))

# ── Visualization color palette ───────────────────────────────────────────────
COLOR_NORMAL    = (0, 200, 80)     # Green  — Normal behavior
COLOR_ANOMALY   = (0, 60, 255)     # Red    — Anomaly detected
COLOR_BUFFERING = (0, 200, 220)    # Cyan   — Accumulating frames

# ── Model initialization ──────────────────────────────────────────────────────
print("[App] Initializing YOLOv8 + ByteTrack detector...")
yolo = YOLODetector('yolov8n.pt')

print("[App] Initializing ResNet18 + Bi-GRU classifier...")
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
    description="Real-time pipeline: YOLOv8 + ByteTrack → ResNet18 + Bi-GRU + Attention"
)


def _draw_label(frame, x1: int, y1: int, text: str, color: tuple) -> None:
    """Draw a filled label box with white text above a bounding box."""
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_pipeline_hud(frame, frame_idx: int, num_tracks: int) -> None:
    """Overlay pipeline status text in the top-left corner."""
    hud = f"Frame #{frame_idx:05d} | Tracks: {num_tracks} | Model: ResNet18 + Bi-GRU"
    # Shadow
    cv2.putText(frame, hud, (11, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),   2, cv2.LINE_AA)
    # Foreground
    cv2.putText(frame, hud, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def generate_frames():
    """Main generator: reads frames, runs the full pipeline, yields MJPEG bytes."""
    source = int(IP_CAMERA_URL) if IP_CAMERA_URL.isdigit() else IP_CAMERA_URL
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"[App] ERROR — Cannot open stream: {IP_CAMERA_URL}")
        return

    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        # ── Stage 1 & 2: YOLOv8 detection + ByteTrack ────────────────────────
        results = yolo.track(frame)
        crops   = yolo.get_tracked_crops(frame, results)
        active_ids: set[int] = set()

        for item in crops:
            x1, y1, x2, y2 = item['bbox']
            track_id        = item['track_id']
            active_ids.add(track_id)

            if classifier:
                # ── Stage 3–6: Feature extraction + Temporal inference ────────
                label, confidence, buf_len = classifier.predict(item['image'], track_id)

                if label == 'Anomaly':
                    color        = COLOR_ANOMALY
                    bbox_thick   = 3
                    display_text = f"ID:{track_id} ANOMALY {confidence:.0%}"
                elif label == 'Buffering...':
                    color        = COLOR_BUFFERING
                    bbox_thick   = 1
                    display_text = f"ID:{track_id} [{buf_len}/{SEQUENCE_LENGTH}]"
                else:  # Normal
                    color        = COLOR_NORMAL
                    bbox_thick   = 2
                    display_text = f"ID:{track_id} Normal {confidence:.0%}"
            else:
                color        = (128, 128, 128)
                bbox_thick   = 1
                display_text = f"ID:{track_id} [no model]"

            # ── Stage 7: Visualization ────────────────────────────────────────
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thick)
            _draw_label(frame, x1, y1, display_text, color)

        # Evict stale per-track buffers from memory
        if classifier:
            classifier.cleanup_tracks(active_ids)

        # HUD overlay (pipeline info)
        _draw_pipeline_hud(frame, frame_idx, len(crops))

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
        "pipeline": [
            "1. Camera / Video Stream",
            "2. YOLOv8 (person detection) + ByteTrack (multi-object tracking)",
            "3. OpenCV crop → PIL Image per tracked person",
            f"4. Sliding Window Sequence Builder ({SEQUENCE_LENGTH} frames per track)",
            "5. ResNet18 Feature Extractor (512-dim spatial features)",
            "6. Bi-GRU + Temporal Attention Analyzer",
            "7. Binary Classification: Normal / Anomaly",
            "8. Visualization & MJPEG Stream",
        ],
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
