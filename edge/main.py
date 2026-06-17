"""
Edge Device — Main Entry Point

Wires all services together via dependency injection and starts the
FastAPI MJPEG streaming server.

Composition Root: all dependencies are created here and injected downward.
No service creates its own dependencies.
"""

from __future__ import annotations

import os
import sys
from typing import Generator

# Ensure project root is on the path for inference module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "training")))

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

from edge.config import load_config
from edge.logging_config import setup_logging, get_logger
from edge.models import EventType
from edge.clients.api_client import ApiClient
from edge.services.camera_service import CameraService
from edge.services.detection_service import DetectionService
from edge.services.alert_service import AlertService
from edge.ui.stream_renderer import StreamRenderer


# ── Bootstrap ─────────────────────────────────────────────────────────────────
setup_logging()
logger = get_logger(__name__)

config = load_config()

logger.info(
    "Configuration loaded: DEVICE_MODE=%s | TARGET_FPS=%d | PROCESSING=%.1fs | SLEEP=%.1fs",
    config.detection.device_mode,
    config.camera.target_fps,
    config.state_machine.processing_duration,
    config.state_machine.sleep_duration,
)

# ── Dependency Injection ──────────────────────────────────────────────────────
api_client = ApiClient(config.api)

camera_service = CameraService(
    camera_config=config.camera,
    state_config=config.state_machine,
)

detection_service = DetectionService(
    detection_config=config.detection,
    state_machine_config=config.state_machine,
)

alert_service = AlertService(
    alert_config=config.alert,
    api_client=api_client,
)

renderer = StreamRenderer(
    sequence_length=config.detection.sequence_length,
)

# Initialize ML models
detection_service.initialize()

# Start background retry worker for failed alerts
alert_service.start_retry_worker()

# ── FastAPI Application ───────────────────────────────────────────────────────
app = FastAPI(title="Edge Device — Suspicious Behavior Detection")


def generate_frames() -> Generator[bytes, None, None]:
    """
    Main pipeline loop: Camera → Detection → Alert → Stream.

    This generator ties together all services to produce annotated
    MJPEG frames for the video endpoint.
    """
    if not camera_service.open():
        logger.error("Failed to open camera. Aborting stream.")
        return

    try:
        for frame in camera_service.read_frames():
            state = camera_service.update_state(frame)

            if state == "IS_PROCESSING":
                # Run detection pipeline
                results = detection_service.process_frame(frame)

                detections = []
                for event, evidence in results:
                    detections.append(event)

                    # Evaluate for alert (threshold + dispatch)
                    if event.event_type == EventType.ANOMALY:
                        alert_service.evaluate(event, evidence)

                # Annotate frame with detections
                frame = renderer.annotate_frame(
                    frame, detections, camera_service.frame_index, state
                )
            else:
                # SLEEPING state — no detection, just render HUD
                frame = renderer.annotate_frame(
                    frame, [], camera_service.frame_index, state
                )

            # Encode and yield MJPEG frame
            jpeg_bytes = renderer.encode_jpeg(frame)
            if jpeg_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
    finally:
        camera_service.close()


@app.get("/")
def index(request: Request):
    """System status and configuration."""
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        html_path = os.path.join(os.path.dirname(__file__), "ui", "dashboard.html")
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())

    return {
        "status": "running",
        "device_mode": config.detection.device_mode,
        "alert_state": alert_service.state,
        "config": {
            "processing_duration": config.state_machine.processing_duration,
            "sleep_duration": config.state_machine.sleep_duration,
            "target_fps": config.camera.target_fps,
            "alert_threshold": config.alert.confidence_threshold,
            "confirmation_frames": config.alert.confirmation_frames,
            "cooldown_seconds": config.alert.cooldown_seconds,
        },
        "model_loaded": detection_service.has_classifier,
        "api_backend": config.api.detections_url,
    }


@app.get("/video")
def video_feed():
    """Live annotated MJPEG video stream."""
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/alerts-history")
def get_alerts_history():
    """Return recent alerts logged locally in the CSV file."""
    import csv
    csv_path = os.path.join(config.alert.alerts_directory, config.alert.csv_filename)
    if not os.path.exists(csv_path):
        return []
        
    alerts = []
    try:
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                alerts.append({
                    "timestamp": row.get("timestamp", ""),
                    "track_id": int(row.get("track_id", "0")),
                    "class_name": row.get("class_name", ""),
                    "confidence": float(row.get("confidence", "0.0")),
                })
    except Exception as e:
        logger.error("Error reading alert CSV history: %s", e)
        
    # Return latest alerts first (UI displays them accordingly, but let's return all)
    return alerts


@app.on_event("shutdown")
def shutdown_event():
    """Clean up resources on server shutdown."""
    alert_service.stop_retry_worker()
    camera_service.close()
    logger.info("Edge device shutdown complete.")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server.host,
        port=config.server.port,
        access_log=False,
    )
