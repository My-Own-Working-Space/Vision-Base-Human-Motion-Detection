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

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
import cv2
import uvicorn

from edge.config import load_config
from edge.logging_config import setup_logging, get_logger
from edge.models import EventType
from edge.clients.api_client import ApiClient
from edge.clients.pms_bridge_client import PmsBridgeClient
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

# PMS Bridge (conditional — only active when PMS_BRIDGE_ENABLED=true)
pms_bridge: PmsBridgeClient | None = None
if config.pms.enabled:
    pms_bridge = PmsBridgeClient(
        pms_base_url=config.pms.base_url,
        pms_endpoint=config.pms.endpoint,
        timeout_seconds=config.pms.timeout_seconds,
    )
    logger.info(
        "PMS Bridge ENABLED: forwarding detections to %s%s",
        config.pms.base_url, config.pms.endpoint,
    )
else:
    logger.info("PMS Bridge DISABLED (set PMS_BRIDGE_ENABLED=true to enable)")

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
    pms_bridge=pms_bridge,
)

renderer = StreamRenderer(
    sequence_length=config.detection.sequence_length,
)

# Initialize ML models
detection_service.initialize()

# Start background retry worker for failed alerts
alert_service.start_retry_worker()

def pms_bridge_worker(bridge, alert_svc, serial, sw_version):
    import time
    drone_id = None
    while drone_id is None:
        try:
            reg_info = bridge.register(serial, sw_version)
            if reg_info and "droneId" in reg_info:
                drone_id = reg_info["droneId"]
                logger.info(f"Device registered with PMS. Drone ID: {drone_id}")
                alert_svc.set_drone_id(drone_id)
            else:
                status = reg_info.get("status") if reg_info else "Offline"
                logger.warning(f"Device registration status: {status}. Retrying in 10s...")
        except Exception as e:
            logger.warning(f"Error during registration: {e}")
        time.sleep(10)

    battery = 95.0
    while True:
        try:
            success = bridge.send_heartbeat(drone_id, battery, 40.0)
            if success:
                logger.info(f"Heartbeat sent to PMS. Battery: {battery}%")
            else:
                logger.warning("Failed to send heartbeat to PMS")
            battery = max(10.0, battery - 0.1)
        except Exception as e:
            logger.warning(f"Error in heartbeat thread: {e}")
        time.sleep(10)

if pms_bridge is not None:
    import threading
    t = threading.Thread(
        target=pms_bridge_worker,
        args=(pms_bridge, alert_service, config.pms.serial_number, config.pms.software_version),
        daemon=True
    )
    t.start()


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
        "pms_bridge": {
            "enabled": config.pms.enabled,
            "url": f"{config.pms.base_url}{config.pms.endpoint}" if config.pms.enabled else None,
        },
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


@app.post("/api/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    analysis_type: str = "General",
):
    """
    Ad-hoc image and video analysis endpoint.

    Accepts an image or video upload, runs YOLO person detection and behavior
    classification, and returns structured detection results as JSON.
    Called by PMS AIAnalysisRequestedConsumer.
    """
    import io
    import tempfile
    import os
    import numpy as np
    from PIL import Image as PILImage
    from datetime import datetime

    content_type = file.content_type.lower()
    is_video = content_type.startswith("video/") or content_type in ["video/mp4", "video/x-msvideo", "video/quicktime", "video/webm"]
    is_image = content_type.startswith("image/") or content_type in ["image/jpeg", "image/png", "image/webp", "image/tiff"]

    if not is_video and not is_image:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {file.content_type}. Allowed: images or videos."},
        )

    detections_output = []
    width, height = 0, 0

    if is_image:
        try:
            contents = await file.read()
            pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
            frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            height, width = frame.shape[:2]
            
            # Resize if wider than config
            max_w = config.camera.max_frame_width
            if width > max_w:
                scale = max_w / width
                frame = cv2.resize(frame, (max_w, int(height * scale)))
                height, width = frame.shape[:2]

            results = detection_service.process_frame(frame, force_predict=True)
            for event, evidence in results:
                det = {
                    "trackId": event.track_id,
                    "eventType": event.event_type.value,
                    "className": event.class_name,
                    "confidence": round(event.confidence, 4),
                    "timestamp": event.timestamp.isoformat(),
                    "bufferLength": event.buffer_length,
                }
                if event.bbox:
                    det["boundingBox"] = {
                        "x1": event.bbox[0],
                        "y1": event.bbox[1],
                        "x2": event.bbox[2],
                        "y2": event.bbox[3],
                    }
                detections_output.append(det)
        except Exception as e:
            logger.error("Failed to process uploaded image: %s", e)
            return JSONResponse(
                status_code=400,
                content={"error": f"Failed to process image: {str(e)}"},
            )

    elif is_video:
        temp_fd, temp_path = tempfile.mkstemp(suffix=os.path.splitext(file.filename)[1])
        try:
            with os.fdopen(temp_fd, 'wb') as tmp:
                tmp.write(await file.read())
            
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Failed to open video file: {file.filename}"},
                )
            
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            # Sample at 2 frames per second to optimize speed while catching motion
            sample_interval = max(1, int(fps / 2))
            frame_idx = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx == 0:
                    height, width = frame.shape[:2]

                if frame_idx % sample_interval == 0:
                    # Resize if wider than config
                    h, w = frame.shape[:2]
                    max_w = config.camera.max_frame_width
                    if w > max_w:
                        scale = max_w / w
                        frame = cv2.resize(frame, (max_w, int(h * scale)))
                    
                    results = detection_service.process_frame(frame)
                    for event, evidence in results:
                        det = {
                            "trackId": event.track_id,
                            "eventType": event.event_type.value,
                            "className": event.class_name,
                            "confidence": round(event.confidence, 4),
                            "timestamp": event.timestamp.isoformat(),
                            "bufferLength": event.buffer_length,
                            "frameIndex": frame_idx,
                        }
                        if event.bbox:
                            det["boundingBox"] = {
                                "x1": event.bbox[0],
                                "y1": event.bbox[1],
                                "x2": event.bbox[2],
                                "y2": event.bbox[3],
                            }
                        detections_output.append(det)
                        
                frame_idx += 1
                
            cap.release()
        except Exception as e:
            logger.error("Failed to process uploaded video: %s", e)
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to process video: {str(e)}"},
            )
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # Build summary
    anomaly_count = sum(1 for d in detections_output if d["eventType"] == "Anomaly")
    normal_count = sum(1 for d in detections_output if d["eventType"] == "Normal")
    persons_detected = len(set(d["trackId"] for d in detections_output))

    return {
        "analysisType": analysis_type,
        "analyzedAt": datetime.utcnow().isoformat() + "Z",
        "mediaType": "Video" if is_video else "Image",
        "fileSize": {"width": width, "height": height},
        "summary": {
            "personsDetected": persons_detected,
            "anomalyCount": anomaly_count,
            "normalCount": normal_count,
            "hasAnomaly": anomaly_count > 0,
        },
        "detections": detections_output,
        "modelInfo": {
            "hasClassifier": detection_service.has_classifier,
            "deviceMode": config.detection.device_mode,
        },
    }


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
