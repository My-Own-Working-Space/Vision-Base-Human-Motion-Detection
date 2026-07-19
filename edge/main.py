"""
Edge Device - FastAPI composition root.

Importing this module creates an ASGI app only. Heavy runtime side effects such
as model initialization, alert retry workers, and PMS heartbeat workers are
started by the FastAPI startup lifecycle or by ``run()`` below.
"""

from __future__ import annotations

import csv
import io
import os
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable

# Ensure project root is on the path for inference module imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_ROOT = PROJECT_ROOT / "training"
for import_path in (PROJECT_ROOT, TRAINING_ROOT):
    import_path_str = str(import_path)
    if import_path_str not in sys.path:
        sys.path.insert(0, import_path_str)

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image as PILImage

from edge.clients.api_client import ApiClient
from edge.clients.pms_bridge_client import PmsBridgeClient
from edge.config import EdgeConfig, load_config
from edge.logging_config import get_logger, setup_logging
from edge.models import DetectionEvent, EventType
from edge.services.alert_service import AlertService
from edge.services.camera_service import CameraService
from edge.services.detection_service import DetectionService
from edge.ui.stream_renderer import StreamRenderer

logger = get_logger(__name__)

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/tiff"}
VIDEO_CONTENT_TYPES = {"video/mp4", "video/x-msvideo", "video/quicktime", "video/webm"}


@dataclass
class EdgeRuntimeServices:
    config: EdgeConfig
    api_client: ApiClient
    camera_service: CameraService
    detection_service: DetectionService
    alert_service: AlertService
    renderer: StreamRenderer
    pms_bridge: PmsBridgeClient | None = None
    pms_stop_event: threading.Event | None = None
    pms_thread: threading.Thread | None = None
    models_initialized: bool = False
    alert_worker_started: bool = False


def compose_services(config: EdgeConfig | None = None) -> EdgeRuntimeServices:
    """Create runtime services without starting models or background workers."""
    cfg = config or load_config()
    api_client = ApiClient(cfg.api)
    pms_bridge = None
    if cfg.pms.enabled:
        pms_bridge = PmsBridgeClient(
            pms_base_url=cfg.pms.base_url,
            pms_endpoint=cfg.pms.endpoint,
            timeout_seconds=cfg.pms.timeout_seconds,
        )

    return EdgeRuntimeServices(
        config=cfg,
        api_client=api_client,
        camera_service=CameraService(cfg.camera, cfg.state_machine),
        detection_service=DetectionService(cfg.detection, cfg.state_machine),
        alert_service=AlertService(cfg.alert, api_client, pms_bridge=pms_bridge),
        renderer=StreamRenderer(sequence_length=cfg.detection.sequence_length),
        pms_bridge=pms_bridge,
    )


def start_runtime_services(
    services: EdgeRuntimeServices,
    initialize_models: bool = True,
    start_background_workers: bool = True,
) -> None:
    """Start lifecycle-managed side effects for a composed app."""
    cfg = services.config
    logger.info(
        "Configuration loaded: DEVICE_MODE=%s | TARGET_FPS=%d | PROCESSING=%.1fs | SLEEP=%.1fs",
        cfg.detection.device_mode,
        cfg.camera.target_fps,
        cfg.state_machine.processing_duration,
        cfg.state_machine.sleep_duration,
    )

    if initialize_models and not services.models_initialized:
        services.detection_service.initialize()
        services.models_initialized = True

    if start_background_workers and not services.alert_worker_started:
        services.alert_service.start_retry_worker()
        services.alert_worker_started = True

    if start_background_workers and services.pms_bridge is not None and services.pms_thread is None:
        services.pms_stop_event = threading.Event()
        services.pms_thread = threading.Thread(
            target=pms_bridge_worker,
            args=(
                services.pms_bridge,
                services.alert_service,
                cfg.pms.serial_number,
                cfg.pms.software_version,
                services.pms_stop_event,
            ),
            daemon=True,
            name="pms-bridge-worker",
        )
        services.pms_thread.start()
        logger.info("PMS Bridge ENABLED: forwarding detections to %s%s", cfg.pms.base_url, cfg.pms.endpoint)
    elif services.pms_bridge is None:
        logger.info("PMS Bridge DISABLED (set PMS_BRIDGE_ENABLED=true to enable)")


def stop_runtime_services(services: EdgeRuntimeServices) -> None:
    """Stop lifecycle-managed workers and release camera resources."""
    if services.alert_worker_started:
        services.alert_service.stop_retry_worker()
        services.alert_worker_started = False

    if services.pms_stop_event is not None:
        services.pms_stop_event.set()
    if services.pms_thread is not None and services.pms_thread.is_alive():
        services.pms_thread.join(timeout=10)
    services.pms_thread = None
    services.pms_stop_event = None

    services.camera_service.close()
    logger.info("Edge device shutdown complete.")


def pms_bridge_worker(
    bridge: PmsBridgeClient,
    alert_svc: AlertService,
    serial: str,
    sw_version: str,
    stop_event: threading.Event,
) -> None:
    drone_id = None
    while drone_id is None and not stop_event.is_set():
        try:
            reg_info = bridge.register(serial, sw_version)
            if reg_info and "droneId" in reg_info:
                drone_id = reg_info["droneId"]
                logger.info("Device registered with PMS. Drone ID: %s", drone_id)
                alert_svc.set_drone_id(drone_id)
            else:
                status = reg_info.get("status") if reg_info else "Offline"
                logger.warning("Device registration status: %s. Retrying in 10s...", status)
        except Exception as exc:
            logger.warning("Error during registration: %s", exc)
        stop_event.wait(timeout=10)

    battery = 95.0
    while drone_id is not None and not stop_event.is_set():
        try:
            success = bridge.send_heartbeat(drone_id, battery, 40.0)
            if success:
                logger.info("Heartbeat sent to PMS. Battery: %.1f%%", battery)
            else:
                logger.warning("Failed to send heartbeat to PMS")
            battery = max(10.0, battery - 0.1)
        except Exception as exc:
            logger.warning("Error in heartbeat thread: %s", exc)
        stop_event.wait(timeout=10)


def create_app(
    services: EdgeRuntimeServices | None = None,
    *,
    config: EdgeConfig | None = None,
    initialize_models: bool = True,
    start_background_workers: bool = True,
) -> FastAPI:
    """Create the FastAPI app with explicit dependency composition."""
    runtime_services = services or compose_services(config)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        start_runtime_services(
            runtime_services,
            initialize_models=initialize_models,
            start_background_workers=start_background_workers,
        )
        try:
            yield
        finally:
            stop_runtime_services(runtime_services)

    app = FastAPI(title="Edge Device - Suspicious Behavior Detection", lifespan=lifespan)
    app.state.services = runtime_services

    @app.get("/")
    def index(request: Request):
        accept_header = request.headers.get("accept", "")
        if "text/html" in accept_header:
            html_path = Path(__file__).resolve().parent / "ui" / "dashboard.html"
            if html_path.exists():
                return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

        cfg = runtime_services.config
        return {
            "status": "running",
            "device_mode": cfg.detection.device_mode,
            "alert_state": runtime_services.alert_service.state,
            "config": {
                "processing_duration": cfg.state_machine.processing_duration,
                "sleep_duration": cfg.state_machine.sleep_duration,
                "target_fps": cfg.camera.target_fps,
                "alert_threshold": cfg.alert.confidence_threshold,
                "confirmation_frames": cfg.alert.confirmation_frames,
                "cooldown_seconds": cfg.alert.cooldown_seconds,
            },
            "model_loaded": runtime_services.detection_service.has_classifier,
            "api_backend": cfg.api.detections_url,
            "pms_bridge": {
                "enabled": cfg.pms.enabled,
                "url": f"{cfg.pms.base_url}{cfg.pms.endpoint}" if cfg.pms.enabled else None,
            },
        }

    @app.get("/video")
    def video_feed():
        return StreamingResponse(
            generate_frames(runtime_services),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/alerts-history")
    def get_alerts_history():
        return read_alerts_history(runtime_services.config)

    @app.post("/api/analyze")
    async def analyze_file(file: UploadFile = File(...), analysis_type: str = "General"):
        return await analyze_upload(file, analysis_type, runtime_services)

    @app.post("/api/analyze-batch")
    async def analyze_batch(files: list[UploadFile] = File(...), analysis_type: str = "General"):
        items = []
        for index, upload in enumerate(files):
            result = await analyze_upload(upload, analysis_type, runtime_services)
            if isinstance(result, JSONResponse):
                items.append({
                    "index": index,
                    "filename": upload.filename,
                    "ok": False,
                    "statusCode": result.status_code,
                    "error": _json_response_body(result),
                })
            else:
                items.append({"index": index, "filename": upload.filename, "ok": True, "result": result})

        detections = [det for item in items if item["ok"] for det in item["result"].get("detections", [])]
        return {
            "analysisType": analysis_type,
            "analyzedAt": datetime.now(timezone.utc).isoformat(),
            "count": len(items),
            "summary": summarize_detections(detections),
            "items": items,
        }

    return app


def generate_frames(services: EdgeRuntimeServices) -> Generator[bytes, None, None]:
    """Main stream loop: Camera -> Detection -> Alert -> MJPEG."""
    if not services.camera_service.open():
        logger.error("Failed to open camera. Aborting stream.")
        return

    try:
        for frame in services.camera_service.read_frames():
            state = services.camera_service.update_state(frame)

            if state == "IS_PROCESSING":
                results = services.detection_service.process_frame(frame)
                detections = []
                for event, evidence in results:
                    detections.append(event)
                    if event.event_type == EventType.ANOMALY:
                        services.alert_service.evaluate(event, evidence)
                frame = services.renderer.annotate_frame(frame, detections, services.camera_service.frame_index, state)
            else:
                frame = services.renderer.annotate_frame(frame, [], services.camera_service.frame_index, state)

            jpeg_bytes = services.renderer.encode_jpeg(frame)
            if jpeg_bytes:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
    finally:
        services.camera_service.close()


def read_alerts_history(config: EdgeConfig) -> list[dict[str, object]]:
    csv_path = Path(config.alert.alerts_directory) / config.alert.csv_filename
    if not csv_path.exists():
        return []

    alerts = []
    try:
        with csv_path.open(mode="r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                alerts.append({
                    "timestamp": row.get("timestamp", ""),
                    "track_id": int(row.get("track_id", "0")),
                    "class_name": row.get("class_name", ""),
                    "confidence": float(row.get("confidence", "0.0")),
                })
    except (OSError, ValueError) as exc:
        logger.error("Error reading alert CSV history: %s", exc)
    return alerts


async def analyze_upload(
    file: UploadFile,
    analysis_type: str,
    services: EdgeRuntimeServices,
) -> dict[str, object] | JSONResponse:
    content_type = (file.content_type or "").lower()
    is_video = content_type.startswith("video/") or content_type in VIDEO_CONTENT_TYPES
    is_image = content_type.startswith("image/") or content_type in IMAGE_CONTENT_TYPES

    if not is_video and not is_image:
        return JSONResponse(
            status_code=400,
            content={"error": f"Unsupported file type: {file.content_type}. Allowed: images or videos."},
        )

    try:
        if is_image:
            result = await analyze_image_upload(file, services)
        else:
            result = await analyze_video_upload(file, services)
    except ValueError as exc:
        logger.warning("Invalid upload %s: %s", file.filename, exc)
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except OSError as exc:
        logger.error("Failed to process upload %s: %s", file.filename, exc)
        return JSONResponse(status_code=500, content={"error": "Failed to process uploaded media."})

    detections_output = result["detections"]
    return {
        "analysisType": analysis_type,
        "analyzedAt": datetime.now(timezone.utc).isoformat(),
        "mediaType": "Video" if is_video else "Image",
        "fileName": file.filename,
        "fileSize": {"width": result["width"], "height": result["height"]},
        "summary": summarize_detections(detections_output),
        "detections": detections_output,
        "modelInfo": {
            "hasClassifier": services.detection_service.has_classifier,
            "deviceMode": services.config.detection.device_mode,
        },
    }


async def analyze_image_upload(file: UploadFile, services: EdgeRuntimeServices) -> dict[str, object]:
    contents = await file.read()
    try:
        pil_image = PILImage.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to process image: {file.filename}") from exc

    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    frame = resize_frame(frame, services.config.camera.max_frame_width)
    height, width = frame.shape[:2]
    detections = run_detection_on_frame(frame, services, force_predict=True)
    return {"width": width, "height": height, "detections": detections}


async def analyze_video_upload(file: UploadFile, services: EdgeRuntimeServices) -> dict[str, object]:
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
    cap = None
    try:
        with os.fdopen(temp_fd, "wb") as tmp:
            tmp.write(await file.read())

        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {file.filename}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        sample_interval = max(1, int(fps / 2))
        frame_idx = 0
        width, height = 0, 0
        detections_output: list[dict[str, object]] = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx == 0:
                height, width = frame.shape[:2]
            if frame_idx % sample_interval == 0:
                frame = resize_frame(frame, services.config.camera.max_frame_width)
                for detection in run_detection_on_frame(frame, services, force_predict=False):
                    detection["frameIndex"] = frame_idx
                    detections_output.append(detection)
            frame_idx += 1

        return {"width": width, "height": height, "detections": detections_output}
    finally:
        if cap is not None:
            cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / width
    return cv2.resize(frame, (max_width, int(height * scale)))


def run_detection_on_frame(
    frame: np.ndarray,
    services: EdgeRuntimeServices,
    force_predict: bool,
) -> list[dict[str, object]]:
    results = services.detection_service.process_frame(frame, force_predict=force_predict)
    return [detection_event_to_response(event) for event, _ in results]


def detection_event_to_response(event: DetectionEvent) -> dict[str, object]:
    item: dict[str, object] = {
        "trackId": event.track_id,
        "eventType": event.event_type.value,
        "className": event.class_name,
        "confidence": round(event.confidence, 4),
        "timestamp": event.timestamp.isoformat(),
        "bufferLength": event.buffer_length,
    }
    if event.bbox:
        item["boundingBox"] = {
            "x1": event.bbox[0],
            "y1": event.bbox[1],
            "x2": event.bbox[2],
            "y2": event.bbox[3],
        }
    return item


def summarize_detections(detections_output: Iterable[dict[str, object]]) -> dict[str, object]:
    detections = list(detections_output)
    anomaly_count = sum(1 for item in detections if item.get("eventType") == EventType.ANOMALY.value)
    normal_count = sum(1 for item in detections if item.get("eventType") == EventType.NORMAL.value)
    persons_detected = len({item.get("trackId") for item in detections if item.get("trackId") is not None})
    return {
        "personsDetected": persons_detected,
        "anomalyCount": anomaly_count,
        "normalCount": normal_count,
        "hasAnomaly": anomaly_count > 0,
    }


def _json_response_body(response: JSONResponse) -> dict[str, object] | str:
    import json

    try:
        return json.loads(response.body.decode("utf-8"))
    except Exception:
        return response.body.decode("utf-8", errors="replace")


# ASGI app for `uvicorn edge.main:app`. No model or worker side effects run until startup.
app = create_app()


def run() -> None:
    setup_logging()
    cfg = load_config()
    uvicorn.run(
        create_app(config=cfg),
        host=cfg.server.host,
        port=cfg.server.port,
        access_log=False,
    )


if __name__ == "__main__":
    run()
