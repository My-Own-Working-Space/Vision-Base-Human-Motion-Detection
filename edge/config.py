"""
Centralized configuration for the Edge Device.

All settings are loaded from environment variables with sensible defaults.
No hardcoded URLs, thresholds, API keys, or device identifiers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass(frozen=True)
class CameraConfig:
    """Camera and frame capture settings."""
    source: str = "0"
    target_fps: int = 8
    max_frame_width: int = 640

    @property
    def source_value(self) -> int | str:
        """Return int for webcam index or str for URL/path."""
        return int(self.source) if self.source.isdigit() else self.source


@dataclass(frozen=True)
class DetectionConfig:
    """Detection pipeline settings."""
    device_mode: str = "server"
    lstm_model_path: str = ""
    yolo_model_path: str = ""
    sequence_length: int = 16
    lstm_train_fps: int = 30
    blur_threshold: float = 30.0
    bytetrack_config: str = ""
    target_fps: int = 8

    @property
    def target_fps_warning_needed(self) -> bool:
        """Whether a warning about FPS mismatch should be emitted."""
        return self.target_fps < self.lstm_train_fps


@dataclass(frozen=True)
class StateMachineConfig:
    """State machine timing parameters."""
    processing_duration: float = 2.0
    sleep_duration: float = 3.0
    motion_threshold: int = 5000
    adaptive_alpha: float = 1.2
    adaptive_beta: int = 5000
    grace_period: int = 30


@dataclass(frozen=True)
class AlertConfig:
    """Alert thresholds and behavior."""
    confidence_threshold: float = 0.70
    confirmation_frames: int = 3
    cooldown_seconds: float = 60.0
    alerts_directory: str = "alerts"
    csv_filename: str = "metadata.csv"
    evidence_frame_index: int = 7
    latitude: float = 10.7769
    longitude: float = 106.7009


@dataclass(frozen=True)
class ApiConfig:
    """Backend API connection settings."""
    enabled: bool = True
    base_url: str = "http://localhost:5000"
    detections_endpoint: str = "/api/detections"
    timeout_seconds: int = 5
    max_retries: int = 3
    retry_delay_seconds: float = 5.0

    @property
    def detections_url(self) -> str:
        return f"{self.base_url.rstrip('/')}{self.detections_endpoint}"


@dataclass(frozen=True)
class ServerConfig:
    """FastAPI server settings."""
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass(frozen=True)
class PmsConfig:
    enabled: bool = False
    # PMS Backend URL Options:
    # - Local (Testing): http://localhost:5196
    # - Remote (Production): https://uavpms.ddns.net (Swagger: https://uavpms.ddns.net/swagger/index.html)
    base_url: str = "http://localhost:5196"
    endpoint: str = "/api/v1/vision/detections"
    timeout_seconds: int = 5
    serial_number: str = "RPI-123456"
    software_version: str = "1.0.0"


@dataclass(frozen=True)
class EdgeConfig:
    """Root configuration container — aggregates all sub-configs."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    pms: PmsConfig = field(default_factory=PmsConfig)


def load_config() -> EdgeConfig:
    """Load configuration from environment variables with defaults."""
    project_root = Path(__file__).resolve().parent.parent

    # Resolve model paths relative to project root
    default_lstm = str(project_root / "models" / "cnn_lstm_ucf_fixed.pt")
    default_yolo = str(project_root / "models" / "yolov8n.pt")
    default_bytetrack = str(project_root / "inference" / "bytetrack_custom.yaml")
    default_motion_threshold = int(os.getenv("MOTION_THRESHOLD", "5000"))

    return EdgeConfig(
        camera=CameraConfig(
            source=os.getenv("IP_CAMERA_URL", "0"),
            target_fps=int(os.getenv("TARGET_FPS", "8")),
            max_frame_width=int(os.getenv("MAX_FRAME_WIDTH", "640")),
        ),
        detection=DetectionConfig(
            device_mode=os.getenv("DEVICE_MODE", "server"),
            lstm_model_path=os.getenv("LSTM_MODEL_PATH", default_lstm),
            yolo_model_path=os.getenv("YOLO_MODEL_PATH", default_yolo),
            sequence_length=int(os.getenv("SEQUENCE_LENGTH", "16")),
            lstm_train_fps=int(os.getenv("LSTM_TRAIN_FPS", "30")),
            blur_threshold=float(os.getenv("BLUR_THRESHOLD", "30.0")),
            bytetrack_config=os.getenv("BYTETRACK_CONFIG", default_bytetrack),
            target_fps=int(os.getenv("TARGET_FPS", "8")),
        ),
        state_machine=StateMachineConfig(
            processing_duration=float(os.getenv("PROCESSING_DURATION", "2.0")),
            sleep_duration=float(os.getenv("SLEEP_DURATION", "3.0")),
            motion_threshold=default_motion_threshold,
            adaptive_alpha=float(os.getenv("ADAPTIVE_ALPHA", "1.2")),
            adaptive_beta=int(os.getenv("ADAPTIVE_BETA", str(default_motion_threshold))),
            grace_period=int(os.getenv("GRACE_PERIOD", "30")),
        ),
        alert=AlertConfig(
            confidence_threshold=float(os.getenv("ALERT_CONFIDENCE_THRESHOLD", "0.70")),
            confirmation_frames=int(os.getenv("ALERT_CONFIRMATION_FRAMES", "3")),
            cooldown_seconds=float(os.getenv("ALERT_COOLDOWN_SECONDS", "60.0")),
            alerts_directory=os.getenv("ALERTS_DIRECTORY", "alerts"),
            csv_filename=os.getenv("ALERTS_CSV_FILENAME", "metadata.csv"),
            evidence_frame_index=int(os.getenv("EVIDENCE_FRAME_INDEX", "7")),
            latitude=float(os.getenv("LATITUDE", "10.7769")),
            longitude=float(os.getenv("LONGITUDE", "106.7009")),
        ),
        api=ApiConfig(
            enabled=os.getenv("DASHBOARD_API_ENABLED", "true").lower() in ("true", "1", "yes"),
            base_url=os.getenv("DASHBOARD_API_URL", "http://localhost:5000"),
            detections_endpoint=os.getenv("DETECTIONS_ENDPOINT", "/api/detections"),
            timeout_seconds=int(os.getenv("API_TIMEOUT_SECONDS", "5")),
            max_retries=int(os.getenv("API_MAX_RETRIES", "3")),
            retry_delay_seconds=float(os.getenv("API_RETRY_DELAY_SECONDS", "5.0")),
        ),
        server=ServerConfig(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "8000")),
        ),
        pms=PmsConfig(
            enabled=os.getenv("PMS_BRIDGE_ENABLED", "false").lower() in ("true", "1", "yes"),
            base_url=os.getenv("PMS_BRIDGE_URL", "http://localhost:5196"),
            endpoint=os.getenv("PMS_BRIDGE_ENDPOINT", "/api/v1/vision/detections"),
            timeout_seconds=int(os.getenv("PMS_BRIDGE_TIMEOUT", "5")),
            serial_number=os.getenv("PMS_BRIDGE_SERIAL", "RPI-123456"),
            software_version=os.getenv("PMS_BRIDGE_SOFTWARE_VERSION", "1.0.0"),
        ),
    )
