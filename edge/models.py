"""
Domain models for the Edge Device event-driven architecture.

Pure data containers — no behavior, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np
from PIL import Image


class EventType(str, Enum):
    """Possible detection event types."""
    NORMAL = "Normal"
    ANOMALY = "Anomaly"
    BUFFERING = "Buffering"


@dataclass
class DetectionEvent:
    """
    Output of DetectionService — the only contract between detection and alerting.

    Attributes:
        event_type: Classification result (Normal, Anomaly, or Buffering).
        confidence: Model confidence probability [0.0, 1.0].
        timestamp: When the detection was made (ISO-8601).
        track_id: ByteTrack persistent identity of the detected person.
        class_name: Specific anomaly class label (e.g., 'Anomaly', 'Fighting').
        buffer_length: Current sliding window fill level.
        bbox: Bounding box coordinates (x1, y1, x2, y2) if available.
    """
    event_type: EventType
    confidence: float
    timestamp: datetime
    track_id: int
    class_name: str = ""
    buffer_length: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None


@dataclass
class AlertPayload:
    """
    Structured payload for transmission to the Backend API.

    Created by AlertService when a DetectionEvent passes threshold validation.
    """
    class_name: str
    confidence: float
    timestamp: str
    latitude: float
    longitude: float
    image_path: str
    image_name: str
    track_id: int
    upload_status: str = "Pending"


@dataclass
class FrameContext:
    """
    Container for a single frame's processing context.

    Carries the raw frame and all detection results through the pipeline.
    """
    frame: np.ndarray
    frame_index: int
    detections: list[DetectionEvent] = field(default_factory=list)
    evidence_frames: dict[int, Image.Image | np.ndarray] = field(default_factory=dict)
    state: str = "IS_PROCESSING"
    num_tracks: int = 0
