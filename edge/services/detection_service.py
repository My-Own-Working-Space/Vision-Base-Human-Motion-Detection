"""
DetectionService — Wraps the ML inference pipeline as a black box.

Returns only: DetectionEvent(event_type, confidence, timestamp).
No alert logic, no CSV persistence, no HTTP transmission.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from edge.config import DetectionConfig, StateMachineConfig
from edge.logging_config import get_logger
from edge.models import DetectionEvent, EventType

logger = get_logger(__name__)


class DetectionService:
    """
    Orchestrates YOLO tracking + LSTM temporal classification.

    Treats the underlying ML models as black boxes. Translates their raw
    outputs into clean DetectionEvent domain objects.
    """

    def __init__(
        self,
        detection_config: DetectionConfig,
        state_machine_config: StateMachineConfig,
    ) -> None:
        self._config = detection_config
        self._state_config = state_machine_config
        self._yolo = None
        self._classifier = None

    def initialize(self) -> None:
        """Load and initialize the ML models."""
        from inference.yolo_detector import YOLODetector

        logger.info("Initializing YOLOv8 + ByteTrack...")
        self._yolo = YOLODetector(
            model_path=self._config.yolo_model_path,
            device_mode=self._config.device_mode,
        )

        lstm_path = self._config.lstm_model_path
        if os.path.exists(lstm_path):
            from inference.resnet_lstm_classifier import ResNetLSTMClassifier

            logger.info("Initializing ResNet18 + LSTM classifier...")
            self._classifier = ResNetLSTMClassifier(
                model_path=lstm_path,
                sequence_length=self._config.sequence_length,
            )
            self._classifier.grace_period = self._state_config.grace_period
        else:
            logger.warning("LSTM model not found at: %s — running detection-only mode", lstm_path)

        if self._config.target_fps_warning_needed:
            logger.warning(
                "Target FPS (%d) < training FPS (%d) — classification may degrade",
                self._config.lstm_train_fps, self._config.lstm_train_fps,
            )

    @property
    def has_classifier(self) -> bool:
        """Whether the temporal LSTM classifier is loaded."""
        return self._classifier is not None

    def process_frame(
        self,
        frame: np.ndarray,
    ) -> list[tuple[DetectionEvent, Optional[np.ndarray]]]:
        """
        Run the full detection pipeline on a single frame.

        Args:
            frame: BGR frame from the camera.

        Returns:
            List of (DetectionEvent, evidence_frame_or_None) tuples.
        """
        if self._yolo is None:
            logger.error("DetectionService not initialized. Call initialize() first.")
            return []

        # YOLO + ByteTrack tracking
        results = self._yolo.track(frame)
        crops = self._yolo.get_tracked_crops(frame, results)
        now = datetime.now()

        detections: list[tuple[DetectionEvent, Optional[np.ndarray]]] = []
        active_ids: set[int] = set()

        for item in crops:
            x1, y1, x2, y2 = item["bbox"]
            track_id = item["track_id"]
            active_ids.add(track_id)

            if self._classifier:
                label, confidence, buf_len = self._classifier.predict(
                    item["image"], track_id, full_frame=frame
                )
                event = self._label_to_event(
                    label, confidence, now, track_id, buf_len, (x1, y1, x2, y2)
                )

                # Extract evidence frame for anomaly alerts
                evidence = None
                if event.event_type == EventType.ANOMALY:
                    evidence = self._get_evidence_frame(track_id)

                detections.append((event, evidence))

                if label != "Buffering...":
                    logger.debug(
                        "Track %d: %s (%.1f%%)", track_id, label, confidence * 100
                    )
            else:
                event = DetectionEvent(
                    event_type=EventType.BUFFERING,
                    confidence=0.0,
                    timestamp=now,
                    track_id=track_id,
                    class_name="no_model",
                    buffer_length=0,
                    bbox=(x1, y1, x2, y2),
                )
                detections.append((event, None))

        # Handle interpolated tracks (temporarily lost persons)
        if self._classifier:
            interpolated_ids = self._classifier.cleanup_tracks(active_ids)
            for tid in interpolated_ids:
                label, confidence, buf_len = self._classifier.predict_interpolated(tid)
                if label != "Buffering...":
                    event = self._label_to_event(
                        label, confidence, now, tid, buf_len, bbox=None
                    )
                    evidence = None
                    if event.event_type == EventType.ANOMALY:
                        evidence = self._get_evidence_frame(tid)
                    detections.append((event, evidence))

        logger.debug("Processed frame: %d person(s) detected", len(crops))
        return detections

    def _label_to_event(
        self,
        label: str,
        confidence: float,
        timestamp: datetime,
        track_id: int,
        buf_len: int,
        bbox: Optional[tuple[int, int, int, int]],
    ) -> DetectionEvent:
        """Convert raw classifier output to a DetectionEvent."""
        if label == "Buffering...":
            event_type = EventType.BUFFERING
        elif label == "Anomaly":
            event_type = EventType.ANOMALY
        else:
            event_type = EventType.NORMAL

        return DetectionEvent(
            event_type=event_type,
            confidence=confidence,
            timestamp=timestamp,
            track_id=track_id,
            class_name=label,
            buffer_length=buf_len,
            bbox=bbox,
        )

    def _get_evidence_frame(self, track_id: int) -> Optional[np.ndarray]:
        """Extract the evidence frame from the classifier's frame buffer."""
        if not self._classifier:
            return None

        frame_seq = self._classifier.track_frame_buffers.get(track_id)
        if not frame_seq or len(frame_seq) < 8:
            return None

        # Frame at index 7 is the 8th frame (middle of 16-frame window)
        return frame_seq[7]
