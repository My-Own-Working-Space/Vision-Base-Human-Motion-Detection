"""
AlertService — Anomaly confirmation state machine with cooldown.

State Flow:
    NORMAL → consecutive anomaly frames detected
    SUSPECTED → anomaly count reaches confirmation threshold
    ANOMALY → alert dispatched, cooldown timer starts
    COOLDOWN → suppresses duplicate alerts for a configurable duration
    COOLDOWN expires → back to NORMAL, ready for next anomaly

Single Responsibility: Decide whether a confirmed anomaly warrants an alert,
persist evidence locally, and transmit to the Backend API.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from edge.clients.api_client import ApiClient
from edge.config import AlertConfig
from edge.logging_config import get_logger
from edge.models import AlertPayload, DetectionEvent, EventType

logger = get_logger(__name__)


class AlertState(str, Enum):
    """Alert state machine states."""
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    COOLDOWN = "COOLDOWN"


class AlertService:
    """
    Evaluates DetectionEvents using a confirmation + cooldown state machine.

    State Machine:
        NORMAL     → anomaly detected → SUSPECTED
        SUSPECTED  → consecutive count >= threshold → dispatch alert → COOLDOWN
        SUSPECTED  → normal frame resets streak → NORMAL
        COOLDOWN   → suppresses alerts for cooldown_seconds → NORMAL
    """

    def __init__(
        self,
        alert_config: AlertConfig,
        api_client: ApiClient,
    ) -> None:
        self._config = alert_config
        self._api_client = api_client

        # State machine
        self._state = AlertState.NORMAL
        self._consecutive_anomaly_count: int = 0
        self._last_alert_time: float = 0.0

        # Evidence: keep the best (highest confidence) frame during confirmation
        self._best_evidence_frame: Optional[np.ndarray | Image.Image] = None
        self._best_confidence: float = 0.0
        self._best_event: Optional[DetectionEvent] = None

        # Queue for failed uploads — retried in background
        self._retry_queue: deque[AlertPayload] = deque(maxlen=100)
        self._retry_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Ensure alerts directory exists
        os.makedirs(self._config.alerts_directory, exist_ok=True)

    @property
    def state(self) -> str:
        """Current alert state machine state."""
        return self._state.value

    def start_retry_worker(self) -> None:
        """Start the background retry thread for failed uploads."""
        self._shutdown_event.clear()
        self._retry_thread = threading.Thread(
            target=self._retry_loop, daemon=True, name="alert-retry-worker"
        )
        self._retry_thread.start()
        logger.info("Alert retry worker started.")

    def stop_retry_worker(self) -> None:
        """Signal the retry worker to stop."""
        self._shutdown_event.set()
        if self._retry_thread and self._retry_thread.is_alive():
            self._retry_thread.join(timeout=10)
            logger.info("Alert retry worker stopped.")

    def evaluate(
        self,
        event: DetectionEvent,
        evidence_frame: Optional[np.ndarray | Image.Image] = None,
    ) -> bool:
        """
        Feed a DetectionEvent into the confirmation state machine.

        Args:
            event: The detection event to evaluate.
            evidence_frame: The frame to save as visual evidence.

        Returns:
            True if an alert was dispatched (only on confirmed anomaly).
        """
        is_anomaly = (
            event.event_type == EventType.ANOMALY
            and event.confidence >= self._config.confidence_threshold
        )

        # ── COOLDOWN: suppress all alerts until timer expires ──
        if self._state == AlertState.COOLDOWN:
            elapsed = time.monotonic() - self._last_alert_time
            if elapsed >= self._config.cooldown_seconds:
                self._transition_to(AlertState.NORMAL)
            return False

        # ── NORMAL / SUSPECTED: accumulate or reset ──
        if is_anomaly:
            self._consecutive_anomaly_count += 1

            # Track the best evidence frame (highest confidence) during confirmation window
            if event.confidence > self._best_confidence:
                self._best_confidence = event.confidence
                self._best_evidence_frame = evidence_frame
                self._best_event = event

            if self._state == AlertState.NORMAL:
                self._transition_to(AlertState.SUSPECTED)

            # Check if consecutive count meets confirmation threshold
            if self._consecutive_anomaly_count >= self._config.confirmation_frames:
                return self._dispatch_alert()

            return False
        else:
            # Normal frame breaks the anomaly streak
            if self._state == AlertState.SUSPECTED:
                self._reset_streak()
                self._transition_to(AlertState.NORMAL)
            return False

    def _dispatch_alert(self) -> bool:
        """
        Confirmed anomaly: save evidence, send to backend, enter cooldown.
        """
        event = self._best_event
        evidence = self._best_evidence_frame

        if event is None:
            self._reset_streak()
            return False

        logger.info(
            "ANOMALY CONFIRMED: %d consecutive frames | best confidence=%.1f%%",
            self._consecutive_anomaly_count,
            self._best_confidence * 100,
        )

        # Save evidence image
        image_path, image_name = self._save_evidence(event.track_id, evidence)
        if not image_path:
            logger.warning("Could not save evidence frame — skipping alert dispatch")
            self._reset_streak()
            return False

        # Build alert payload
        timestamp_iso = event.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        payload = AlertPayload(
            class_name=event.class_name,
            confidence=self._best_confidence,
            timestamp=timestamp_iso,
            latitude=self._config.latitude,
            longitude=self._config.longitude,
            image_path=image_path,
            image_name=image_name,
            track_id=event.track_id,
        )

        # Attempt API upload
        success = self._api_client.send_alert(payload)
        if success:
            payload.upload_status = "Uploaded"
            logger.info(
                "Alert dispatched to backend: %s (%.1f%%)",
                event.class_name, self._best_confidence * 100,
            )
        else:
            payload.upload_status = "Pending"
            self._retry_queue.append(payload)
            logger.warning("Alert queued for retry (backend unreachable)")

        # Persist to CSV
        self._write_csv(payload)

        # Enter cooldown
        self._last_alert_time = time.monotonic()
        self._reset_streak()
        self._transition_to(AlertState.COOLDOWN)

        return True

    def _transition_to(self, new_state: AlertState) -> None:
        """Log state transitions at DEBUG level to keep terminal clean."""
        if self._state != new_state:
            logger.debug("Alert state: %s -> %s", self._state.value, new_state.value)
            self._state = new_state

    def _reset_streak(self) -> None:
        """Reset the confirmation window accumulators."""
        self._consecutive_anomaly_count = 0
        self._best_confidence = 0.0
        self._best_evidence_frame = None
        self._best_event = None

    def clear_track(self, track_id: int) -> None:
        """No-op for global state machine (kept for interface compatibility)."""
        pass

    def _save_evidence(
        self,
        track_id: int,
        evidence_frame: Optional[np.ndarray | Image.Image],
    ) -> tuple[str, str]:
        """
        Save the evidence frame to the alerts directory.

        Returns:
            Tuple of (absolute_path, filename) or ("", "") on failure.
        """
        if evidence_frame is None:
            return "", ""

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = f"alert_track_{track_id}_{timestamp_str}.jpg"
        image_path = os.path.abspath(
            os.path.join(self._config.alerts_directory, image_name)
        )

        try:
            if isinstance(evidence_frame, Image.Image):
                evidence_frame.save(image_path)
            else:
                cv2.imwrite(image_path, evidence_frame)
            return image_path, image_name
        except Exception:
            logger.exception("Failed to save evidence frame")
            return "", ""

    def _write_csv(self, payload: AlertPayload) -> None:
        """Append alert metadata to the local CSV audit ledger."""
        csv_path = os.path.abspath(
            os.path.join(self._config.alerts_directory, self._config.csv_filename)
        )
        file_exists = os.path.exists(csv_path)

        try:
            with open(csv_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "track_id", "class_name", "confidence",
                        "lat", "lng", "image_path", "upload_status",
                    ])
                writer.writerow([
                    payload.timestamp,
                    payload.track_id,
                    payload.class_name,
                    f"{payload.confidence:.4f}",
                    f"{payload.latitude:.6f}",
                    f"{payload.longitude:.6f}",
                    payload.image_path,
                    payload.upload_status,
                ])
        except Exception:
            logger.exception("Failed to write CSV metadata")

    def _retry_loop(self) -> None:
        """Background thread: retry failed uploads from the queue."""
        while not self._shutdown_event.is_set():
            if not self._retry_queue:
                self._shutdown_event.wait(timeout=self._api_client.retry_delay)
                continue

            payload = self._retry_queue.popleft()
            logger.info("Retrying upload: %s", payload.class_name)

            success = self._api_client.send_alert(payload)
            if success:
                payload.upload_status = "Uploaded"
                logger.info("Retry succeeded")
            else:
                self._retry_queue.append(payload)
                logger.warning("Retry failed, re-queued (%d pending)", len(self._retry_queue))
                self._shutdown_event.wait(timeout=self._api_client.retry_delay)
