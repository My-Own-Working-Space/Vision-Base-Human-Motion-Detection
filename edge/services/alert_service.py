"""
AlertService — Responsible for threshold validation, event creation,
retry logic, and queueing failed API requests.

Single Responsibility: Decide whether a DetectionEvent warrants an alert,
persist evidence locally, and transmit to the Backend API.
"""

from __future__ import annotations

import csv
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from edge.clients.api_client import ApiClient
from edge.config import AlertConfig
from edge.logging_config import get_logger
from edge.models import AlertPayload, DetectionEvent, EventType

logger = get_logger(__name__)


class AlertService:
    """
    Evaluates DetectionEvents against configurable thresholds and
    manages the full alert lifecycle: save evidence → upload → CSV log → retry.
    """

    def __init__(
        self,
        alert_config: AlertConfig,
        api_client: ApiClient,
    ) -> None:
        self._config = alert_config
        self._api_client = api_client

        # Track which track_ids have already triggered an alert (throttle: once per track)
        self._alert_triggered: dict[int, bool] = {}

        # Queue for failed uploads — retried in background
        self._retry_queue: deque[AlertPayload] = deque(maxlen=100)
        self._retry_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Ensure alerts directory exists
        os.makedirs(self._config.alerts_directory, exist_ok=True)

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
        Evaluate a DetectionEvent and trigger an alert if it meets criteria.

        Args:
            event: The detection event to evaluate.
            evidence_frame: The frame to save as visual evidence.

        Returns:
            True if an alert was created and dispatched.
        """
        # Only alert on anomalies above threshold
        if event.event_type != EventType.ANOMALY:
            return False

        if event.confidence < self._config.confidence_threshold:
            logger.debug(
                "Track %d: anomaly below threshold (%.2f < %.2f)",
                event.track_id, event.confidence, self._config.confidence_threshold,
            )
            return False

        # Throttle: one alert per track_id
        if self._alert_triggered.get(event.track_id, False):
            return False

        self._alert_triggered[event.track_id] = True

        # Save evidence image
        image_path, image_name = self._save_evidence(event.track_id, evidence_frame)
        if not image_path:
            logger.warning("Track %d: could not save evidence frame — skipping alert", event.track_id)
            return False

        # Build alert payload
        timestamp_iso = event.timestamp.strftime("%Y-%m-%dT%H:%M:%S")
        payload = AlertPayload(
            class_name=event.class_name,
            confidence=event.confidence,
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
                "Alert uploaded: Track %d | %s (%.1f%%)",
                event.track_id, event.class_name, event.confidence * 100,
            )
        else:
            payload.upload_status = "Pending"
            self._retry_queue.append(payload)
            logger.warning(
                "Alert queued for retry: Track %d | %s (%.1f%%)",
                event.track_id, event.class_name, event.confidence * 100,
            )

        # Persist to CSV (always, regardless of upload status)
        self._write_csv(payload)

        return True

    def clear_track(self, track_id: int) -> None:
        """Clear the alert throttle for a specific track (when track is lost)."""
        self._alert_triggered.pop(track_id, None)

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
            logger.info("Evidence saved: %s", image_path)
            return image_path, image_name
        except Exception:
            logger.exception("Failed to save evidence for Track %d", track_id)
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
            logger.info("CSV metadata recorded: %s", csv_path)
        except Exception:
            logger.exception("Failed to write CSV metadata")

    def _retry_loop(self) -> None:
        """Background thread: retry failed uploads from the queue."""
        while not self._shutdown_event.is_set():
            if not self._retry_queue:
                self._shutdown_event.wait(timeout=self._api_client.retry_delay)
                continue

            payload = self._retry_queue.popleft()
            logger.info("Retrying upload: Track %d | %s", payload.track_id, payload.class_name)

            success = self._api_client.send_alert(payload)
            if success:
                payload.upload_status = "Uploaded"
                logger.info("Retry succeeded: Track %d", payload.track_id)
                # Update CSV would require rewrite — for now, log success
            else:
                # Re-enqueue if still failing
                self._retry_queue.append(payload)
                logger.warning(
                    "Retry failed, re-queued: Track %d (%d pending)",
                    payload.track_id, len(self._retry_queue),
                )
                # Back off before next attempt
                self._shutdown_event.wait(timeout=self._api_client.retry_delay)
