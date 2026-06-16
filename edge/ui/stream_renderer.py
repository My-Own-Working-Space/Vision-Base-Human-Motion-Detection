"""
StreamRenderer — Pure presentation layer for the FastAPI MJPEG video stream.

Handles frame annotation (bounding boxes, labels, HUD) and MJPEG encoding.
No business logic — receives pre-computed DetectionEvents and renders them.
"""

from __future__ import annotations

from typing import Generator

import cv2
import numpy as np

from edge.config import DetectionConfig
from edge.logging_config import get_logger
from edge.models import DetectionEvent, EventType

logger = get_logger(__name__)

# Visual constants
COLOR_NORMAL = (0, 200, 80)
COLOR_ANOMALY = (0, 60, 255)
COLOR_BUFFERING = (0, 200, 220)
COLOR_SLEEPING = (180, 130, 50)


class StreamRenderer:
    """
    Annotates frames with detection results and encodes for MJPEG streaming.
    """

    def __init__(self, sequence_length: int = 16) -> None:
        self._sequence_length = sequence_length
        self._cached_labels: dict[int, tuple[tuple, int, str]] = {}

    def annotate_frame(
        self,
        frame: np.ndarray,
        detections: list[DetectionEvent],
        frame_index: int,
        state: str,
    ) -> np.ndarray:
        """
        Draw detection annotations onto the frame.

        Args:
            frame: BGR frame to annotate (modified in place).
            detections: List of DetectionEvents for this frame.
            frame_index: Current frame counter.
            state: Current state machine state.

        Returns:
            Annotated frame.
        """
        active_ids: set[int] = set()

        for event in detections:
            if event.bbox is None:
                continue

            active_ids.add(event.track_id)
            x1, y1, x2, y2 = event.bbox

            color, bbox_thick, display_text = self._resolve_visual(event)
            self._cached_labels[event.track_id] = (color, bbox_thick, display_text)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, bbox_thick)
            self._draw_label(frame, x1, y1, display_text, color)

        # Clean stale cached labels
        stale = [tid for tid in self._cached_labels if tid not in active_ids]
        for tid in stale:
            del self._cached_labels[tid]

        # Draw sleeping overlay
        if state == "SLEEPING":
            overlay_text = "SLEEPING - Watchdog Active"
            cv2.putText(frame, overlay_text, (11, 56),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, overlay_text, (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SLEEPING, 1, cv2.LINE_AA)

        # Draw HUD
        num_tracks = len([d for d in detections if d.bbox is not None])
        self._draw_hud(frame, frame_index, num_tracks, state)

        return frame

    def encode_jpeg(self, frame: np.ndarray) -> bytes | None:
        """
        Encode a frame as JPEG bytes for MJPEG streaming.

        Returns:
            JPEG bytes or None on failure.
        """
        ret, buffer = cv2.imencode(".jpg", frame)
        return buffer.tobytes() if ret else None

    def clear_cache(self) -> None:
        """Clear cached label data (e.g., on state transition)."""
        self._cached_labels.clear()

    def _resolve_visual(
        self, event: DetectionEvent
    ) -> tuple[tuple, int, str]:
        """Map a DetectionEvent to visual style (color, thickness, text)."""
        tid = event.track_id
        seq_len = self._sequence_length

        if event.event_type == EventType.ANOMALY:
            color = COLOR_ANOMALY
            thickness = 3
            text = f"ID:{tid} Human - ANOMALY"
        elif event.event_type == EventType.BUFFERING:
            color = COLOR_BUFFERING
            thickness = 1
            text = f"ID:{tid} Human - [{event.buffer_length}/{seq_len}]"
        elif event.event_type == EventType.NORMAL:
            color = COLOR_NORMAL
            thickness = 2
            text = f"ID:{tid} Human - Normal"
        else:
            color = (128, 128, 128)
            thickness = 1
            text = f"ID:{tid} Human - [no model]"

        return color, thickness, text

    @staticmethod
    def _draw_label(frame: np.ndarray, x1: int, y1: int, text: str, color: tuple) -> None:
        """Draw a text label with background pill above a bounding box."""
        b, g, r = color
        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = (0, 0, 0) if luminance > 128 else (255, 255, 255)

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(frame, text, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1, cv2.LINE_AA)

    @staticmethod
    def _draw_hud(frame: np.ndarray, frame_idx: int, num_tracks: int, state: str) -> None:
        """Draw the pipeline HUD (head-up display) on the frame."""
        hud = f"Frame #{frame_idx:05d} | Tracks: {num_tracks} | {state} | ResNet18+LSTM"
        cv2.putText(frame, hud, (11, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, hud, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
