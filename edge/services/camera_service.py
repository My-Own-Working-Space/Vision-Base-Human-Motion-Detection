"""
CameraService — Responsible for frame capture, FPS throttling, and frame preprocessing.

Single Responsibility: Acquire and normalize camera frames.
No detection logic, no alert logic, no business rules.
"""

from __future__ import annotations

import os
import time
from typing import Generator

import cv2
import numpy as np

from edge.config import CameraConfig, StateMachineConfig
from edge.logging_config import get_logger

logger = get_logger(__name__)


class CameraService:
    """
    Manages camera stream lifecycle and frame delivery.

    Handles:
        - Camera open/close
        - FPS throttling via frame skipping
        - Frame resize normalization
        - State machine (IS_PROCESSING / SLEEPING) transitions
        - Motion watchdog during SLEEPING state
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        state_config: StateMachineConfig,
    ) -> None:
        self._camera_config = camera_config
        self._state_config = state_config
        self._cap: cv2.VideoCapture | None = None
        self._frame_skip: int = 0
        self._frame_index: int = 0

        # State machine
        self._state: str = "IS_PROCESSING"
        self._state_start_time: float = time.monotonic()
        self._prev_gray: np.ndarray | None = None
        self._bg_motion_ema: float = float(state_config.motion_threshold)

    @property
    def state(self) -> str:
        """Current state machine state."""
        return self._state

    @property
    def frame_index(self) -> int:
        """Current frame counter."""
        return self._frame_index

    def open(self) -> bool:
        """
        Open the camera stream.

        Returns:
            True if camera opened successfully.
        """
        source = self._camera_config.source_value
        self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            logger.error("Cannot open camera stream: %s", self._camera_config.source)
            return False

        native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0 or native_fps > 120:
            native_fps = 30

        target_fps = self._camera_config.target_fps
        if target_fps < native_fps:
            self._frame_skip = max(1, round(native_fps / target_fps)) - 1
            logger.info(
                "FPS throttling active: native=%.0f → target=%d (skip=%d)",
                native_fps, target_fps, self._frame_skip,
            )
        else:
            self._frame_skip = 0

        logger.info("Camera opened: source=%s", self._camera_config.source)
        return True

    def close(self) -> None:
        """Release the camera stream."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera stream released.")

    def read_frames(self) -> Generator[np.ndarray, None, None]:
        """
        Yield preprocessed frames from the camera stream.

        Handles:
            - Frame skipping for FPS throttling
            - Video file looping
            - Resize to max width

        Yields:
            Preprocessed BGR frame as numpy array.
        """
        if self._cap is None or not self._cap.isOpened():
            logger.error("Camera not opened. Call open() first.")
            return

        source = self._camera_config.source_value

        while True:
            success, frame = self._cap.read()
            if not success:
                # Loop video files
                if isinstance(source, str) and os.path.exists(source):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    success, frame = self._cap.read()
                    if not success:
                        break
                else:
                    break

            # Skip frames for FPS throttling
            if self._frame_skip > 0:
                for _ in range(self._frame_skip):
                    self._cap.read()

            self._frame_index += 1

            # Resize to max width
            h, w = frame.shape[:2]
            max_w = self._camera_config.max_frame_width
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (max_w, int(h * scale)))

            yield frame

    def update_state(self, frame: np.ndarray) -> str:
        """
        Evaluate and transition the state machine based on elapsed time and motion.

        Args:
            frame: Current BGR frame for motion detection in SLEEPING mode.

        Returns:
            Current state after evaluation ("IS_PROCESSING" or "SLEEPING").
        """
        elapsed = time.monotonic() - self._state_start_time
        cfg = self._state_config

        if self._state == "IS_PROCESSING":
            if elapsed >= cfg.processing_duration:
                self._state = "SLEEPING"
                self._state_start_time = time.monotonic()
                self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                logger.debug("State transition: IS_PROCESSING → SLEEPING")

        elif self._state == "SLEEPING":
            watchdog_triggered = False

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._prev_gray is not None:
                diff = cv2.absdiff(self._prev_gray, gray)
                motion_score = int(np.sum(diff > 25))

                if motion_score < cfg.motion_threshold * 3:
                    self._bg_motion_ema = self._bg_motion_ema * 0.95 + motion_score * 0.05

                dynamic_threshold = int(
                    cfg.adaptive_alpha * self._bg_motion_ema + cfg.adaptive_beta
                )

                if motion_score > dynamic_threshold:
                    watchdog_triggered = True
                    logger.debug(
                        "Watchdog: motion detected (%d > threshold %d) → waking up",
                        motion_score, dynamic_threshold,
                    )

            self._prev_gray = gray

            if watchdog_triggered or elapsed >= cfg.sleep_duration:
                reason = "motion" if watchdog_triggered else "timeout"
                self._state = "IS_PROCESSING"
                self._state_start_time = time.monotonic()
                logger.debug("State transition: SLEEPING → IS_PROCESSING (reason=%s)", reason)

        return self._state
