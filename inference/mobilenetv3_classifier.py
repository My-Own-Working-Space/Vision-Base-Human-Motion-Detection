"""
MobileNetV3 Single-Frame Classifier — Black Box Inference Module.

Classifies individual person crops as Normal or Anomaly using a
fine-tuned MobileNetV3-Large model (timm architecture).

Unlike the ResNet18+LSTM temporal classifier, this module does NOT
require a sliding window buffer. Each frame is classified independently.

Responsibilities:
    - MobileNetV3 single-frame classification
    - Per-track evidence frame storage (for alert capture)
    - Quality/blur filtering

NOT responsible for:
    - Alert triggering
    - CSV persistence
    - HTTP uploads
    - Threshold validation
"""

import os

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from edge.logging_config import get_logger

logger = get_logger(__name__)


class MobileNetV3Classifier:
    """
    Single-frame behavior classifier using a fine-tuned MobileNetV3-Large.

    Provides the same public interface as ResNetLSTMClassifier so that
    DetectionService can use either classifier interchangeably.
    """

    CLASS_NAMES = ["Normal", "Anomaly"]

    def __init__(
        self,
        model_path: str,
        sequence_length: int = 16,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.sequence_length = sequence_length

        # Load optimal threshold from training artifacts
        threshold_path = os.path.join(os.path.dirname(model_path), "best_threshold.npy")
        if os.path.exists(threshold_path):
            self.threshold = float(np.load(threshold_path))
            logger.info("Loaded threshold: %.4f", self.threshold)
        else:
            self.threshold = 0.5
            logger.warning("best_threshold.npy not found, using default %.2f", self.threshold)

        # Build MobileNetV3-Large with 2 output classes and load weights
        self.model = timm.create_model("mobilenetv3_large_100", num_classes=2)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Loaded MobileNetV3 checkpoint: %s", model_path)

        # Quality/blur check parameters
        self.blur_threshold = float(os.getenv("BLUR_THRESHOLD", "30.0"))

        # Per-track evidence frame storage (needed by AlertService)
        self.track_frame_buffers: dict[int, list[np.ndarray]] = {}
        self.track_missing_count: dict[int, int] = {}
        self.grace_period = 30

        # Image preprocessing (must match training-time transforms)
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict(
        self,
        pil_image: Image.Image,
        track_id: int,
        full_frame: np.ndarray = None,
    ) -> tuple[str, float, int]:
        """
        Classify behavior for a tracked person crop.

        Returns:
            Tuple of (label, confidence, buffer_length).
            buffer_length is always equal to sequence_length for compatibility.
        """
        # Quality filter: skip extremely blurry crops
        try:
            img_np = np.array(pil_image)
            if img_np.ndim == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np
            quality = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        except Exception:
            quality = float("inf")

        if quality < self.blur_threshold:
            logger.debug(
                "Track %d: blurry crop (variance=%.1f) — skipping classification",
                track_id, quality,
            )
            return "Buffering...", 0.0, 0

        # Store evidence frame
        frame_to_store = full_frame.copy() if full_frame is not None else pil_image.copy()
        if track_id not in self.track_frame_buffers:
            self.track_frame_buffers[track_id] = []
        # Keep only the last few frames for evidence extraction
        self.track_frame_buffers[track_id].append(frame_to_store)
        if len(self.track_frame_buffers[track_id]) > self.sequence_length:
            self.track_frame_buffers[track_id] = self.track_frame_buffers[track_id][
                -self.sequence_length :
            ]
        self.track_missing_count[track_id] = 0

        # Classify single frame
        try:
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                anomaly_prob = probs[0, 1].item()
                predicted = 1 if anomaly_prob >= self.threshold else 0
                confidence = anomaly_prob if predicted == 1 else probs[0, 0].item()
                label = self.CLASS_NAMES[predicted]

            return label, confidence, self.sequence_length

        except Exception as e:
            logger.warning(
                "Track %d: classification failed: %s", track_id, e,
            )
            return "Buffering...", 0.0, 0

    def predict_interpolated(self, track_id: int) -> tuple[str, float, int]:
        """
        No-op for single-frame classifier.
        Interpolation only makes sense for temporal (LSTM) models.
        """
        return "Buffering...", 0.0, 0

    def cleanup_tracks(self, active_ids: set[int]) -> list[int]:
        """Clean up stale tracks that have disappeared from the scene."""
        to_delete = []
        for tid in list(self.track_frame_buffers.keys()):
            if tid not in active_ids:
                self.track_missing_count[tid] = self.track_missing_count.get(tid, 0) + 1
                if self.track_missing_count[tid] > self.grace_period:
                    to_delete.append(tid)

        for tid in to_delete:
            self._clear_track(tid)

        # No interpolation for single-frame classifier
        return []

    def _clear_track(self, track_id: int) -> None:
        """Remove all state for a single track."""
        self.track_frame_buffers.pop(track_id, None)
        self.track_missing_count.pop(track_id, None)

    def clear_all_buffers(self, reason: str = "State machine transition") -> None:
        """Clear all track buffers."""
        logger.debug("Cleared all buffers. Reason: %s", reason)
        self.track_frame_buffers.clear()
        self.track_missing_count.clear()
