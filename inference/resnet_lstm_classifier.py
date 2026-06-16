"""
ResNet18 + LSTM Temporal Classifier — Black Box Inference Module.

Maintains per-track sliding window buffers and performs temporal
behavior classification. This module is intentionally kept as a
self-contained inference engine.

Responsibilities:
    - ResNet18 spatial feature extraction
    - LSTM temporal sequence classification
    - Per-track buffer management (sliding window, interpolation, cleanup)
    - Quality/blur filtering

NOT responsible for:
    - Alert triggering
    - CSV persistence
    - HTTP uploads
    - Threshold validation
"""

import os
import sys
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque
from PIL import Image

from edge.logging_config import get_logger

logger = get_logger(__name__)


class OriginalTemporalModel(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_layers=2, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.5 if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        out = self.classifier(last_step)
        return out, None


class ResNetLSTMClassifier:
    CLASS_NAMES = ['Normal', 'Anomaly']

    def __init__(self, model_path: str, sequence_length: int = 16, device: str = 'cpu'):
        self.device = torch.device(device)
        self.sequence_length = sequence_length

        # ResNet18 spatial feature extractor (must match training-time pretrained weights)
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone.fc = nn.Identity()
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # Load optimal threshold from training artifacts
        threshold_path = os.path.join(os.path.dirname(model_path), 'best_threshold.npy')
        if os.path.exists(threshold_path):
            self.threshold = float(np.load(threshold_path))
            logger.info("Loaded threshold: %.4f", self.threshold)
        else:
            self.threshold = 0.5
            logger.warning("best_threshold.npy not found, using default %.2f", self.threshold)

        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Dynamically determine model dimensions from checkpoint keys/shapes
        input_dim = 512
        hidden_dim = 256
        num_layers = 2
        num_classes = 2
        
        if 'lstm.weight_ih_l0' in checkpoint:
            shape_ih = checkpoint['lstm.weight_ih_l0'].shape
            hidden_dim = shape_ih[0] // 4
            input_dim = shape_ih[1]
            
            layers = 0
            while f'lstm.weight_ih_l{layers}' in checkpoint:
                layers += 1
            num_layers = max(1, layers)
            
        if 'classifier.3.bias' in checkpoint:
            num_classes = checkpoint['classifier.3.bias'].shape[0]
            
        logger.info(
            "Loading %s | Detected: input_dim=%d, hidden_dim=%d, num_layers=%d, num_classes=%d",
            model_path, input_dim, hidden_dim, num_layers, num_classes,
        )
        
        self.num_classes = num_classes
        if num_classes == 2:
            self.class_names = ['Normal', 'Anomaly']
        else:
            self.class_names = ['Normal'] + [f'Anomaly (Class {i})' for i in range(1, num_classes)]

        # LSTM behavior classifier
        self.temporal_model = OriginalTemporalModel(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes
        ).to(self.device)

        self.temporal_model.load_state_dict(checkpoint)
        self.temporal_model.eval()
        logger.info("Loaded checkpoint: %s", model_path)

        # Quality/blur check parameters
        self.blur_threshold = float(os.getenv("BLUR_THRESHOLD", "30.0"))
        logger.info("Quality filter active: blur_threshold=%.1f", self.blur_threshold)

        self.track_buffers = {}
        self.track_frame_buffers = {}
        self.track_missing_count = {}
        self.grace_period = 30

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def _extract_feature(self, pil_image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.backbone(img_tensor)   
        return feat.squeeze(0)                  

    def predict(self, pil_image: Image.Image, track_id: int, full_frame: np.ndarray = None) -> tuple[str, float, int]:
        """
        Classify behavior for a tracked person.

        Returns:
            Tuple of (label, confidence, buffer_length).
        """
        # Quality filter: check if crop is too blurry
        try:
            img_np = np.array(pil_image)
            if img_np.ndim == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np
            quality = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        except Exception:
            quality = float('inf')

        is_blurry = quality < self.blur_threshold
        
        if is_blurry and track_id in self.track_buffers and len(self.track_buffers[track_id]) > 0:
            logger.debug(
                "Track %d: blurry crop (variance=%.1f < threshold=%.1f) → repeating last feature",
                track_id, quality, self.blur_threshold,
            )
            feat = self.track_buffers[track_id][-1]
            frame_to_store = (
                self.track_frame_buffers[track_id][-1]
                if track_id in self.track_frame_buffers and len(self.track_frame_buffers[track_id]) > 0
                else (full_frame.copy() if full_frame is not None else pil_image.copy())
            )
        else:
            try:
                feat = self._extract_feature(pil_image)
                frame_to_store = full_frame.copy() if full_frame is not None else pil_image.copy()
            except Exception as e:
                logger.warning("Track %d: feature extraction failed, resetting buffer: %s", track_id, e)
                self._clear_track(track_id)
                return 'Buffering...', 0.0, 0

        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.sequence_length)
            self.track_frame_buffers[track_id] = deque(maxlen=self.sequence_length)

        self.track_buffers[track_id].append(feat)
        self.track_frame_buffers[track_id].append(frame_to_store)
        self.track_missing_count[track_id] = 0
        buf_len = len(self.track_buffers[track_id])

        if buf_len < self.sequence_length:
            return 'Buffering...', 0.0, buf_len

        try:
            seq = torch.stack(list(self.track_buffers[track_id]), dim=0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs, _ = self.temporal_model(seq)
                probs = torch.softmax(outputs, dim=1)
                
                if self.num_classes == 2:
                    anomaly_prob = probs[0, 1].item()
                    predicted = 1 if anomaly_prob >= self.threshold else 0
                    confidence = anomaly_prob if predicted == 1 else probs[0, 0].item()
                    label = self.class_names[predicted]
                else:
                    max_prob, max_idx = torch.max(probs, dim=1)
                    predicted_class_idx = max_idx.item()
                    confidence = max_prob.item()
                    label = 'Normal' if predicted_class_idx == 0 else 'Anomaly'

            return label, confidence, buf_len

        except Exception as e:
            logger.warning("Track %d: temporal prediction failed, resetting buffer: %s", track_id, e)
            self._clear_track(track_id)
            return 'Buffering...', 0.0, 0

    def predict_interpolated(self, track_id: int) -> tuple[str, float, int]:
        """Predict using the interpolated (extended) buffer for a temporarily lost track."""
        buf_len = len(self.track_buffers[track_id])
        if buf_len < self.sequence_length:
            return 'Buffering...', 0.0, buf_len

        try:
            seq = torch.stack(list(self.track_buffers[track_id]), dim=0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs, _ = self.temporal_model(seq)
                probs = torch.softmax(outputs, dim=1)
                
                if self.num_classes == 2:
                    anomaly_prob = probs[0, 1].item()
                    predicted = 1 if anomaly_prob >= self.threshold else 0
                    confidence = anomaly_prob if predicted == 1 else probs[0, 0].item()
                    label = self.class_names[predicted]
                else:
                    max_prob, max_idx = torch.max(probs, dim=1)
                    predicted_class_idx = max_idx.item()
                    confidence = max_prob.item()
                    label = 'Normal' if predicted_class_idx == 0 else 'Anomaly'

            return label, confidence, buf_len

        except Exception as e:
            logger.warning("Track %d: interpolated prediction failed, resetting: %s", track_id, e)
            self._clear_track(track_id)
            return 'Buffering...', 0.0, 0

    def cleanup_tracks(self, active_ids: set[int]) -> list[int]:
        """Clean up stale tracks and interpolate temporarily lost ones."""
        to_delete = []
        interpolated_ids = []
        for tid in list(self.track_buffers.keys()):
            if tid not in active_ids:
                self.track_missing_count[tid] = self.track_missing_count.get(tid, 0) + 1
                if self.track_missing_count[tid] > self.grace_period:
                    to_delete.append(tid)
                elif self.track_missing_count[tid] <= 3 and len(self.track_buffers[tid]) > 0:
                    # Feature-level interpolation: duplicate last known feature
                    last_feat = self.track_buffers[tid][-1]
                    self.track_buffers[tid].append(last_feat)
                    # Sync frame buffer interpolation
                    if tid in self.track_frame_buffers and len(self.track_frame_buffers[tid]) > 0:
                        last_frame = self.track_frame_buffers[tid][-1]
                        self.track_frame_buffers[tid].append(last_frame)
                    interpolated_ids.append(tid)
                    logger.debug(
                        "Track %d: interpolated (missing %d frame(s))",
                        tid, self.track_missing_count[tid],
                    )
            else:
                self.track_missing_count[tid] = 0

        for tid in to_delete:
            logger.debug(
                "Track %d: deleted (missing %d frames > grace_period %d)",
                tid, self.track_missing_count.get(tid, 0), self.grace_period,
            )
            self._clear_track(tid)

        return interpolated_ids

    def _clear_track(self, track_id: int) -> None:
        """Remove all state for a single track."""
        self.track_buffers.pop(track_id, None)
        self.track_frame_buffers.pop(track_id, None)
        self.track_missing_count.pop(track_id, None)

    def clear_all_buffers(self, reason: str = "State machine transition") -> None:
        """Clear all track buffers."""
        logger.info("Cleared all buffers. Reason: %s", reason)
        self.track_buffers.clear()
        self.track_frame_buffers.clear()
        self.track_missing_count.clear()
