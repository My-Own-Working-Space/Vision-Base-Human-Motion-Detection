import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'training'))

import torch
import torch.nn as nn
from torchvision import models, transforms
from collections import deque
from PIL import Image

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

        # ResNet18 spatial feature extractor
        self.backbone = models.resnet18(weights=None)
        self.backbone.fc = nn.Identity()
        self.backbone = self.backbone.to(self.device)
        self.backbone.eval()

        # LSTM behavior classifier
        self.temporal_model = OriginalTemporalModel(
            input_dim=512,
            hidden_dim=256,
            num_layers=2,
            num_classes=2
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.temporal_model.load_state_dict(checkpoint)
        self.temporal_model.eval()
        print(f"[ResNetLSTMClassifier] Loaded checkpoint: {model_path}")

        self.track_buffers = {}
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

    def predict(self, pil_image: Image.Image, track_id: int) -> tuple[str, float, int]:
        try:
            feat = self._extract_feature(pil_image)
        except Exception as e:
            print(f"[ResetReason] TrackID={track_id} feature extraction failed. Resetting buffer. Exception: {e}")
            if track_id in self.track_buffers:
                del self.track_buffers[track_id]
                del self.track_missing_count[track_id]
            return 'Buffering...', 0.0, 0

        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.sequence_length)

        self.track_buffers[track_id].append(feat)
        self.track_missing_count[track_id] = 0
        buf_len = len(self.track_buffers[track_id])

        print(f"TrackID={track_id}, Buffer={buf_len}")

        if buf_len < self.sequence_length:
            return 'Buffering...', 0.0, buf_len

        try:
            seq = torch.stack(list(self.track_buffers[track_id]), dim=0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs, _ = self.temporal_model(seq)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

            label = self.CLASS_NAMES[predicted.item()]
            print(f"[BufferStatus] TrackID={track_id} predicted successfully: {label} ({confidence.item():.1%}). Sliding window preserved (buf_len={buf_len}). No reset.")
            return label, confidence.item(), buf_len

        except Exception as e:
            print(f"[ResetReason] TrackID={track_id} temporal prediction failed. Resetting buffer. Exception: {e}")
            if track_id in self.track_buffers:
                del self.track_buffers[track_id]
                del self.track_missing_count[track_id]
            return 'Buffering...', 0.0, 0

    def predict_interpolated(self, track_id: int) -> tuple[str, float, int]:
        buf_len = len(self.track_buffers[track_id])
        if buf_len < self.sequence_length:
            return 'Buffering...', 0.0, buf_len

        try:
            seq = torch.stack(list(self.track_buffers[track_id]), dim=0).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs, _ = self.temporal_model(seq)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

            label = self.CLASS_NAMES[predicted.item()]
            print(f"[BufferStatus] TrackID={track_id} (Interpolated) predicted successfully: {label} ({confidence.item():.1%}). buf_len={buf_len}.")
            return label, confidence.item(), buf_len

        except Exception as e:
            print(f"[ResetReason] TrackID={track_id} interpolated prediction failed. Resetting buffer. Exception: {e}")
            if track_id in self.track_buffers:
                del self.track_buffers[track_id]
                del self.track_missing_count[track_id]
            return 'Buffering...', 0.0, 0

    def cleanup_tracks(self, active_ids: set[int]) -> list[int]:
        to_delete = []
        interpolated_ids = []
        for tid in self.track_buffers:
            if tid not in active_ids:
                self.track_missing_count[tid] = self.track_missing_count.get(tid, 0) + 1
                if self.track_missing_count[tid] > self.grace_period:
                    to_delete.append(tid)
                elif self.track_missing_count[tid] <= 3 and len(self.track_buffers[tid]) > 0:
                    # Feature-level interpolation: duplicate last known feature
                    last_feat = self.track_buffers[tid][-1]
                    self.track_buffers[tid].append(last_feat)
                    interpolated_ids.append(tid)
                    print(f"[Interpolation] TrackID={tid} missing for {self.track_missing_count[tid]} frame(s). Interpolated using last known feature.")
            else:
                self.track_missing_count[tid] = 0

        for tid in to_delete:
            print(f"[ResetReason] Deleted TrackID={tid} from buffer. Reason: lost tracking/disappeared (missing for {self.track_missing_count[tid]} frames > grace_period {self.grace_period})")
            del self.track_buffers[tid]
            del self.track_missing_count[tid]

        return interpolated_ids

    def clear_buffer(self, reason: str = "State Machine wakeup/transition") -> None:
        print(f"[ResetReason] Cleared all buffers. Reason: {reason}")
        self.track_buffers.clear()
        self.track_missing_count.clear()
