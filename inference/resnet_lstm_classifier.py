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
        feat = self._extract_feature(pil_image)

        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.sequence_length)

        self.track_buffers[track_id].append(feat)
        self.track_missing_count[track_id] = 0
        buf_len = len(self.track_buffers[track_id])

        print(f"TrackID={track_id}, Buffer={buf_len}")

        if buf_len < self.sequence_length:
            return 'Buffering...', 0.0, buf_len

        seq = torch.stack(list(self.track_buffers[track_id]), dim=0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, _ = self.temporal_model(seq)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        label = self.CLASS_NAMES[predicted.item()]
        return label, confidence.item(), buf_len

    def cleanup_tracks(self, active_ids: set[int]) -> None:
        to_delete = []
        for tid in self.track_buffers:
            if tid not in active_ids:
                self.track_missing_count[tid] = self.track_missing_count.get(tid, 0) + 1
                if self.track_missing_count[tid] > self.grace_period:
                    to_delete.append(tid)
            else:
                self.track_missing_count[tid] = 0
        for tid in to_delete:
            del self.track_buffers[tid]
            del self.track_missing_count[tid]

    def clear_buffer(self) -> None:
        self.track_buffers.clear()
        self.track_missing_count.clear()
