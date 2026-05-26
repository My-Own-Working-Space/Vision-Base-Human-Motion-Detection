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
            print(f"[ResNetLSTMClassifier] Loaded threshold: {self.threshold:.4f}")
        else:
            self.threshold = 0.5
            print(f"[ResNetLSTMClassifier] WARNING: best_threshold.npy not found, using default {self.threshold}")

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
            
        print(f"[ResNetLSTMClassifier] Loading {model_path} | Detected: input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}, num_classes={num_classes}")
        
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
        print(f"[ResNetLSTMClassifier] Loaded checkpoint: {model_path}")

        # Quality/blur check parameters
        self.blur_threshold = float(os.getenv("BLUR_THRESHOLD", "30.0"))
        print(f"[ResNetLSTMClassifier] Quality filter active: blur_threshold={self.blur_threshold}")

        self.track_buffers = {}
        self.track_frame_buffers = {}
        self.alert_triggered = {}
        self.track_missing_count = {}
        self.grace_period = 30

        # Create alerts directory if not exists
        os.makedirs("alerts", exist_ok=True)

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

    def _trigger_alert(self, track_id: int, confidence: float, class_name: str) -> None:
        import csv
        from datetime import datetime
        import requests

        if self.alert_triggered.get(track_id, False):
            return

        self.alert_triggered[track_id] = True

        # 1. Retrieve the 8th frame (middle of the 16-frame sequence)
        frame_seq = self.track_frame_buffers.get(track_id)
        if not frame_seq or len(frame_seq) < 8:
            print(f"[AlertSystem] Warning: No frame sequence available at index 7 for TrackID={track_id}")
            return

        # Frame at index 7 is the 8th frame in the sequence
        alert_frame = frame_seq[7]
        
        # 2. Save the frame to alerts/ directory
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        image_name = f"alert_track_{track_id}_{timestamp_str}.jpg"
        image_path = os.path.abspath(os.path.join("alerts", image_name))
        
        try:
            if isinstance(alert_frame, Image.Image):
                alert_frame.save(image_path)
            else:
                # OpenCV numpy array
                cv2.imwrite(image_path, alert_frame)
            print(f"[AlertSystem] Saved alert frame to {image_path}")
        except Exception as e:
            print(f"[AlertSystem] Error saving alert frame for TrackID={track_id}: {e}")
            return

        # 3. Prepare metadata matching new contract
        lat = float(os.getenv("LATITUDE", "10.7769"))
        lng = float(os.getenv("LONGITUDE", "106.7009"))
        # Strip fractional seconds to match clean ISO format from contract, e.g. "2024-01-15T10:23:45"
        timestamp_iso = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        metadata = {
            'timestamp': timestamp_iso,
            'track_id': track_id,
            'class_name': class_name,
            'confidence': confidence,
            'lat': lat,
            'lng': lng,
            'image_path': image_path,
            'upload_status': 'Pending'
        }

        # 4. Upload to .NET Dashboard matching exact API model:
        # POST /api/detections
        # Content-Type: multipart/form-data
        # image: file
        # class_name: string
        # confidence: float
        # timestamp: string (ISO)
        # lat: float
        # lng: float
        dashboard_url = os.getenv("DASHBOARD_API_URL", "http://localhost:5000/api/detections")
        try:
            with open(image_path, 'rb') as f:
                files = {'image': (image_name, f, 'image/jpeg')}
                data = {
                    'class_name': class_name,
                    'confidence': f"{confidence:.4f}",
                    'timestamp': timestamp_iso,
                    'lat': f"{lat:.6f}",
                    'lng': f"{lng:.6f}"
                }
                response = requests.post(dashboard_url, files=files, data=data, timeout=5)
                if response.status_code in [200, 201]:
                    metadata['upload_status'] = 'Uploaded'
                    print(f"[AlertSystem] Successfully uploaded alert to .NET Dashboard ({dashboard_url})")
                else:
                    print(f"[AlertSystem] Dashboard API returned status code {response.status_code}")
        except Exception as e:
            print(f"[AlertSystem] Failed to upload alert to .NET Dashboard (offline/unreachable): {e}")

        # 5. Write metadata to local CSV file
        csv_path = os.path.abspath(os.path.join("alerts", "metadata.csv"))
        file_exists = os.path.exists(csv_path)
        try:
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(['timestamp', 'track_id', 'class_name', 'confidence', 'lat', 'lng', 'image_path', 'upload_status'])
                writer.writerow([
                    metadata['timestamp'],
                    metadata['track_id'],
                    metadata['class_name'],
                    f"{metadata['confidence']:.4f}",
                    f"{metadata['lat']:.6f}",
                    f"{metadata['lng']:.6f}",
                    metadata['image_path'],
                    metadata['upload_status']
                ])
            print(f"[AlertSystem] Recorded metadata to {csv_path}")
        except Exception as e:
            print(f"[AlertSystem] Error writing to CSV metadata: {e}")

    def predict(self, pil_image: Image.Image, track_id: int, full_frame: np.ndarray = None) -> tuple[str, float, int]:
        # Quality filter: check if crop is too blurry (e.g. from drone shaking)
        try:
            img_np = np.array(pil_image)
            if img_np.ndim == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np
            quality = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        except Exception as e:
            quality = float('inf')

        is_blurry = quality < self.blur_threshold
        
        if is_blurry and track_id in self.track_buffers and len(self.track_buffers[track_id]) > 0:
            print(f"[QualityFilter] TrackID={track_id}: crop blurry (variance={quality:.1f} < threshold={self.blur_threshold}) -> repeating last feature.")
            feat = self.track_buffers[track_id][-1]
            frame_to_store = self.track_frame_buffers[track_id][-1] if track_id in self.track_frame_buffers and len(self.track_frame_buffers[track_id]) > 0 else (full_frame.copy() if full_frame is not None else pil_image.copy())
        else:
            try:
                feat = self._extract_feature(pil_image)
                frame_to_store = full_frame.copy() if full_frame is not None else pil_image.copy()
            except Exception as e:
                print(f"[ResetReason] TrackID={track_id} feature extraction failed. Resetting buffer. Exception: {e}")
                if track_id in self.track_buffers:
                    del self.track_buffers[track_id]
                if track_id in self.track_frame_buffers:
                    del self.track_frame_buffers[track_id]
                if track_id in self.alert_triggered:
                    del self.alert_triggered[track_id]
                if track_id in self.track_missing_count:
                    del self.track_missing_count[track_id]
                return 'Buffering...', 0.0, 0

        if track_id not in self.track_buffers:
            self.track_buffers[track_id] = deque(maxlen=self.sequence_length)
            self.track_frame_buffers[track_id] = deque(maxlen=self.sequence_length)

        self.track_buffers[track_id].append(feat)
        self.track_frame_buffers[track_id].append(frame_to_store)
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
                
                if self.num_classes == 2:
                    anomaly_prob = probs[0, 1].item()
                    predicted = 1 if anomaly_prob >= self.threshold else 0
                    confidence = anomaly_prob if predicted == 1 else probs[0, 0].item()
                    label = self.class_names[predicted]
                    class_name = label
                else:
                    max_prob, max_idx = torch.max(probs, dim=1)
                    predicted_class_idx = max_idx.item()
                    confidence = max_prob.item()
                    label = 'Normal' if predicted_class_idx == 0 else 'Anomaly'
                    class_name = self.class_names[predicted_class_idx]
                    if predicted_class_idx > 0:
                        specific_anomaly_label = self.class_names[predicted_class_idx]
                        print(f"[MultiClassDetail] TrackID={track_id} predicted as {specific_anomaly_label} with probability {confidence:.1%}")

            print(f"[BufferStatus] TrackID={track_id} predicted successfully: {label} ({confidence:.1%}). Sliding window preserved (buf_len={buf_len}). No reset.")
            
            # Throttled Alert System Trigger
            if label == 'Anomaly' and confidence > 0.70:
                self._trigger_alert(track_id, confidence, class_name)

            return label, confidence, buf_len

        except Exception as e:
            print(f"[ResetReason] TrackID={track_id} temporal prediction failed. Resetting buffer. Exception: {e}")
            if track_id in self.track_buffers:
                del self.track_buffers[track_id]
            if track_id in self.track_frame_buffers:
                del self.track_frame_buffers[track_id]
            if track_id in self.alert_triggered:
                del self.alert_triggered[track_id]
            if track_id in self.track_missing_count:
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
                    if predicted_class_idx > 0:
                        specific_anomaly_label = self.class_names[predicted_class_idx]
                        print(f"[MultiClassDetail] TrackID={track_id} (Interpolated) predicted as {specific_anomaly_label} with probability {confidence:.1%}")

            print(f"[BufferStatus] TrackID={track_id} (Interpolated) predicted successfully: {label} ({confidence:.1%}). buf_len={buf_len}.")
            return label, confidence, buf_len

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
                    # Sync frame buffer interpolation
                    if tid in self.track_frame_buffers and len(self.track_frame_buffers[tid]) > 0:
                        last_frame = self.track_frame_buffers[tid][-1]
                        self.track_frame_buffers[tid].append(last_frame)
                    interpolated_ids.append(tid)
                    print(f"[Interpolation] TrackID={tid} missing for {self.track_missing_count[tid]} frame(s). Interpolated using last known feature/frame.")
            else:
                self.track_missing_count[tid] = 0

        for tid in to_delete:
            print(f"[ResetReason] Deleted TrackID={tid} from buffer. Reason: lost tracking/disappeared (missing for {self.track_missing_count[tid]} frames > grace_period {self.grace_period})")
            if tid in self.track_buffers:
                del self.track_buffers[tid]
            if tid in self.track_frame_buffers:
                del self.track_frame_buffers[tid]
            if tid in self.alert_triggered:
                del self.alert_triggered[tid]
            if tid in self.track_missing_count:
                del self.track_missing_count[tid]

        return interpolated_ids

    def clear_buffer(self, reason: str = "State Machine wakeup/transition") -> None:
        print(f"[ResetReason] Cleared all buffers. Reason: {reason}")
        self.track_buffers.clear()
        self.track_frame_buffers.clear()
        self.alert_triggered.clear()
        self.track_missing_count.clear()
