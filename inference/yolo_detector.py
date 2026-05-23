from ultralytics import YOLO
import cv2
import os
import torch
from PIL import Image


class YOLODetector:
    """
    YOLOv8 person detector with built-in ByteTrack multi-object tracking.
    Each detected person receives a persistent track_id across frames,
    which the ResNetLSTMClassifier uses to maintain per-person frame buffers.

    Supports hardware-aware model loading:
      - embedded mode: tries TensorRT engines (int8 → fp16) then falls back to .pt
      - server mode:   loads standard .pt model
    """

    def __init__(self, model_path: str = 'yolov8n.pt', device_mode: str = 'server'):
        self.device_mode = device_mode
        actual_model = self._resolve_model(model_path)
        self.model = YOLO(actual_model)
        self.device = self._resolve_device()
        print(f"[YOLODetector] Loaded model: {actual_model} | device: {self.device} | mode: {device_mode}")

    def _resolve_model(self, base_path: str) -> str:
        """Hardware-aware model selection for embedded devices."""
        if self.device_mode != 'embedded':
            return base_path

        model_dir = os.path.dirname(base_path) or '.'
        # Priority: INT8 TensorRT → FP16 TensorRT → original .pt
        for engine_name in ['yolov8n_int8.engine', 'yolov8n_fp16.engine']:
            engine_path = os.path.join(model_dir, engine_name)
            if os.path.exists(engine_path):
                print(f"[YOLODetector] Found TensorRT engine: {engine_path}")
                return engine_path

        print(f"[YOLODetector] No TensorRT engine found, falling back to {base_path}")
        return base_path

    def _resolve_device(self) -> str:
        """Select best available device."""
        if self.device_mode == 'embedded' and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    # New: ByteTrack tracking
    def track(self, frame):
        """
        Run YOLOv8 + ByteTrack on a frame.
        classes=[0] restricts detection to 'person' only (COCO class 0).
        persist=True keeps the tracker state between calls.
        Uses custom ByteTrack config with extended track_buffer for stable IDs.
        """
        tracker_cfg = os.path.join(os.path.dirname(__file__), 'bytetrack_custom.yaml')
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            classes=[0],             # person only
            conf=0.15,               # Lower confidence → detect people more reliably
            verbose=False
        )
        return results[0]

    def get_tracked_crops(self, frame, results) -> list[dict]:
        """
        Extract per-person crops with their ByteTrack track_id.

        Returns list of dicts:
            image    — PIL.Image RGB crop of the person bounding box
            bbox     — (x1, y1, x2, y2) pixel coordinates
            track_id — persistent integer ID from ByteTrack
            conf     — YOLO detection confidence
        """
        crops = []
        if results.boxes is None or results.boxes.id is None:
            return crops

        for box in results.boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            crop_cv2 = frame[y1:y2, x1:x2]
            if crop_cv2.size == 0:
                continue

            crop_pil = Image.fromarray(cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2RGB))
            crops.append({
                'image': crop_pil,
                'bbox': (x1, y1, x2, y2),
                'track_id': track_id,
                'conf': conf,
            })
        return crops

    # Legacy: single-frame detection (kept for backward compatibility)
    def detect(self, frame):
        results = self.model(frame, verbose=False)
        return results[0]

    def get_crops(self, frame, results) -> list[dict]:
        crops = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            crop_cv2 = frame[y1:y2, x1:x2]
            if crop_cv2.size == 0:
                continue
            crop_pil = Image.fromarray(cv2.cvtColor(crop_cv2, cv2.COLOR_BGR2RGB))
            crops.append({
                'image': crop_pil,
                'bbox': (x1, y1, x2, y2),
                'label': cls_name,
                'conf': float(box.conf[0]),
            })
        return crops
