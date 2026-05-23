from ultralytics import YOLO
import cv2
import os
import torch
from PIL import Image

class YOLODetector:
    def __init__(self, model_path: str = 'yolov8n.pt', device_mode: str = 'server'):
        self.device_mode = device_mode
        actual_model = self._resolve_model(model_path)
        self.model = YOLO(actual_model)
        self.device = self._resolve_device()
        print(f"[YOLODetector] Loaded model: {actual_model} | device: {self.device} | mode: {device_mode}")

    def _resolve_model(self, base_path: str) -> str:
        if self.device_mode != 'embedded':
            return base_path

        model_dir = os.path.dirname(base_path) or '.'
        for engine_name in ['yolov8n_int8.engine', 'yolov8n_fp16.engine']:
            engine_path = os.path.join(model_dir, engine_name)
            if os.path.exists(engine_path):
                print(f"[YOLODetector] Found TensorRT engine: {engine_path}")
                return engine_path

        print(f"[YOLODetector] No TensorRT engine found, falling back to {base_path}")
        return base_path

    def _resolve_device(self) -> str:
        if self.device_mode == 'embedded' and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def track(self, frame):
        tracker_cfg = os.path.join(os.path.dirname(__file__), 'bytetrack_custom.yaml')
        results = self.model.track(
            frame,
            persist=True,
            tracker=tracker_cfg,
            classes=[0],  # Person only
            conf=0.15,
            verbose=False
        )
        return results[0]

    def get_tracked_crops(self, frame, results) -> list[dict]:
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
