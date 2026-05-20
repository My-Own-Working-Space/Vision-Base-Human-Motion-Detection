from ultralytics import YOLO
import cv2
from PIL import Image


class YOLODetector:
    """
    YOLOv8 person detector with built-in ByteTrack multi-object tracking.
    Each detected person receives a persistent track_id across frames,
    which the ResNetLSTMClassifier uses to maintain per-person frame buffers.
    """

    def __init__(self, model_path: str = 'yolov8n.pt'):
        self.model = YOLO(model_path)
        print(f"[YOLODetector] Loaded model: {model_path}")

    # New: ByteTrack tracking
    def track(self, frame):
        """
        Run YOLOv8 + ByteTrack on a frame.
        classes=[0] restricts detection to 'person' only (COCO class 0).
        persist=True keeps the tracker state between calls.
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker='bytetrack.yaml',
            classes=[0],             # person only
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
