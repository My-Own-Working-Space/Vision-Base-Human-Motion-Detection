Camera/Video Stream
        ↓
  YOLOv8 (person detection) + ByteTrack (multi-object tracking)
        ↓
  OpenCV crop → PIL image per tracked person
        ↓
  Sliding Window Sequence Builder  (16 frames per track_id)
        ↓
  ResNet18 Feature Extractor       (512-dim spatial features)
        ↓
  Bi-GRU + Temporal Attention      (analyzes motion over time)
        ↓
  Binary Classification: Normal / Anomaly
        ↓
  Visualization & MJPEG Stream → FastAPI /video