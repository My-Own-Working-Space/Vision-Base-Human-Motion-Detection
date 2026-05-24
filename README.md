# Vision-Based Human Motion and Suspicious Behavior Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)

An advanced, real-time AI surveillance system designed to detect and classify anomalous human behaviors (such as loitering, running, intruding, and falling). The system combines state-of-the-art spatial object tracking with temporal sequence analysis to ensure highly accurate, production-grade security monitoring.

## Key Features

- **Multi-Object Tracking:** Utilizes **YOLOv8 + ByteTrack** to detect humans and maintain persistent track IDs across frames.
- **Temporal Sequence Analysis:** Replaces static frame classification with a dynamic **ResNet18 + LSTM** architecture. It analyzes a sliding window of 16 frames per individual to understand the temporal context of human motion.
- **State Machine Engine:** Intelligent `IS_PROCESSING` ↔ `SLEEPING` state machine with a lightweight **motion watchdog** that detects sudden movement via frame differencing and wakes the AI pipeline instantly — dramatically reducing CPU usage while maintaining responsiveness.
- **Adaptive Thresholding:** Dynamic watchdog threshold ($T_{motion}^{(t)} = \alpha \cdot \bar{B}_t + \beta$) adapts to wind, shadows, and environment lighting to prevent false-triggers.
- **Feature-Level Interpolation:** Keeps LSTM sequences continuous by duplicating the last known feature vector if a track is temporarily lost (up to 3 frames), preventing abrupt buffer resets.
- **Hardware-Aware Loading:** Automatically selects the optimal model format: TensorRT INT8/FP16 engines on embedded devices (Jetson, RPi) with CUDA, or standard `.pt` on server/desktop.
- **FPS Throttling (Server Mode):** Automatically adjusts frame read rate to match `TARGET_FPS`, preventing CPU overload on high-FPS streams.
- **Real-Time Video Streaming:** Features a high-performance **FastAPI** backend that annotates frames (bounding boxes, track IDs, anomaly warnings) and serves them via an MJPEG stream.
- **IP Camera Ready:** Seamlessly integrates with physical webcams, RTSP IP Cameras, or pre-recorded video files via environment variables.

## Repository Structure

- `/training`: Source code for training the ResNet+LSTM temporal models on temporal datasets (e.g., ShanghaiTech).
- `/inference`: The core FastAPI application that bridges YOLO+ByteTrack tracking with the sliding-window LSTM sequence classifier.
- `/models`: Storage for pre-trained weights (`.pt`, `.pth`). *(Ignored in git to save space)*.
- `/integration`: Utility scripts (e.g., `data_bridge.py`) for simulating and pushing alerts.
- `api.http`: VS Code REST Client test script for API endpoints.

## Getting Started

### 1. Setup Python Environment
Create a clean virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn opencv-python ultralytics torch torchvision pillow python-multipart matplotlib
```

### 2. Configure Environment Variables
Copy or edit the `.env` file at the project root:
```env
# Device mode: "embedded" (Jetson/RPi with TensorRT) or "server" (desktop/cloud)
DEVICE_MODE=server

# State machine timing (seconds)
PROCESSING_DURATION=2.0    # AI active burst duration
SLEEP_DURATION=3.0         # Watchdog-only rest period
MOTION_THRESHOLD=5000      # Base pixel diff threshold to wake from SLEEPING

# FPS control (server mode only)
TARGET_FPS=8               # Target processing FPS
LSTM_TRAIN_FPS=30          # Reference FPS the LSTM was trained at

# Camera source
IP_CAMERA_URL=0            # 0 = webcam, or IP camera URL / video file path
```

### 3. Run the Real-Time AI Inference Stream
Navigate to the inference directory and start the FastAPI server:
```bash
python inference/app.py
```
The annotated AI video stream will be available at: **http://localhost:8000/video**

### 4. Test API Endpoints
You can use the provided [api.http](api.http) file in VS Code or run quick HTTP queries using curl:
```bash
# Query server status
curl http://localhost:8000/
```

## Architecture Overview

```
Camera Stream → [State Machine] → IS_PROCESSING / SLEEPING
                                       │
                 ┌─────────────────────┘
                 ▼
         IS_PROCESSING:
         ├── YOLOv8 + ByteTrack (person detection + tracking)
         ├── OpenCV crop → PIL per tracked person
         ├── ResNet18 feature extraction (512-dim)
         ├── LSTM temporal classifier (16-frame window)
         └── Binary Classification: Normal / Anomaly
                 │
                 ▼
         SLEEPING:
         ├── Frame differencing watchdog (no AI)
         └── Wake on motion > Adaptive Threshold
                 │
                 ▼
         FastAPI MJPEG Stream → Browser / Client
```

### Pipeline Stages

1. **YOLOv8 + ByteTrack:** Detects humans and assigns unique `track_id`s with custom ByteTrack config (120-frame track buffer for stable IDs).
2. **Crop & Buffer:** Extracts spatial crops for each person and maintains an independent 16-frame queue with a 30-frame grace period.
3. **ResNet18 Backbone:** Extracts 512-dimensional feature vectors per frame.
4. **LSTM Classifier:** Analyzes the sequence of 16 feature vectors to classify the behavior as `Normal` or `Anomaly`.
5. **State Machine:** Alternates between full AI processing and lightweight watchdog mode to optimize CPU usage.
6. **FastAPI MJPEG Stream:** Renders colored bounding boxes (Green=Normal, Red=Anomaly, Cyan=Buffering) with high-contrast auto-text in real-time.

### State Machine

| State | What Runs | CPU Load | Duration |
|---|---|---|---|
| `IS_PROCESSING` | Full YOLO → ByteTrack → ResNet18 → LSTM | **High** | `PROCESSING_DURATION` (default 2s) |
| `SLEEPING` | Frame differencing watchdog only | **Minimal** | `SLEEP_DURATION` (default 3s) or until motion detected |

### Dataset

Trained on the **ShanghaiTech Campus** dataset — 13 scenes, 317,398 frames, 130 abnormal events including chasing, brawling, and sudden motion anomalies with pixel-level annotations.

> Liu, W., Luo, W., Lian, D., & Gao, S. (2018). *Future Frame Prediction for Anomaly Detection — A New Baseline*. CVPR 2018.