# Vision-Based Human Motion and Suspicious Behavior Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)

An advanced, real-time AI surveillance system designed to detect and classify anomalous human behaviors (such as loitering, running, intruding, and falling). The system combines state-of-the-art spatial object tracking with temporal sequence analysis to ensure highly accurate, production-grade security monitoring.

## Architecture Overview

The system follows a **clean event-driven architecture** where the Raspberry Pi (or any edge device) acts as an autonomous detection node:

```
Camera → CameraService → DetectionService → AlertService → ApiClient → Backend
              ↓                   ↓                ↓
         frame capture      DetectionEvent     threshold check
         FPS throttle       (type, conf, ts)   retry queue
         state machine                         CSV persistence
```

### Data Flow

1. **CameraService** captures frames from a camera/video source, applies FPS throttling, resizes to target width, and manages the `IS_PROCESSING ↔ SLEEPING` state machine.

2. **DetectionService** receives raw frames and runs the ML pipeline as a black box:
   - **YOLOv8 + ByteTrack** detects and tracks persons with persistent IDs
   - **ResNet18** extracts 512-dimensional spatial features per person crop
   - **LSTM** classifies the temporal sequence (16-frame sliding window) as Normal or Anomaly
   - Returns a `DetectionEvent(event_type, confidence, timestamp)` — nothing more

3. **AlertService** evaluates each `DetectionEvent`:
   - Checks if `confidence ≥ threshold` (configurable, default 0.70)
   - Throttles to one alert per track ID
   - Saves evidence image to local `alerts/` directory
   - Dispatches to Backend via **ApiClient**
   - Queues failed uploads for **automatic retry** (background thread)
   - Persists all alerts to CSV audit ledger (regardless of upload status)

4. **ApiClient** is a pure HTTP transport layer — sends multipart/form-data POST to the Backend API. No business logic.

5. **StreamRenderer** annotates frames with bounding boxes, labels, and HUD — then encodes as MJPEG for browser viewing. Zero business logic.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Edge Device (Raspberry Pi)                      │
│                                                                         │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────────┐      │
│  │              │    │                  │    │                   │      │
│  │ CameraService│───▶│ DetectionService │───▶│  AlertService     │      │
│  │              │    │                  │    │                   │      │
│  │ • frame cap  │    │ • YOLO + Track   │    │ • threshold check │      │
│  │ • FPS ctrl   │    │ • ResNet18 feat  │    │ • CSV persistence │      │
│  │ • state mach │    │ • LSTM classify  │    │ • retry queue     │      │
│  │ • resize     │    │                  │    │                   │      │
│  └──────────────┘    │  Returns:        │    └────────┬──────────┘      │
│         │            │  DetectionEvent  │             │                  │
│         │            │  (type,conf,ts)  │             │                  │
│         │            └──────────────────┘             │                  │
│         │                                             │                  │
│         ▼                                             ▼                  │
│  ┌──────────────┐                          ┌──────────────────┐         │
│  │              │                          │                  │         │
│  │StreamRenderer│                          │    ApiClient     │────────▶│ Backend API
│  │              │                          │                  │         │
│  │ • annotate   │                          │ • POST multipart │         │
│  │ • MJPEG enc  │                          │ • health check   │         │
│  │ • HUD        │                          │ • timeout/retry  │         │
│  └──────────────┘                          └──────────────────┘         │
│         │                                                               │
│         ▼                                                               │
│    FastAPI /video                                                       │
│    (MJPEG Stream)                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Design Principles

| Principle | Implementation |
|---|---|
| **Single Responsibility** | Each service has one job: CameraService captures, DetectionService infers, AlertService alerts |
| **Separation of Concerns** | ML inference knows nothing about HTTP/CSV; UI code has no business logic |
| **Dependency Injection** | All services receive their dependencies via constructor; `main.py` is the Composition Root |
| **Dependency Inversion** | Services depend on Configuration abstractions, not concrete env-var reads |
| **No Hardcoded Values** | Every URL, threshold, API key, and device ID comes from `.env` configuration |
| **Structured Logging** | All modules use `logging.getLogger(__name__)` — no `print()` statements |

## Repository Structure

```
edge/                              # Production edge runtime
├── main.py                        # Entry point — wires services via DI
├── config.py                      # Configuration (env vars → frozen dataclasses)
├── models.py                      # Domain models (DetectionEvent, AlertPayload)
├── logging_config.py              # Structured logging setup
├── services/
│   ├── camera_service.py          # Frame capture, FPS throttle, state machine
│   ├── detection_service.py       # YOLO + LSTM → DetectionEvent
│   └── alert_service.py           # Threshold, retry queue, CSV persistence
├── clients/
│   └── api_client.py              # Pure HTTP transport to Backend
└── ui/
    └── stream_renderer.py         # FastAPI MJPEG stream (presentation only)
inference/                         # ML models — treated as black box
├── yolo_detector.py               # YOLOv8 + ByteTrack tracking
├── resnet_lstm_classifier.py      # ResNet18 + LSTM temporal classifier
└── bytetrack_custom.yaml          # ByteTrack configuration
training/                          # Training scripts (not part of edge runtime)
models/                            # Pre-trained model weights (.pt, .pth)
tests/                             # Test suites
```

## Getting Started

### 1. Setup Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your camera source, API URL, coordinates, etc.
```

All configuration is loaded from environment variables — see `.env.example` for the full list.

### 3. Run the Edge Device

```bash
# New event-driven architecture
python edge/main.py

# Legacy monolithic app (still functional but deprecated)
python inference/app.py
```

The annotated AI video stream will be available at: **http://localhost:8000/video**

### 4. API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | System status and configuration |
| `GET` | `/video` | Live annotated MJPEG video stream |

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `IP_CAMERA_URL` | `0` | Camera source (0=webcam, RTSP URL, or file path) |
| `TARGET_FPS` | `8` | Target processing FPS |
| `DEVICE_MODE` | `server` | `embedded` (Jetson/RPi) or `server` (desktop/cloud) |
| `PROCESSING_DURATION` | `2.0` | AI active burst duration (seconds) |
| `SLEEP_DURATION` | `3.0` | Watchdog-only rest period (seconds) |
| `MOTION_THRESHOLD` | `5000` | Base pixel diff threshold for motion |
| `ALERT_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence to trigger alert |
| `DASHBOARD_API_URL` | `http://localhost:5000` | Backend API base URL |
| `DETECTIONS_ENDPOINT` | `/api/detections` | Backend detections endpoint |
| `API_MAX_RETRIES` | `3` | Max retry attempts for failed uploads |
| `LATITUDE` / `LONGITUDE` | `10.7769` / `106.7009` | Device GPS coordinates |

## State Machine

| State | What Runs | CPU Load | Duration |
|---|---|---|---|
| `IS_PROCESSING` | Full YOLO → ByteTrack → ResNet18 → LSTM | **High** | `PROCESSING_DURATION` (default 2s) |
| `SLEEPING` | Frame differencing watchdog only | **Minimal** | `SLEEP_DURATION` (default 3s) or until motion detected |

## Alert Pipeline

```
[LSTM Inference] → DetectionEvent (type=ANOMALY, conf > threshold)
                        │
                        ▼
           [AlertService.evaluate()]
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
  [Save Evidence Image]       [ApiClient.send_alert()]
  alerts/*.jpg                   POST /api/detections
         │                             │
         ▼                     ┌───────┴───────┐
  [CSV Audit Ledger]           ▼               ▼
  alerts/metadata.csv     Success          Failure
                          (Uploaded)     → Retry Queue
                                          (background)
```

## Dataset

Trained on the **ShanghaiTech Campus** dataset — 13 scenes, 317,398 frames, 130 abnormal events including chasing, brawling, and sudden motion anomalies with pixel-level annotations.

> Liu, W., Luo, W., Lian, D., & Gao, S. (2018). *Future Frame Prediction for Anomaly Detection — A New Baseline*. CVPR 2018.