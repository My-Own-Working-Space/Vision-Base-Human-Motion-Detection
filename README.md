# Vision-Based Human Motion and Suspicious Behavior Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)

An advanced, real-time AI surveillance system designed to detect and classify anomalous human behaviors (such as loitering, running, intruding, and falling). The system combines state-of-the-art spatial object tracking with real-time behavior classification to ensure highly accurate, production-grade security monitoring.

## Architecture Overview

The system follows a **clean event-driven architecture** where the Raspberry Pi (or any edge device) acts as an autonomous detection node:

```
Camera → CameraService → DetectionService → AlertService → ApiClient → Backend
              ↓                   ↓                ↓
         frame capture      DetectionEvent     state machine
         FPS throttle       (type, conf, ts)   confirmation & cooldown
         state machine                         CSV & local persistence
```

### Data Flow

1. **CameraService** captures frames from a camera/video source, applies FPS throttling, resizes to target width, and manages the `IS_PROCESSING ↔ SLEEPING` state machine.

2. **DetectionService** receives raw frames and runs the ML pipeline as a black box:
   - **YOLOv8 + ByteTrack** detects and tracks persons with persistent IDs.
   - **Classifier (Auto-Detected)**:
     - **MobileNetV3-Large**: Single-frame classifier model (`mobilenetv3_ucf_crime.pt`) classifying behavior on individual crops.
     - **ResNet18 + LSTM**: Sequence classifier using a sliding window buffer.
   - Returns a `DetectionEvent(event_type, confidence, timestamp)` — nothing more.

3. **AlertService** evaluates each `DetectionEvent` using a **Confirmation & Cooldown State Machine**:
   - **NORMAL**: Scans the camera stream.
   - **SUSPECTED**: Initiated upon finding anomaly frames.
   - **ANOMALY CONFIRMED**: Triggered only if `ALERT_CONFIRMATION_FRAMES` (default 3) consecutive frames classify as an anomaly. Gathers the highest confidence frame as evidence, logs to CSV, and dispatches to Backend via **ApiClient**.
   - **COOLDOWN**: Active after dispatching an alert to suppress duplicate alarms for `ALERT_COOLDOWN_SECONDS` (default 60s). Transitions back to `NORMAL` once elapsed.
   - Queues failed uploads for **automatic retry** (background thread).
   - Persists all alerts to CSV audit ledger `alerts/metadata.csv`.

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
│  │ • frame cap  │    │ • YOLO + Track   │    │ • confirmation SM │      │
│  │ • FPS ctrl   │    │ • Auto-Detect    │    │ • cooldown timer  │      │
│  │ • state mach │    │   Classifier     │    │ • CSV persistence │      │
│  │ • resize     │    │                  │    │ • retry queue     │      │
│  └──────────────┘    │  Returns:        │    └────────┬──────────┘      │
│         │            │  DetectionEvent  │             │                  │
│         │            │  (type,conf,ts)  │             │                  │
│         │            └──────────────────┘             │                  │
│         │                                             │                  │
│         │                                             ▼                  │
│         ▼                                    ┌──────────────────┐         │
│  ┌──────────────┐                            │                  │         │
│  │              │                            │    ApiClient     │────────▶│ Backend API
│  │StreamRenderer│                            │                  │         │
│  │ • annotate   │                            │ • POST multipart │         │
│  │ • MJPEG enc  │                            │ • health check   │         │
│  │ • HUD        │                            │ • timeout/retry  │         │
│  └──────────────┘                            └──────────────────┘         │
│         │                                                                 │
│         ▼                                                                 │
│    FastAPI /video                                                         │
│    (MJPEG Stream)                                                         │
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
backend/                           # FastAPI Mock Backend Service
├── app.py                         # Unified structured logger & router entrypoint
├── routes/
│   ├── alerts.py                  # Standard alert listing & count endpoints
│   └── detections.py              # Multipart receiver endpoint for edge uploads
├── services/                      # Backend storage & aggregation logic
└── storage/                       # Thread-safe in-memory cache
edge/                              # Production edge runtime
├── main.py                        # Entry point — wires services via DI
├── config.py                      # Configuration (env vars → frozen dataclasses)
├── models.py                      # Domain models (DetectionEvent, AlertPayload)
├── logging_config.py              # Structured logging setup
├── services/
│   ├── camera_service.py          # Frame capture, FPS throttle, state machine
│   ├── detection_service.py       # YOLO + Classifiers → DetectionEvent
│   └── alert_service.py           # Confirmation + cooldown state machine, CSV
├── clients/
│   └── api_client.py              # Pure HTTP transport to Backend
└── ui/
    └── stream_renderer.py         # FastAPI MJPEG stream (presentation only)
inference/                         # ML models — treated as black box
├── yolo_detector.py               # YOLOv8 + ByteTrack tracking
├── mobilenetv3_classifier.py      # MobileNetV3 single-frame classifier (timm)
├── resnet_lstm_classifier.py      # ResNet18 + LSTM temporal classifier
└── bytetrack_custom.yaml          # ByteTrack configuration
models/                            # Pre-trained model weights (.pt, .pth)
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

### 3. Run the Services

**Mock Backend Service (Terminal 1)**:
```bash
./.venv/bin/uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

**Edge Client Device (Terminal 2)**:
```bash
./.venv/bin/python edge/main.py
```

The annotated AI video stream will be available at: **http://localhost:8001/video**
The Edge Client Dashboard is at: **http://localhost:8001**

### App Factory

`edge.main` exposes `create_app(...)` and `compose_services(...)`. Importing `edge.main:app` creates an ASGI app only; model loading, alert retry workers, and PMS heartbeat workers start during FastAPI lifespan startup. Tests can inject fake services with `initialize_models=False` and `start_background_workers=False`.

### 4. API Endpoints (Edge Client)

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | System status, configurations, and Dashboard interface |
| `GET` | `/video` | Live annotated MJPEG video stream |
| `GET` | `/api/alerts-history` | Fetch local confirmed anomalies list from CSV logs |
| `POST` | `/api/analyze` | Analyze one uploaded image or video and return detection JSON |
| `POST` | `/api/analyze-batch` | Analyze multiple uploaded images/videos and return per-file detection JSON |

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `IP_CAMERA_URL` | `0` | Camera source (0=webcam, RTSP URL, or HTTP camera URL) |
| `TARGET_FPS` | `8` | Target processing FPS |
| `LSTM_MODEL_PATH` | `models/mobilenetv3_ucf_crime.pt` | Classifier weights (auto-detects architecture) |
| `PROCESSING_DURATION` | `999999.0` | Force continuous processing mode |
| `SLEEP_DURATION` | `0.0` | rest period (seconds) |
| `ALERT_CONFIDENCE_THRESHOLD` | `0.70` | Minimum confidence to register anomaly |
| `ALERT_CONFIRMATION_FRAMES` | `3` | Consecutive anomaly frames to trigger dispatch |
| `ALERT_COOLDOWN_SECONDS` | `60.0` | Mute duplicate alerts after confirmation |
| `DASHBOARD_API_URL` | `http://localhost:8000` | Backend API URL |

## Harness Engineering Runtime

This repo now includes a provider-independent harness architecture alongside the existing edge camera service. It is designed around explicit runtime, loop, context, prompt, tool, provider, memory, and observability boundaries. The first vertical slice runs offline with a deterministic fake vision provider; it does not require Roboflow, internet access, MCP, camera access, or model weights.

Run the local fake-provider harness on a representative alert image:

```bash
./.venv/bin/python -m edge.cli.run_fake_harness alerts/alert_track_2_20260526_172904_540897.jpg
```

Run the offline harness tests:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Harness checkpoints and traces are written under `harness_runs/`, which is ignored by git. Fake-provider results include `FAKE_RESULT_DO_NOT_TREAT_AS_MODEL_INFERENCE` and must not be interpreted as real model output.

Architecture docs:

- `docs/harness-architecture-audit.md`
- `docs/harness-refactor-plan.md`
- `docs/roboflow-integration-status.md`

## Roboflow Workflow Integration

This repo includes a focused client for the published Roboflow Workflow:

- Workspace: `les-workspace-ijdwd`
- Workflow: `evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`
- Endpoint: `https://serverless.roboflow.com/les-workspace-ijdwd/workflows/evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`

The workflow definition was grounded through Roboflow MCP on 2026-07-19. It declares one image input named `image`, no runtime parameters, and one JSON output named `predictions`.

Configure `.env` with:

```bash
ROBOFLOW_API_KEY=your_private_api_key
ROBOFLOW_WORKFLOW_ENABLED=true
```

Run the smoke test:

```bash
./.venv/bin/python smoke_roboflow_workflow.py
```

Use the client from Python:

```python
from edge.clients.roboflow_workflow_client import RoboflowWorkflowClient
from edge.config import load_config

client = RoboflowWorkflowClient(load_config().roboflow_workflow)
result = client.run_evn_object_detection("https://example.com/image.jpg")
predictions = result.outputs["predictions"]
```

Image-shaped workflow outputs are decoded to `ROBOFLOW_IMAGE_OUTPUT_DIRECTORY` and are not logged. The current published workflow returned a Roboflow server-side configuration error during MCP verification on 2026-07-19: the inner workflow step binds `model_id`, but the child workflow does not declare that input. Republish or fix the workflow in Roboflow before expecting the smoke test to pass.

## State Machine (Alert Confirmation Flow)

```
       [NORMAL]
          │
  Anomaly detected
          │
          ▼
     [SUSPECTED] ◀── Normal frame breaks streak ──┐
          │                                      │
Consecutive count ≥ threshold                    │
          │                                      │
          ▼                                      │
  [ANOMALY CONFIRMED]                            │
          │                                      │
   1. Save evidence                              │
   2. Send POST /api/detections                  │
   3. Log alerts/metadata.csv                    │
          │                                      │
          ▼                                      │
      [COOLDOWN]                                 │
          │                                      │
    Timer expires ───────────────────────────────┘
```

## Dataset

Trained on the **UCF Crime** & **ShanghaiTech Campus** datasets for human-motion anomaly behavior categorization and abnormal incident alerting.

> Liu, W., Luo, W., Lian, D., & Gao, S. (2018). *Future Frame Prediction for Anomaly Detection — A New Baseline*. CVPR 2018.