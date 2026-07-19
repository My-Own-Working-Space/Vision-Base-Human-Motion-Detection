# Docker Setup & Guide

This document describes how to run the Vision-Based Human Motion and Suspicious Behavior Detection system using Docker and Docker Compose.

---

## Prerequisites

- **Docker**: Installed and running on your system.
- **Docker Compose** (V2 recommended): Installed on your system.
- **Camera Source**: Either an IP camera (RTSP/HTTP link) or a local USB/integrated webcam.

---

## Project Services

The Docker Compose setup currently defines one deployable service:

1. **vision-edge**: The AI processing container running on port `8001`. It runs YOLOv8 and behavior classifiers (MobileNetV3 / LSTM), serves the dashboard/video stream, and exposes `/api/analyze` plus `/api/analyze-batch`.

If you use a separate dashboard/backend service, configure `DASHBOARD_API_URL` in `.env` and set `DASHBOARD_API_ENABLED=true`.

---

## Configuration (`.env`)

Before starting the containers, configure your settings in the local `.env` file (copied from `.env.example`). Docker Compose will automatically read and apply these settings.

Key settings to review:

- `IP_CAMERA_URL`: The video stream URL. Use `0` for a local webcam, an RTSP URL for an IP camera, or a video file path mounted into the container.
- `TARGET_FPS`: Processing speed target, default `8`.
- `ALERT_CONFIDENCE_THRESHOLD`: Threshold for registering an anomaly.
- `DEVICE_MODE`: Use `server` for the default CPU deploy. The current code path does not require GPU.
- `DASHBOARD_API_ENABLED`: Set `true` only when a backend receiver is reachable. Default in compose is `false` for standalone deploy.
- `SERVER_PORT`: Compose publishes container port `8001` to host port `8001`.

---

## How to Run

### 1. Build and Start the Containers

To build the image and run the edge service in the background:
```bash
docker compose up -d --build
```

### 2. Verify Running Services

Check the container status:
```bash
docker compose ps
```

To view live container logs:
```bash
docker compose logs -f
```

### 3. Access the Dashboards

- **Edge Client Dashboard & Live Video Stream**:
  - Open [http://localhost:8001](http://localhost:8001) in your browser.
  - View the live annotated stream directly at [http://localhost:8001/video](http://localhost:8001/video).
- **Analyze API**:
  - Single image/video: `POST http://localhost:8001/api/analyze`
  - Multiple images/videos: `POST http://localhost:8001/api/analyze-batch`
- **Healthcheck**:
  - Docker Compose checks `GET http://127.0.0.1:8001/` inside the container.

---

## USB/Integrated Webcam Setup (Optional)

If you configure `IP_CAMERA_URL=0` in `.env` to capture from a physical USB or built-in webcam, Docker needs access to the host's video device.

1. Open `docker-compose.yml`.
2. Uncomment the `devices` section under the `edge` service:
   ```yaml
   devices:
     - "/dev/video0:/dev/video0"
   ```
3. Restart the containers:
   ```bash
   docker compose down && docker compose up -d
   ```

---

## Volumes & Persistence

- **Alerts and Evidence**: The `alerts/` folder on your host machine is mapped to `/app/alerts` inside the container. This ensures that:
  - All captured evidence frames (`.jpg`) are saved directly on the host machine.
  - The CSV audit ledger (`alerts/metadata.csv`) is preserved across container restarts.
- **Model Weights**: The `models/` directory on your host is mapped read-only to `/app/models` inside the container. You can update or replace model weights on the host and restart the container.
- **Harness Runs**: `harness_runs/` is mapped to `/app/harness_runs` for local harness checkpoints.
- **Roboflow Outputs**: `roboflow_outputs/` is mapped to `/app/roboflow_outputs` for decoded workflow image outputs if the Roboflow client is used.

---

## Troubleshooting

### OpenCV Library Errors
If you run into missing library errors like `libGL.so.1` or `libgthread-2.0.so.0`, the OpenCV system requirements might have changed. The `Dockerfile.edge` is preconfigured to install these via:
```dockerfile
apt-get install -y libgl1 libglib2.0-0
```

### GPU Acceleration

GPU is **not required** for the current deploy. `Dockerfile.edge` intentionally installs CPU PyTorch wheels, and the current runtime works on CPU.

Important implementation detail: the current code only attempts CUDA in the YOLO wrapper when `DEVICE_MODE=embedded` and `torch.cuda.is_available()`. The behavior classifiers currently default to CPU. So simply adding NVIDIA Docker runtime is not enough for full GPU acceleration; the model-device plumbing should be refactored first if you want a real GPU deploy.

Recommended default:

```bash
DEVICE_MODE=server
docker compose up -d --build
```

For future GPU work:

1. Add explicit device config for YOLO and classifiers.
2. Switch Docker to CUDA-compatible PyTorch wheels/base image.
3. Add NVIDIA runtime settings to Compose.
4. Re-test `/api/analyze`, `/api/analyze-batch`, and `/video` under GPU.
