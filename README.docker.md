# Docker Setup & Guide

This document describes how to run the Vision-Based Human Motion and Suspicious Behavior Detection system using Docker and Docker Compose.

---

## Prerequisites

- **Docker**: Installed and running on your system.
- **Docker Compose** (V2 recommended): Installed on your system.
- **Camera Source**: Either an IP camera (RTSP/HTTP link) or a local USB/integrated webcam.

---

## Project Services

The setup consists of two containerized services:
1. **vision-backend**: A lightweight FastAPI Mock Backend service running on port `8000` to receive anomaly detection alerts.
2. **vision-edge**: The AI processing container running on port `8001`. It runs YOLOv8 and behavior classifiers (MobileNetV3 / LSTM) on the camera input.

---

## Configuration (`.env`)

Before starting the containers, configure your settings in the local `.env` file (copied from `.env.example`). Docker Compose will automatically read and apply these settings.

Key settings to review:
- `IP_CAMERA_URL`: The video stream URL. If you are using a local webcam, set this to `0`.
- `TARGET_FPS`: Processing speed target (default is `8`).
- `ALERT_CONFIDENCE_THRESHOLD`: Threshold for registering an anomaly.

---

## How to Run

### 1. Build and Start the Containers

To build the images and run both services in the background:
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
- **Backend Health Endpoint**:
  - Verification endpoint: [http://localhost:8000/health](http://localhost:8000/health).

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
- **Model Weights**: The `models/` directory on your host is mapped to `/app/models` inside the container. You can update or replace the model weights (e.g. `mobilenetv3_ucf_crime.pt`) on your host, and the edge service will automatically use them without needing a container rebuild.

---

## Troubleshooting

### OpenCV Library Errors
If you run into missing library errors like `libGL.so.1` or `libgthread-2.0.so.0`, the OpenCV system requirements might have changed. The `Dockerfile.edge` is preconfigured to install these via:
```dockerfile
apt-get install -y libgl1 libglib2.0-0
```

### GPU Acceleration
This setup runs on the **CPU** by default (ideal for Edge/Raspberry Pi deployment). If you require GPU acceleration via CUDA inside the container, you will need to:
1. Ensure the host has Nvidia CUDA drivers and `nvidia-container-toolkit` installed.
2. Modify `Dockerfile.edge` to use a CUDA-compatible base image (e.g., `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`).
3. Add the `deploy` configurations for GPU in `docker-compose.yml`.
