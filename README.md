# Vision-Based Human Motion and Suspicious Behavior Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)
[![ASP.NET Core](https://img.shields.io/badge/ASP.NET%20Core-9.0-512BD4.svg?logo=dotnet)](https://dotnet.microsoft.com/)

An advanced, real-time AI surveillance system designed to detect and classify anomalous human behaviors (such as loitering, running, intruding, and falling). The system combines state-of-the-art spatial object tracking with temporal sequence analysis to ensure highly accurate, production-grade security monitoring.

## ✨ Key Features

- **Multi-Object Tracking:** Utilizes **YOLOv8 + ByteTrack** to detect humans and maintain persistent track IDs across frames.
- **Temporal Sequence Analysis:** Replaces static frame classification with a dynamic **ResNet18 + LSTM** architecture. It analyzes a sliding window of 16 frames per individual to understand the temporal context of human motion.
- **Real-Time Video Streaming:** Features a high-performance **FastAPI** backend that annotates frames (bounding boxes, track IDs, anomaly warnings) and serves them via an MJPEG stream.
- **IP Camera Ready:** Seamlessly integrates with physical webcams, RTSP IP Cameras, or pre-recorded video files via environment variables.
- **Security Dashboard:** Includes an elegant **ASP.NET Core 9** web dashboard for visualizing alerts, camera mapping, and security activity logs.

## 📁 Repository Structure

- `/training`: Source code for training the ResNet+LSTM temporal models on temporal datasets (e.g., ShanghaiTech).
- `/inference`: The core FastAPI application that bridges YOLO+ByteTrack tracking with the sliding-window LSTM sequence classifier.
- `/models`: Storage for pre-trained weights (`.pt`, `.pth`). *(Ignored in git to save space)*.
- `/web`: The ASP.NET Core 9 security management dashboard.
- `/integration`: Utility scripts (e.g., `data_bridge.py`) for simulating and pushing alerts to the dashboard.

## 🚀 Getting Started

### 1. Setup Python Environment
Create a clean virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn opencv-python ultralytics torch torchvision pillow python-multipart
```

### 2. Run the Real-Time AI Inference Stream
Navigate to the inference directory and start the FastAPI server:
```bash
cd inference
# Optional: Set a custom video file or RTSP stream (defaults to webcam 0)
export IP_CAMERA_URL="/path/to/test_video.mp4" 
python app.py
```
The annotated AI video stream will be available at: **http://localhost:8000/video**

### 3. Run the Security Web Dashboard
Open a new terminal, navigate to the web directory, and launch the ASP.NET Core dashboard:
```bash
cd web/HumanMotionDetection.Web
dotnet run
```
Access the dashboard at: **http://localhost:5004**

## 🧠 Architecture Overview

1. **YOLOv8 + ByteTrack:** Detects humans and assigns unique `track_id`s.
2. **Crop & Buffer:** Extracts spatial crops for each person and maintains an independent 16-frame queue.
3. **ResNet18 Backbone:** Extracts 512-dimensional feature vectors per frame.
4. **LSTM / Bi-GRU Classifier:** Analyzes the sequence of 16 feature vectors to classify the behavior as `Normal` or `Anomaly`.
5. **FastAPI MJPEG Stream:** Renders colored bounding boxes (Green=Normal, Red=Anomaly) in real-time.