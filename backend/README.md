# Edge AI Alert System Mock Backend

This is a lightweight, mock FastAPI backend designed to receive, validate, and store abnormal behavior alerts from Raspberry Pi devices during integration testing. It serves as a temporary substitute before migrating to the final ASP.NET Core backend.

## Architecture

The project has a clear, separated layered architecture:

- **`app.py`**: Entry point of the application. Handles initialization, logging configurations, routing setup, and global exception handlers.
- **`models/alert.py`**: Pydantic models for data validation and schema definitions (`AlertCreate` and `Alert`).
- **`routes/alerts.py`**: REST API endpoints mapping.
- **`services/alert_service.py`**: Business logic layer (computes timestamps, invokes logging, manages persistence calls).
- **`storage/memory_store.py`**: Thread-safe in-memory alert repository.

## Features

- **Pydantic Validation**: Automatic schema enforcement and range checks (e.g. confidence within `[0.0, 1.0]`).
- **Dependency Injection**: Decoupled routers and services via FastAPI's `Depends` mechanisms.
- **Structured JSON Logging**: Custom JSON log formatter writing single-line JSON log outputs for stdout parsing.
- **Validation Error Handling**: Intercepts `RequestValidationError` to log the incorrect payloads alongside the detailed errors.

---

## Getting Started

### 1. Install Dependencies

You can run this inside a virtual environment. Assuming you are in the `backend/` directory:

```bash
pip install -r requirements.txt
```

### 2. Start the Server

Run the following command from the `backend/` directory to start Uvicorn with auto-reload enabled:

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

---

## API Documentation

### 1. Health Check
* **Endpoint**: `GET /health`
* **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Submit Alert
* **Endpoint**: `POST /api/alerts`
* **Headers**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "deviceId": "raspberry-pi-01",
    "eventType": "AbnormalBehavior",
    "confidence": 0.92,
    "capturedAt": "2026-06-17T12:00:00Z"
  }
  ```
* **Response**:
  ```json
  {
    "success": true,
    "alertId": 1
  }
  ```

### 3. Get All Alerts
* **Endpoint**: `GET /api/alerts`
* **Response**:
  ```json
  [
    {
      "deviceId": "raspberry-pi-01",
      "eventType": "AbnormalBehavior",
      "confidence": 0.92,
      "capturedAt": "2026-06-17T12:00:00Z",
      "id": 1,
      "receivedAt": "2026-06-17T10:34:02.000Z"
    }
  ]
  ```

### 4. Get Alerts Count
* **Endpoint**: `GET /api/alerts/count`
* **Response**:
  ```json
  {
    "count": 1
  }
  ```
