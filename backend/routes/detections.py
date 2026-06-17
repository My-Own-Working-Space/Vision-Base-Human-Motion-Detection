"""
Detection receiver route — accepts multipart/form-data from the edge ApiClient.

This endpoint mirrors what the edge device actually sends:
    - image: JPEG evidence file
    - class_name, confidence, timestamp, lat, lng: metadata fields

Internally converts the multipart payload into the standard AlertCreate
format and stores it in the same shared memory store.
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, UploadFile, status
from routes.alerts import AlertCreateResponse, get_alert_service
from models.alert import AlertCreate
from services.alert_service import AlertService

logger = logging.getLogger("backend.routes.detections")

router = APIRouter(prefix="/api/detections", tags=["detections"])

# Directory to save received evidence images
_RECEIVED_DIR = os.path.join("alerts", "received")
os.makedirs(_RECEIVED_DIR, exist_ok=True)


@router.post(
    "",
    response_model=AlertCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Receive detection from edge device",
    description="Accepts multipart/form-data with image evidence and metadata from the Raspberry Pi edge client."
)
async def receive_detection(
    image: UploadFile = File(...),
    class_name: str = Form(...),
    confidence: float = Form(...),
    timestamp: str = Form(...),
    lat: float = Form(...),
    lng: float = Form(...),
    service: AlertService = Depends(get_alert_service),
) -> AlertCreateResponse:
    """
    POST /api/detections

    Converts the multipart edge payload into a standard AlertCreate
    and stores it alongside JSON-submitted alerts.
    """
    # Save evidence image
    save_path = os.path.join(_RECEIVED_DIR, image.filename)
    with open(save_path, "wb") as f:
        f.write(await image.read())

    logger.info(
        "Detection received: class=%s confidence=%.4f timestamp=%s",
        class_name, confidence, timestamp,
    )

    # Convert to AlertCreate format for unified storage
    alert_create = AlertCreate(
        deviceId="edge-device",
        eventType=class_name,
        confidence=min(confidence, 1.0),
        capturedAt=datetime.fromisoformat(timestamp) if timestamp else datetime.now(timezone.utc),
    )

    alert_id = service.create_alert(alert_create)
    return AlertCreateResponse(success=True, alertId=alert_id)
