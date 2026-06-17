from datetime import datetime
from pydantic import BaseModel, Field


class AlertCreate(BaseModel):
    deviceId: str = Field(
        ..., 
        description="The unique identifier of the Raspberry Pi device sending the alert"
    )
    eventType: str = Field(
        ..., 
        description="The type of event detected, e.g., AbnormalBehavior"
    )
    confidence: float = Field(
        ..., 
        description="The detection confidence score (between 0.0 and 1.0)",
        ge=0.0,
        le=1.0
    )
    capturedAt: datetime = Field(
        ..., 
        description="The ISO 8601 timestamp when the event was captured on the device"
    )


class Alert(AlertCreate):
    id: int = Field(
        ..., 
        description="The auto-incremented unique ID of the alert in the backend storage"
    )
    receivedAt: datetime = Field(
        ..., 
        description="The ISO 8601 timestamp when the alert was received by the backend"
    )
