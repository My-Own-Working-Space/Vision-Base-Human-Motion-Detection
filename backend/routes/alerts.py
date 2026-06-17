import logging
from typing import List
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from models.alert import Alert, AlertCreate
from services.alert_service import AlertService
from storage.memory_store import MemoryStore

logger = logging.getLogger("backend.routes.alerts")

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

# Single shared memory store instance for the application
_store = MemoryStore()


def get_alert_service() -> AlertService:
    """
    Dependency injection provider for the AlertService.
    """
    return AlertService(_store)


class AlertCreateResponse(BaseModel):
    success: bool = Field(..., description="Indicates whether the alert was successfully processed")
    alertId: int = Field(..., description="The unique ID of the created alert")


class AlertCountResponse(BaseModel):
    count: int = Field(..., description="The total number of alerts received")


@router.post(
    "",
    response_model=AlertCreateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new alert",
    description="Receives an alert from the edge device, validates it, and stores it in memory."
)
def create_alert(
    alert_create: AlertCreate,
    service: AlertService = Depends(get_alert_service)
) -> AlertCreateResponse:
    """
    POST /api/alerts
    """
    alert_id = service.create_alert(alert_create)
    return AlertCreateResponse(success=True, alertId=alert_id)


@router.get(
    "",
    response_model=List[Alert],
    summary="Get all alerts",
    description="Retrieves a list of all historical alerts stored in memory."
)
def get_alerts(
    service: AlertService = Depends(get_alert_service)
) -> List[Alert]:
    """
    GET /api/alerts
    """
    return service.get_all_alerts()


@router.get(
    "/count",
    response_model=AlertCountResponse,
    summary="Get alert count",
    description="Retrieves the total number of alerts received by this mock service."
)
def get_alerts_count(
    service: AlertService = Depends(get_alert_service)
) -> AlertCountResponse:
    """
    GET /api/alerts/count
    """
    count = service.get_alerts_count()
    return AlertCountResponse(count=count)
