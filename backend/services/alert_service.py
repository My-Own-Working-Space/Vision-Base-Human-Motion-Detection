import logging
from datetime import datetime, timezone
from typing import List
from models.alert import Alert, AlertCreate
from storage.memory_store import MemoryStore

logger = logging.getLogger("backend.services.alert_service")


class AlertService:
    """
    Service layer responsible for business logic around alerts.
    """
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def create_alert(self, alert_create: AlertCreate) -> int:
        """
        Records the received timestamp, saves the alert to memory storage,
        and logs the creation with structured details.
        """
        received_at = datetime.now(timezone.utc)
        alert_id = self._store.add(alert_create, received_at)
        logger.info(
            "Alert created successfully",
            extra={
                "alertId": alert_id,
                "deviceId": alert_create.deviceId,
                "eventType": alert_create.eventType,
                "confidence": alert_create.confidence,
                "capturedAt": alert_create.capturedAt.isoformat(),
                "receivedAt": received_at.isoformat()
            }
        )
        return alert_id

    def get_all_alerts(self) -> List[Alert]:
        """
        Retrieves all stored alerts.
        """
        alerts = self._store.get_all()
        logger.debug("All alerts retrieved", extra={"count": len(alerts)})
        return alerts

    def get_alerts_count(self) -> int:
        """
        Retrieves the count of all stored alerts.
        """
        count = self._store.count()
        logger.debug("Alerts count retrieved", extra={"count": count})
        return count
