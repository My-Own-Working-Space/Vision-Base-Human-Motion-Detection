import threading
from datetime import datetime
from typing import List
from models.alert import Alert, AlertCreate


class MemoryStore:
    """
    Thread-safe, in-memory repository for storing and querying alerts.
    """
    def __init__(self) -> None:
        self._alerts: List[Alert] = []
        self._next_id: int = 1
        self._lock = threading.Lock()

    def add(self, alert_create: AlertCreate, received_at: datetime) -> int:
        """
        Creates a new Alert from AlertCreate, assigns an ID, stores it,
        and returns the generated ID.
        """
        with self._lock:
            alert_id = self._next_id
            alert = Alert(
                id=alert_id,
                deviceId=alert_create.deviceId,
                eventType=alert_create.eventType,
                confidence=alert_create.confidence,
                capturedAt=alert_create.capturedAt,
                receivedAt=received_at
            )
            self._alerts.append(alert)
            self._next_id += 1
            return alert_id

    def get_all(self) -> List[Alert]:
        """
        Returns a list of all received alerts.
        """
        with self._lock:
            # Return a shallow copy of the list to prevent external modification
            return list(self._alerts)

    def count(self) -> int:
        """
        Returns the total number of alerts received.
        """
        with self._lock:
            return len(self._alerts)
