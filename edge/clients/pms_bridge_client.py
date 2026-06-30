from __future__ import annotations

import requests

from edge.logging_config import get_logger
from edge.models import AlertPayload

logger = get_logger(__name__)

DEFAULT_PMS_URL = "http://localhost:5196"
DEFAULT_PMS_ENDPOINT = "/api/v1/vision/detections"


class PmsBridgeClient:

    def __init__(
        self,
        pms_base_url: str = DEFAULT_PMS_URL,
        pms_endpoint: str = DEFAULT_PMS_ENDPOINT,
        timeout_seconds: int = 5,
    ) -> None:
        self._base_url = pms_base_url.rstrip("/")
        self._endpoint = pms_endpoint
        self._timeout = timeout_seconds
        self._url = f"{self._base_url}{self._endpoint}"

    @property
    def pms_url(self) -> str:
        return self._url

    def register(self, serial_number: str, software_version: str) -> dict | None:
        register_url = f"{self._base_url}/api/v1/devices/register"
        try:
            payload = {
                "serialNumber": serial_number,
                "softwareVersion": software_version
            }
            response = requests.post(register_url, json=payload, timeout=self._timeout)
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            logger.warning(f"Registration failed: {e}")
            return None

    def send_heartbeat(self, drone_id: str, battery: float, temperature: float) -> bool:
        heartbeat_url = f"{self._base_url}/api/v1/devices/heartbeat"
        try:
            payload = {
                "droneId": drone_id,
                "battery": battery,
                "temperature": temperature
            }
            response = requests.post(heartbeat_url, json=payload, timeout=self._timeout)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Heartbeat failed: {e}")
            return False

    def send_detection(self, payload: AlertPayload, drone_id: str) -> bool:
        try:
            files = {}
            if payload.image_path:
                try:
                    files["image"] = (
                        payload.image_name,
                        open(payload.image_path, "rb"),
                        "image/jpeg",
                    )
                except FileNotFoundError:
                    logger.warning(
                        "PMS Bridge: evidence file not found: %s", payload.image_path
                    )

            data = {
                "drone_id": drone_id,
                "class_name": payload.class_name,
                "confidence": f"{payload.confidence:.4f}",
                "timestamp": payload.timestamp,
                "lat": f"{payload.latitude:.6f}",
                "lng": f"{payload.longitude:.6f}",
                "track_id": str(payload.track_id),
            }

            response = requests.post(
                self._url,
                files=files if files else None,
                data=data,
                timeout=self._timeout,
            )

            if "image" in files:
                files["image"][1].close()

            if response.status_code in (200, 201):
                logger.info(
                    "PMS Bridge: detection forwarded successfully → %d | %s",
                    response.status_code,
                    response.text[:200],
                )
                return True
            else:
                logger.warning(
                    "PMS Bridge: forward failed → %d %s",
                    response.status_code,
                    response.text[:200],
                )
                return False

        except requests.ConnectionError:
            logger.warning("PMS Bridge: PMS backend unreachable at %s", self._url)
            return False
        except requests.Timeout:
            logger.warning(
                "PMS Bridge: timeout connecting to %s (>%ds)",
                self._url,
                self._timeout,
            )
            return False
        except Exception:
            logger.exception("PMS Bridge: unexpected error forwarding to %s", self._url)
            return False

    def health_check(self) -> bool:
        health_url = f"{self._base_url}/api/v1/vision/health"
        try:
            response = requests.get(health_url, timeout=self._timeout)
            is_healthy = response.status_code == 200
            if is_healthy:
                logger.info("PMS Bridge: health check OK → %s", response.text[:100])
            else:
                logger.warning(
                    "PMS Bridge: health check failed → %d", response.status_code
                )
            return is_healthy
        except Exception:
            logger.warning("PMS Bridge: health check failed — PMS unreachable")
            return False
