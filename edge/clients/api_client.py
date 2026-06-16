"""
ApiClient — Responsible only for HTTP communication with the Backend API.

No business logic. No threshold checking. No CSV persistence.
Pure transport layer.
"""

from __future__ import annotations

import requests

from edge.config import ApiConfig
from edge.logging_config import get_logger
from edge.models import AlertPayload

logger = get_logger(__name__)


class ApiClient:
    """
    HTTP client for transmitting alert payloads to the Backend Dashboard API.

    Sends multipart/form-data POST requests containing:
        - image: JPEG evidence file
        - class_name, confidence, timestamp, lat, lng: metadata fields
    """

    def __init__(self, api_config: ApiConfig) -> None:
        self._config = api_config

    @property
    def retry_delay(self) -> float:
        """Delay between retry attempts in seconds."""
        return self._config.retry_delay_seconds

    def send_alert(self, payload: AlertPayload) -> bool:
        """
        POST an alert payload to the Backend API.

        Args:
            payload: The alert to transmit.

        Returns:
            True if the API responded with a success status code.
        """
        url = self._config.detections_url

        try:
            with open(payload.image_path, "rb") as f:
                files = {"image": (payload.image_name, f, "image/jpeg")}
                data = {
                    "class_name": payload.class_name,
                    "confidence": f"{payload.confidence:.4f}",
                    "timestamp": payload.timestamp,
                    "lat": f"{payload.latitude:.6f}",
                    "lng": f"{payload.longitude:.6f}",
                }
                response = requests.post(
                    url,
                    files=files,
                    data=data,
                    timeout=self._config.timeout_seconds,
                )

            if response.status_code in (200, 201):
                logger.info("API upload success: %s → %d", url, response.status_code)
                return True
            else:
                logger.warning(
                    "API upload failed: %s → %d %s",
                    url, response.status_code, response.text[:200],
                )
                return False

        except requests.ConnectionError:
            logger.warning("API unreachable: %s", url)
            return False
        except requests.Timeout:
            logger.warning("API timeout: %s (>%ds)", url, self._config.timeout_seconds)
            return False
        except FileNotFoundError:
            logger.error("Evidence file not found: %s", payload.image_path)
            return False
        except Exception:
            logger.exception("Unexpected error during API upload to %s", url)
            return False

    def health_check(self) -> bool:
        """
        Test connectivity to the Backend API.

        Returns:
            True if the base URL is reachable.
        """
        try:
            response = requests.get(
                self._config.base_url,
                timeout=self._config.timeout_seconds,
            )
            return response.status_code < 500
        except Exception:
            return False
