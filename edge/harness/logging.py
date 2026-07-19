from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import Any

from edge.harness.events import HarnessEvent

SECRET_MARKERS = ("api_key", "apikey", "token", "secret", "password", "authorization")
REDACTED = "[REDACTED]"


def redact(value: Any) -> Any:
    if isinstance(value, Mapping):
        redacted = {}
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in SECRET_MARKERS):
                redacted[key] = REDACTED
            else:
                redacted[key] = redact(item)
        return redacted
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact(item) for item in value)
    return value


class StructuredEventLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("edge.harness")
        self.events: list[dict[str, Any]] = []

    def emit(self, event: HarnessEvent) -> None:
        data = redact(event.to_dict())
        self.events.append(data)
        self._logger.info(json.dumps(data, sort_keys=True, default=str))
