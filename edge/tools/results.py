from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(str, Enum):
    SUCCESS = "success"
    RETRYABLE_FAILURE = "retryable_failure"
    PERMANENT_FAILURE = "permanent_failure"
    INVALID_INPUT = "invalid_input"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ToolResult:
    status: ToolStatus
    output: Any = None
    error_message: str | None = None
    error_category: str | None = None
    attempts: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        return self.status == ToolStatus.SUCCESS

    @staticmethod
    def success(output: Any, attempts: int = 1, metadata: dict[str, Any] | None = None) -> "ToolResult":
        return ToolResult(ToolStatus.SUCCESS, output=output, attempts=attempts, metadata=metadata or {})

    @staticmethod
    def failure(status: ToolStatus, message: str, category: str, attempts: int = 1) -> "ToolResult":
        return ToolResult(status=status, error_message=message, error_category=category, attempts=attempts)
