from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class CapabilityStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class WorkflowDescription:
    provider_name: str
    workflow_ref: str
    inputs: tuple[str, ...]
    parameters: dict[str, str] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()
    capability_status: CapabilityStatus = CapabilityStatus.AVAILABLE
    raw_schema_ref: str | None = None


@dataclass(frozen=True)
class WorkflowRunRequest:
    workflow_ref: str
    image_path: Path
    parameters: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None


@dataclass(frozen=True)
class WorkflowArtifact:
    name: str
    path: Path
    media_type: str


@dataclass(frozen=True)
class WorkflowRunResult:
    provider_name: str
    workflow_ref: str
    outputs: dict[str, Any]
    artifacts: tuple[WorkflowArtifact, ...] = ()
    fake: bool = False
