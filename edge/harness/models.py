from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    FAILED = "failed"
    ESCALATED = "escalated"


@dataclass(frozen=True)
class LocalImageTrigger:
    image_path: Path
    workflow_ref: str = "fake/vision-workflow"
    goal: str = "Run a vision workflow on a local image"
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunResult:
    run_id: str
    status: RunStatus
    final_state: str
    iteration: int
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    checkpoint_path: Path | None = None
    error_summary: str | None = None
    fake_provider: bool = False
