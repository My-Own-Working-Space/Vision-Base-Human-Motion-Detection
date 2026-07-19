from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = "harness-checkpoint-v1"


@dataclass(frozen=True)
class CompletedActionSummary:
    action_id: str
    action_type: str
    status: str
    attempts: int
    tool_name: str | None = None


@dataclass(frozen=True)
class Checkpoint:
    run_id: str
    goal: str
    current_state: str
    iteration: int
    completed_actions: tuple[CompletedActionSummary, ...] = ()
    latest_verification_result: dict[str, Any] = field(default_factory=dict)
    retry_counters: dict[str, int] = field(default_factory=dict)
    provider_capability_snapshot: dict[str, Any] = field(default_factory=dict)
    artifact_references: tuple[str, ...] = ()
    error_summary: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
