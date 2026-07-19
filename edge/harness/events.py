from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    CONTEXT_ASSEMBLED = "context.assembled"
    PLAN_CREATED = "plan.created"
    ACTION_SELECTED = "action.selected"
    TOOL_STARTED = "tool.started"
    TOOL_COMPLETED = "tool.completed"
    TOOL_FAILED = "tool.failed"
    VERIFICATION_COMPLETED = "verification.completed"
    CHECKPOINT_SAVED = "checkpoint.saved"
    STATE_TRANSITIONED = "state.transitioned"
    RUN_COMPLETED = "run.completed"
    RUN_BLOCKED = "run.blocked"
    RUN_FAILED = "run.failed"


@dataclass(frozen=True)
class HarnessEvent:
    event_type: EventType
    run_id: str
    iteration_id: str | None = None
    action_id: str | None = None
    tool_name: str | None = None
    provider: str | None = None
    state_from: str | None = None
    state_to: str | None = None
    duration_ms: float | None = None
    retry_attempt: int | None = None
    outcome: str | None = None
    error_category: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        data = {
            "timestamp": self.timestamp,
            "event_type": self.event_type.value,
            "run_id": self.run_id,
            "iteration_id": self.iteration_id,
            "action_id": self.action_id,
            "tool_name": self.tool_name,
            "provider": self.provider,
            "state_from": self.state_from,
            "state_to": self.state_to,
            "duration_ms": self.duration_ms,
            "retry_attempt": self.retry_attempt,
            "outcome": self.outcome,
            "error_category": self.error_category,
            "details": self.details,
        }
        return {key: value for key, value in data.items() if value is not None}
