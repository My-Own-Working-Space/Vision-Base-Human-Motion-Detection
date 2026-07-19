from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from edge.context.models import IterationContext
from edge.loop.actions import Action
from edge.memory.models import CompletedActionSummary
from edge.tools.results import ToolResult


class LoopState(str, Enum):
    CREATED = "CREATED"
    CONTEXT_READY = "CONTEXT_READY"
    PLANNED = "PLANNED"
    ACTION_SELECTED = "ACTION_SELECTED"
    ACTION_RUNNING = "ACTION_RUNNING"
    VERIFYING = "VERIFYING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    RETRY_PENDING = "RETRY_PENDING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"


VALID_TRANSITIONS = {
    LoopState.CREATED: {LoopState.CONTEXT_READY, LoopState.FAILED},
    LoopState.CONTEXT_READY: {LoopState.PLANNED, LoopState.BLOCKED, LoopState.FAILED},
    LoopState.PLANNED: {LoopState.ACTION_SELECTED, LoopState.FAILED},
    LoopState.ACTION_SELECTED: {LoopState.ACTION_RUNNING, LoopState.FAILED},
    LoopState.ACTION_RUNNING: {LoopState.VERIFYING, LoopState.RETRY_PENDING, LoopState.BLOCKED, LoopState.FAILED},
    LoopState.RETRY_PENDING: {LoopState.ACTION_RUNNING, LoopState.FAILED},
    LoopState.VERIFYING: {LoopState.CHECKPOINTED, LoopState.FAILED},
    LoopState.CHECKPOINTED: {LoopState.COMPLETED, LoopState.BLOCKED, LoopState.FAILED},
    LoopState.COMPLETED: set(),
    LoopState.BLOCKED: set(),
    LoopState.FAILED: set(),
    LoopState.ESCALATED: set(),
}


def validate_transition(current: LoopState, new: LoopState) -> None:
    if new not in VALID_TRANSITIONS[current]:
        raise ValueError(f"Invalid loop transition: {current.value} -> {new.value}")


@dataclass
class LoopRunState:
    run_id: str
    iteration_id: str
    iteration: int
    state: LoopState = LoopState.CREATED
    context: IterationContext | None = None
    plan: tuple[Action, ...] = ()
    selected_action: Action | None = None
    tool_result: ToolResult | None = None
    verification_result: dict[str, Any] = field(default_factory=dict)
    completed_actions: list[CompletedActionSummary] = field(default_factory=list)
    error_summary: str | None = None
