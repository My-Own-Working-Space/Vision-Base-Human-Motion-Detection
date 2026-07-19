from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CapabilitySnapshot:
    provider_name: str
    workflow_ref: str
    status: str
    inputs: tuple[str, ...] = ()
    parameters: dict[str, str] = field(default_factory=dict)
    outputs: tuple[str, ...] = ()


@dataclass(frozen=True)
class IterationContext:
    run_id: str
    iteration_id: str
    iteration: int
    goal: str
    image_path: Path
    workflow_ref: str
    parameters: dict[str, Any]
    previous_action_summaries: tuple[str, ...] = ()
    repository_instructions: tuple[str, ...] = ()
    retrieved_documents: tuple[str, ...] = ()
    available_tools: tuple[str, ...] = ()
    capability_snapshot: CapabilitySnapshot | None = None
    error_history: tuple[str, ...] = ()
    remaining_budget_seconds: float | None = None
