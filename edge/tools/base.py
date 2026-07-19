from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from edge.tools.policies import Idempotency, RetrySafety, SideEffect
from edge.tools.results import ToolResult


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    input_schema: str
    output_schema: str
    timeout_seconds: float
    retry_safety: RetrySafety
    side_effect: SideEffect
    idempotency: Idempotency
    required_capability: str | None = None
    redaction_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolRequest:
    run_id: str
    iteration_id: str
    action_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None


class Tool(Protocol):
    metadata: ToolMetadata

    def execute(self, request: ToolRequest) -> ToolResult:
        ...
