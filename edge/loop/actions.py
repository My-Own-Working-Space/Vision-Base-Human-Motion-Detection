from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    VISION_WORKFLOW = "vision_workflow"


@dataclass(frozen=True)
class Action:
    action_id: str
    action_type: ActionType
    tool_name: str
    payload: dict[str, Any] = field(default_factory=dict)
    idempotency_key: str | None = None
