from __future__ import annotations

from edge.memory.models import Checkpoint


def summarize_history(checkpoint: Checkpoint | None) -> tuple[str, ...]:
    if checkpoint is None:
        return ()
    return tuple(
        f"{item.action_id}:{item.action_type}:{item.status}:{item.attempts}"
        for item in checkpoint.completed_actions
    )
