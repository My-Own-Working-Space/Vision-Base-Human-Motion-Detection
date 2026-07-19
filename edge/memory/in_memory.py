from __future__ import annotations

from edge.memory.models import Checkpoint


class InMemoryCheckpointStore:
    def __init__(self) -> None:
        self._items: dict[str, Checkpoint] = {}

    def save(self, checkpoint: Checkpoint) -> str:
        self._items[checkpoint.run_id] = checkpoint
        return checkpoint.run_id

    def load(self, run_id: str) -> Checkpoint:
        return self._items[run_id]
