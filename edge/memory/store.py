from __future__ import annotations

from typing import Protocol

from edge.memory.models import Checkpoint


class CheckpointStore(Protocol):
    def save(self, checkpoint: Checkpoint) -> str:
        ...

    def load(self, run_id: str) -> Checkpoint:
        ...
