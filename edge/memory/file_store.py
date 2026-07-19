from __future__ import annotations

import json
from pathlib import Path

from edge.memory.models import Checkpoint, CompletedActionSummary


class FileCheckpointStore:
    def __init__(self, root: Path | str = "harness_runs") -> None:
        self.root = Path(root)

    def save(self, checkpoint: Checkpoint) -> str:
        run_dir = self.root / checkpoint.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "checkpoint.json"
        path.write_text(json.dumps(checkpoint.to_dict(), indent=2, sort_keys=True, default=str))
        return str(path)

    def load(self, run_id: str) -> Checkpoint:
        path = self.root / run_id / "checkpoint.json"
        data = json.loads(path.read_text())
        actions = tuple(CompletedActionSummary(**item) for item in data.get("completed_actions", []))
        return Checkpoint(
            run_id=data["run_id"],
            goal=data["goal"],
            current_state=data["current_state"],
            iteration=int(data["iteration"]),
            completed_actions=actions,
            latest_verification_result=data.get("latest_verification_result", {}),
            retry_counters=data.get("retry_counters", {}),
            provider_capability_snapshot=data.get("provider_capability_snapshot", {}),
            artifact_references=tuple(data.get("artifact_references", [])),
            error_summary=data.get("error_summary"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            schema_version=data.get("schema_version", "harness-checkpoint-v1"),
        )

    def path_for(self, run_id: str) -> Path:
        return self.root / run_id / "checkpoint.json"
