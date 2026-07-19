from __future__ import annotations

from pathlib import Path

from edge.loop.state import LoopRunState
from edge.memory.models import Checkpoint
from edge.memory.store import CheckpointStore


def build_checkpoint(run_state: LoopRunState) -> Checkpoint:
    artifacts: tuple[str, ...] = ()
    if run_state.tool_result and getattr(run_state.tool_result.output, "artifacts", None):
        artifacts = tuple(str(item.path) for item in run_state.tool_result.output.artifacts)

    capability = {}
    if run_state.context and run_state.context.capability_snapshot:
        snap = run_state.context.capability_snapshot
        capability = {
            "provider_name": snap.provider_name,
            "workflow_ref": snap.workflow_ref,
            "status": snap.status,
            "inputs": list(snap.inputs),
            "parameters": snap.parameters,
            "outputs": list(snap.outputs),
            "image_path": str(run_state.context.image_path),
        }

    return Checkpoint(
        run_id=run_state.run_id,
        goal=run_state.context.goal if run_state.context else "",
        current_state=run_state.state.value,
        iteration=run_state.iteration,
        completed_actions=tuple(run_state.completed_actions),
        latest_verification_result=run_state.verification_result,
        retry_counters={item.action_id: item.attempts for item in run_state.completed_actions},
        provider_capability_snapshot=capability,
        artifact_references=artifacts,
        error_summary=run_state.error_summary,
    )


class CheckpointManager:
    def __init__(self, store: CheckpointStore) -> None:
        self._store = store

    def save(self, run_state: LoopRunState) -> Path | None:
        location = self._store.save(build_checkpoint(run_state))
        try:
            return Path(location)
        except TypeError:
            return None
