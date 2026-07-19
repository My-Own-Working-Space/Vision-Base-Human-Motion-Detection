from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from edge.context.capabilities import CapabilitySnapshotter
from edge.context.history import summarize_history
from edge.context.models import IterationContext
from edge.context.repository_context import RepositoryContextLoader
from edge.context.retrieval import retrieve_documents
from edge.harness.errors import InvalidTriggerError
from edge.harness.models import LocalImageTrigger
from edge.memory.models import Checkpoint
from edge.tools.registry import ToolRegistry

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ContextAssemblyRequest:
    run_id: str
    iteration_id: str
    iteration: int
    trigger: LocalImageTrigger
    previous_checkpoint: Checkpoint | None = None
    remaining_budget_seconds: float | None = None


class ContextAssembler:
    def __init__(
        self,
        repository_loader: RepositoryContextLoader,
        capability_snapshotter: CapabilitySnapshotter,
        tool_registry: ToolRegistry,
    ) -> None:
        self._repository_loader = repository_loader
        self._capability_snapshotter = capability_snapshotter
        self._tool_registry = tool_registry

    def assemble(self, request: ContextAssemblyRequest) -> IterationContext:
        image_path = Path(request.trigger.image_path)
        if not image_path.exists():
            raise InvalidTriggerError(f"Image does not exist: {image_path}")
        if image_path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
            raise InvalidTriggerError(f"Unsupported image type: {image_path.suffix}")

        capability_snapshot = self._capability_snapshotter.snapshot(request.trigger.workflow_ref)
        return IterationContext(
            run_id=request.run_id,
            iteration_id=request.iteration_id,
            iteration=request.iteration,
            goal=request.trigger.goal,
            image_path=image_path,
            workflow_ref=request.trigger.workflow_ref,
            parameters=request.trigger.parameters,
            previous_action_summaries=summarize_history(request.previous_checkpoint),
            repository_instructions=self._repository_loader.load_instructions(),
            retrieved_documents=retrieve_documents(request.trigger.goal),
            available_tools=tuple(self._tool_registry.list_names()),
            capability_snapshot=capability_snapshot,
            error_history=tuple(filter(None, [request.previous_checkpoint.error_summary if request.previous_checkpoint else None])),
            remaining_budget_seconds=request.remaining_budget_seconds,
        )
