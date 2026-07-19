from __future__ import annotations

import uuid
from pathlib import Path

from edge.context.assembler import ContextAssembler, ContextAssemblyRequest
from edge.context.capabilities import CapabilitySnapshotter
from edge.context.repository_context import RepositoryContextLoader
from edge.harness.checkpoint import CheckpointManager
from edge.harness.events import EventType, HarnessEvent
from edge.harness.logging import StructuredEventLogger
from edge.harness.models import LocalImageTrigger, RunResult, RunStatus
from edge.harness.retry import RetryPolicy
from edge.loop.controller import LoopController
from edge.loop.planner import Planner
from edge.loop.policies import ActionSelectionPolicy
from edge.loop.state import LoopRunState, LoopState
from edge.loop.verifier import VisionWorkflowVerifier
from edge.memory.file_store import FileCheckpointStore
from edge.memory.models import Checkpoint
from edge.providers.base import VisionWorkflowProvider
from edge.providers.roboflow.fake import FakeVisionWorkflowProvider
from edge.tools.executor import ToolExecutor
from edge.tools.registry import ToolRegistry
from edge.tools.vision_workflow import VisionWorkflowTool


class HarnessRuntime:
    def __init__(
        self,
        provider: VisionWorkflowProvider | None = None,
        checkpoint_store: FileCheckpointStore | None = None,
        event_logger: StructuredEventLogger | None = None,
        repository_root: Path | str = ".",
        retry_policy: RetryPolicy | None = None,
    ) -> None:
        self.provider = provider or FakeVisionWorkflowProvider()
        self.event_logger = event_logger or StructuredEventLogger()
        self.checkpoint_store = checkpoint_store or FileCheckpointStore()
        self.checkpoint_manager = CheckpointManager(self.checkpoint_store)
        self.tool_registry = ToolRegistry()
        self.tool_registry.register(VisionWorkflowTool(self.provider))
        self.context_assembler = ContextAssembler(
            repository_loader=RepositoryContextLoader(repository_root),
            capability_snapshotter=CapabilitySnapshotter(self.provider),
            tool_registry=self.tool_registry,
        )
        self.controller = LoopController(
            planner=Planner(),
            selection_policy=ActionSelectionPolicy(),
            tool_registry=self.tool_registry,
            tool_executor=ToolExecutor(retry_policy or RetryPolicy(), self.event_logger),
            verifier=VisionWorkflowVerifier(),
            event_logger=self.event_logger,
        )

    def start_run(self, trigger: LocalImageTrigger) -> RunResult:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        return self._execute(run_id=run_id, trigger=trigger, previous_checkpoint=None, iteration=1)

    def resume_run(self, run_id: str) -> RunResult:
        checkpoint = self.checkpoint_store.load(run_id)
        if checkpoint.current_state == LoopState.COMPLETED.value:
            return RunResult(
                run_id=checkpoint.run_id,
                status=RunStatus.COMPLETED,
                final_state=checkpoint.current_state,
                iteration=checkpoint.iteration,
                checkpoint_path=self.checkpoint_store.path_for(run_id),
                error_summary=checkpoint.error_summary,
            )
        image_path = Path(checkpoint.provider_capability_snapshot.get("image_path", ""))
        if not image_path:
            raise ValueError("Checkpoint does not contain a resumable image path")
        trigger = LocalImageTrigger(
            image_path=image_path,
            workflow_ref=str(checkpoint.provider_capability_snapshot.get("workflow_ref", "fake/vision-workflow")),
            goal=checkpoint.goal,
        )
        return self._execute(run_id=run_id, trigger=trigger, previous_checkpoint=checkpoint, iteration=checkpoint.iteration + 1)

    def _execute(
        self,
        run_id: str,
        trigger: LocalImageTrigger,
        previous_checkpoint: Checkpoint | None,
        iteration: int,
    ) -> RunResult:
        iteration_id = f"{run_id}:iter-{iteration}"
        self.event_logger.emit(HarnessEvent(EventType.RUN_STARTED, run_id=run_id, iteration_id=iteration_id))
        run_state = LoopRunState(run_id=run_id, iteration_id=iteration_id, iteration=iteration)

        try:
            context = self.context_assembler.assemble(ContextAssemblyRequest(
                run_id=run_id,
                iteration_id=iteration_id,
                iteration=iteration,
                trigger=trigger,
                previous_checkpoint=previous_checkpoint,
            ))
            run_state.context = context
            self.controller.transition(run_state, LoopState.CONTEXT_READY)
            self.event_logger.emit(HarnessEvent(EventType.CONTEXT_ASSEMBLED, run_id=run_id, iteration_id=iteration_id))
            run_state = self.controller.run_once(run_state)
        except Exception as exc:
            if run_state.state not in (LoopState.FAILED, LoopState.BLOCKED):
                try:
                    self.controller.transition(run_state, LoopState.FAILED)
                except Exception:
                    run_state.state = LoopState.FAILED
            run_state.error_summary = str(exc)

        checkpoint_path = self.checkpoint_manager.save(run_state)
        self.event_logger.emit(HarnessEvent(
            EventType.CHECKPOINT_SAVED,
            run_id=run_id,
            iteration_id=iteration_id,
            outcome=str(checkpoint_path) if checkpoint_path else None,
        ))

        status = self._status_from_state(run_state.state)
        event_type = {
            RunStatus.COMPLETED: EventType.RUN_COMPLETED,
            RunStatus.BLOCKED: EventType.RUN_BLOCKED,
            RunStatus.FAILED: EventType.RUN_FAILED,
        }.get(status, EventType.RUN_FAILED)
        self.event_logger.emit(HarnessEvent(event_type, run_id=run_id, iteration_id=iteration_id, outcome=status.value))

        outputs = {}
        artifacts: list[str] = []
        fake_provider = False
        if run_state.tool_result and run_state.tool_result.output is not None:
            output = run_state.tool_result.output
            outputs = getattr(output, "outputs", {})
            artifacts = [str(item.path) for item in getattr(output, "artifacts", ())]
            fake_provider = bool(getattr(output, "fake", False))

        return RunResult(
            run_id=run_id,
            status=status,
            final_state=run_state.state.value,
            iteration=iteration,
            outputs=outputs,
            artifacts=artifacts,
            checkpoint_path=checkpoint_path,
            error_summary=run_state.error_summary,
            fake_provider=fake_provider,
        )

    def _status_from_state(self, state: LoopState) -> RunStatus:
        if state == LoopState.COMPLETED:
            return RunStatus.COMPLETED
        if state == LoopState.BLOCKED:
            return RunStatus.BLOCKED
        if state == LoopState.ESCALATED:
            return RunStatus.ESCALATED
        return RunStatus.FAILED
