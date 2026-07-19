from __future__ import annotations

from edge.harness.events import EventType, HarnessEvent
from edge.harness.logging import StructuredEventLogger
from edge.loop.planner import Planner
from edge.loop.policies import ActionSelectionPolicy
from edge.loop.state import LoopRunState, LoopState, validate_transition
from edge.loop.verifier import VisionWorkflowVerifier
from edge.memory.models import CompletedActionSummary
from edge.tools.base import ToolRequest
from edge.tools.executor import ToolExecutor
from edge.tools.registry import ToolRegistry
from edge.tools.results import ToolStatus


class LoopController:
    def __init__(
        self,
        planner: Planner,
        selection_policy: ActionSelectionPolicy,
        tool_registry: ToolRegistry,
        tool_executor: ToolExecutor,
        verifier: VisionWorkflowVerifier,
        event_logger: StructuredEventLogger,
    ) -> None:
        self._planner = planner
        self._selection_policy = selection_policy
        self._tool_registry = tool_registry
        self._tool_executor = tool_executor
        self._verifier = verifier
        self._event_logger = event_logger

    def transition(self, run_state: LoopRunState, new_state: LoopState) -> None:
        old = run_state.state
        validate_transition(old, new_state)
        run_state.state = new_state
        self._event_logger.emit(HarnessEvent(
            event_type=EventType.STATE_TRANSITIONED,
            run_id=run_state.run_id,
            iteration_id=run_state.iteration_id,
            state_from=old.value,
            state_to=new_state.value,
        ))

    def run_once(self, run_state: LoopRunState) -> LoopRunState:
        if run_state.context is None:
            raise ValueError("LoopRunState.context is required")

        if run_state.context.capability_snapshot and run_state.context.capability_snapshot.status != "available":
            self.transition(run_state, LoopState.BLOCKED)
            run_state.error_summary = "Required provider capability is unavailable"
            return run_state

        self.transition(run_state, LoopState.PLANNED)
        run_state.plan = self._planner.create_plan(run_state.context)
        self._event_logger.emit(HarnessEvent(EventType.PLAN_CREATED, run_id=run_state.run_id, iteration_id=run_state.iteration_id))

        self.transition(run_state, LoopState.ACTION_SELECTED)
        run_state.selected_action = self._selection_policy.select_next(run_state.plan)
        action = run_state.selected_action
        self._event_logger.emit(HarnessEvent(
            EventType.ACTION_SELECTED,
            run_id=run_state.run_id,
            iteration_id=run_state.iteration_id,
            action_id=action.action_id,
            tool_name=action.tool_name,
        ))

        self.transition(run_state, LoopState.ACTION_RUNNING)
        tool = self._tool_registry.get(action.tool_name)
        result = self._tool_executor.execute(tool, ToolRequest(
            run_id=run_state.run_id,
            iteration_id=run_state.iteration_id,
            action_id=action.action_id,
            payload=action.payload,
            idempotency_key=action.idempotency_key,
        ))
        run_state.tool_result = result
        run_state.completed_actions.append(CompletedActionSummary(
            action_id=action.action_id,
            action_type=action.action_type.value,
            status=result.status.value,
            attempts=result.attempts,
            tool_name=action.tool_name,
        ))

        if result.status == ToolStatus.CAPABILITY_UNAVAILABLE:
            self.transition(run_state, LoopState.BLOCKED)
            run_state.error_summary = result.error_message
            return run_state
        if result.status != ToolStatus.SUCCESS:
            self.transition(run_state, LoopState.FAILED)
            run_state.error_summary = result.error_message
            return run_state

        self.transition(run_state, LoopState.VERIFYING)
        verification = self._verifier.verify(result)
        run_state.verification_result = verification.to_dict()
        self._event_logger.emit(HarnessEvent(
            EventType.VERIFICATION_COMPLETED,
            run_id=run_state.run_id,
            iteration_id=run_state.iteration_id,
            action_id=action.action_id,
            outcome="passed" if verification.passed else "failed",
        ))
        if not verification.passed:
            self.transition(run_state, LoopState.FAILED)
            run_state.error_summary = verification.reason
            return run_state

        self.transition(run_state, LoopState.CHECKPOINTED)
        self.transition(run_state, LoopState.COMPLETED)
        return run_state
