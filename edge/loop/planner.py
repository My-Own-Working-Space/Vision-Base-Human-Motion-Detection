from __future__ import annotations

from edge.context.models import IterationContext
from edge.loop.actions import Action, ActionType


class Planner:
    def create_plan(self, context: IterationContext) -> tuple[Action, ...]:
        action_id = f"{context.iteration_id}:vision-workflow"
        return (Action(
            action_id=action_id,
            action_type=ActionType.VISION_WORKFLOW,
            tool_name="vision.workflow.run",
            payload={
                "workflow_ref": context.workflow_ref,
                "image_path": str(context.image_path),
                "parameters": context.parameters,
            },
            idempotency_key=action_id,
        ),)
