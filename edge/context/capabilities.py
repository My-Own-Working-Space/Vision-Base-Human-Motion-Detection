from __future__ import annotations

from edge.context.models import CapabilitySnapshot
from edge.providers.base import VisionWorkflowProvider


class CapabilitySnapshotter:
    def __init__(self, provider: VisionWorkflowProvider) -> None:
        self._provider = provider

    def snapshot(self, workflow_ref: str) -> CapabilitySnapshot:
        description = self._provider.describe_workflow(workflow_ref)
        return CapabilitySnapshot(
            provider_name=description.provider_name,
            workflow_ref=description.workflow_ref,
            status=description.capability_status.value,
            inputs=description.inputs,
            parameters=description.parameters,
            outputs=description.outputs,
        )
