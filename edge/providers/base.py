from __future__ import annotations

from typing import Protocol

from edge.providers.roboflow.models import WorkflowDescription, WorkflowRunRequest, WorkflowRunResult


class VisionWorkflowProvider(Protocol):
    provider_name: str

    def describe_workflow(self, workflow_ref: str) -> WorkflowDescription:
        ...

    def run_workflow(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        ...
