from __future__ import annotations

from edge.providers.roboflow.errors import CapabilityUnavailableError
from edge.providers.roboflow.models import WorkflowDescription, WorkflowRunRequest, WorkflowRunResult


class RoboflowWorkflowAdapter:
    """Roboflow adapter boundary. Raw MCP/SDK schemas must stay inside this class."""

    provider_name = "roboflow"

    def describe_workflow(self, workflow_ref: str) -> WorkflowDescription:
        # TODO: When Roboflow MCP is used for this adapter, call the real read-only
        # workflow-description tool here and map the real response into WorkflowDescription.
        raise CapabilityUnavailableError("Roboflow live adapter is not wired in this provider-independent slice")

    def run_workflow(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        # TODO: Call the real Roboflow SDK/MCP workflow run operation and convert the
        # real payload immediately into WorkflowRunResult. Do not leak raw payloads.
        raise CapabilityUnavailableError("Roboflow live adapter is not wired in this provider-independent slice")
