from __future__ import annotations

from pathlib import Path

from edge.providers.base import VisionWorkflowProvider
from edge.providers.roboflow.errors import (
    CapabilityUnavailableError,
    InvalidProviderResponseError,
    RetryableProviderError,
)
from edge.providers.roboflow.models import WorkflowRunRequest, WorkflowRunResult
from edge.tools.base import ToolMetadata, ToolRequest
from edge.tools.policies import Idempotency, RetrySafety, SideEffect
from edge.tools.results import ToolResult, ToolStatus


class VisionWorkflowTool:
    metadata = ToolMetadata(
        name="vision.workflow.run",
        description="Run a provider-neutral vision workflow on one local image",
        input_schema="{workflow_ref: str, image_path: str, parameters?: object}",
        output_schema="WorkflowRunResult",
        timeout_seconds=20,
        retry_safety=RetrySafety.SAFE,
        side_effect=SideEffect.READ_ONLY,
        idempotency=Idempotency.IDEMPOTENT,
        required_capability="vision_workflow",
        redaction_fields=("api_key", "token"),
    )

    def __init__(self, provider: VisionWorkflowProvider) -> None:
        self._provider = provider

    def execute(self, request: ToolRequest) -> ToolResult:
        try:
            workflow_ref = str(request.payload["workflow_ref"])
            image_path = Path(str(request.payload["image_path"]))
            parameters = dict(request.payload.get("parameters", {}))
        except KeyError as exc:
            return ToolResult.failure(ToolStatus.INVALID_INPUT, f"Missing payload key: {exc}", "invalid_input")

        try:
            result = self._provider.run_workflow(WorkflowRunRequest(
                workflow_ref=workflow_ref,
                image_path=image_path,
                parameters=parameters,
                idempotency_key=request.idempotency_key,
            ))
        except CapabilityUnavailableError as exc:
            return ToolResult.failure(ToolStatus.CAPABILITY_UNAVAILABLE, str(exc), "capability_unavailable")
        except RetryableProviderError as exc:
            return ToolResult.failure(ToolStatus.RETRYABLE_FAILURE, str(exc), "retryable_provider_error")
        except InvalidProviderResponseError as exc:
            return ToolResult.failure(ToolStatus.PERMANENT_FAILURE, str(exc), "invalid_provider_response")

        if not isinstance(result, WorkflowRunResult):
            return ToolResult.failure(ToolStatus.PERMANENT_FAILURE, "Provider returned an invalid result type", "invalid_provider_response")
        if not isinstance(result.outputs, dict) or "predictions" not in result.outputs:
            return ToolResult.failure(ToolStatus.PERMANENT_FAILURE, "Provider result missing predictions output", "invalid_provider_response")
        return ToolResult.success(result, metadata={"provider": result.provider_name, "fake": result.fake})
