from __future__ import annotations

import hashlib
from pathlib import Path

from edge.providers.roboflow.errors import (
    CapabilityUnavailableError,
    InvalidProviderResponseError,
    RetryableProviderError,
)
from edge.providers.roboflow.models import (
    CapabilityStatus,
    WorkflowDescription,
    WorkflowRunRequest,
    WorkflowRunResult,
)


class FakeVisionWorkflowProvider:
    """Deterministic fake provider for offline harness tests and local examples."""

    provider_name = "fake-vision-provider"

    def __init__(
        self,
        available: bool = True,
        retryable_failures_before_success: int = 0,
        invalid_response: bool = False,
        empty_predictions: bool = False,
    ) -> None:
        self.available = available
        self.retryable_failures_before_success = retryable_failures_before_success
        self.invalid_response = invalid_response
        self.empty_predictions = empty_predictions
        self.run_calls = 0

    def describe_workflow(self, workflow_ref: str) -> WorkflowDescription:
        status = CapabilityStatus.AVAILABLE if self.available else CapabilityStatus.UNAVAILABLE
        return WorkflowDescription(
            provider_name=self.provider_name,
            workflow_ref=workflow_ref,
            inputs=("image",),
            parameters={},
            outputs=("predictions", "provider_notice"),
            capability_status=status,
            raw_schema_ref="fake-local-schema-v1",
        )

    def run_workflow(self, request: WorkflowRunRequest) -> WorkflowRunResult:
        if not self.available:
            raise CapabilityUnavailableError("Fake provider capability is unavailable")
        self.run_calls += 1
        if self.run_calls <= self.retryable_failures_before_success:
            raise RetryableProviderError("Deterministic fake retryable failure")
        if self.invalid_response:
            raise InvalidProviderResponseError("Fake provider returned an invalid response")

        digest = hashlib.sha256(Path(request.image_path).read_bytes()).hexdigest()
        predictions = [] if self.empty_predictions else [{
            "label": "fake_object",
            "confidence": 0.42,
            "source": "deterministic_fake_provider",
            "image_sha256_prefix": digest[:12],
        }]
        return WorkflowRunResult(
            provider_name=self.provider_name,
            workflow_ref=request.workflow_ref,
            outputs={
                "predictions": predictions,
                "provider_notice": "FAKE_RESULT_DO_NOT_TREAT_AS_MODEL_INFERENCE",
            },
            fake=True,
        )
