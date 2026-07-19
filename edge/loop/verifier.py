from __future__ import annotations

from dataclasses import dataclass

from edge.providers.roboflow.models import WorkflowRunResult
from edge.tools.results import ToolResult, ToolStatus


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {"passed": self.passed, "reason": self.reason}


class VisionWorkflowVerifier:
    def verify(self, result: ToolResult) -> VerificationResult:
        if result.status != ToolStatus.SUCCESS:
            return VerificationResult(False, f"Tool did not succeed: {result.status.value}")
        if not isinstance(result.output, WorkflowRunResult):
            return VerificationResult(False, "Tool output was not a WorkflowRunResult")
        predictions = result.output.outputs.get("predictions")
        if not isinstance(predictions, list):
            return VerificationResult(False, "Workflow output 'predictions' must be a list")
        if len(predictions) == 0:
            return VerificationResult(False, "Workflow output 'predictions' was empty")
        return VerificationResult(True, "Workflow output verified")
