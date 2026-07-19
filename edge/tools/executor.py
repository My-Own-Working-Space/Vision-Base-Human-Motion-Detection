from __future__ import annotations

import time
from time import monotonic

from edge.harness.events import EventType, HarnessEvent
from edge.harness.logging import StructuredEventLogger
from edge.harness.retry import RetryPolicy
from edge.tools.base import Tool, ToolRequest
from edge.tools.policies import Idempotency, RetrySafety
from edge.tools.results import ToolResult, ToolStatus


class ToolExecutor:
    def __init__(self, retry_policy: RetryPolicy, event_logger: StructuredEventLogger | None = None) -> None:
        self._retry_policy = retry_policy
        self._event_logger = event_logger or StructuredEventLogger()
        self._executed_non_idempotent: set[str] = set()

    def execute(self, tool: Tool, request: ToolRequest) -> ToolResult:
        fingerprint = f"{tool.metadata.name}:{request.action_id}"
        if tool.metadata.idempotency == Idempotency.NON_IDEMPOTENT and fingerprint in self._executed_non_idempotent:
            return ToolResult.failure(
                ToolStatus.CANCELLED,
                "Duplicate execution prevented for non-idempotent action",
                "duplicate_non_idempotent_action",
                attempts=0,
            )

        max_attempts = 1
        if tool.metadata.retry_safety == RetrySafety.SAFE:
            max_attempts = max(1, self._retry_policy.max_attempts)

        last_result: ToolResult | None = None
        for attempt in range(1, max_attempts + 1):
            delay = self._retry_policy.delay_for_attempt(attempt)
            if delay > 0:
                time.sleep(delay)

            self._event_logger.emit(HarnessEvent(
                event_type=EventType.TOOL_STARTED,
                run_id=request.run_id,
                iteration_id=request.iteration_id,
                action_id=request.action_id,
                tool_name=tool.metadata.name,
                retry_attempt=attempt,
            ))
            started = monotonic()
            try:
                result = tool.execute(request)
            except TimeoutError as exc:
                result = ToolResult.failure(ToolStatus.TIMEOUT, str(exc), "timeout", attempts=attempt)
            except ValueError as exc:
                result = ToolResult.failure(ToolStatus.INVALID_INPUT, str(exc), "invalid_input", attempts=attempt)
            except Exception as exc:
                result = ToolResult.failure(ToolStatus.PERMANENT_FAILURE, str(exc), type(exc).__name__, attempts=attempt)

            duration_ms = (monotonic() - started) * 1000
            last_result = result
            if result.status == ToolStatus.SUCCESS:
                if tool.metadata.idempotency == Idempotency.NON_IDEMPOTENT:
                    self._executed_non_idempotent.add(fingerprint)
                self._event_logger.emit(HarnessEvent(
                    event_type=EventType.TOOL_COMPLETED,
                    run_id=request.run_id,
                    iteration_id=request.iteration_id,
                    action_id=request.action_id,
                    tool_name=tool.metadata.name,
                    duration_ms=duration_ms,
                    retry_attempt=attempt,
                    outcome=result.status.value,
                ))
                return ToolResult.success(result.output, attempts=attempt, metadata=result.metadata)

            self._event_logger.emit(HarnessEvent(
                event_type=EventType.TOOL_FAILED,
                run_id=request.run_id,
                iteration_id=request.iteration_id,
                action_id=request.action_id,
                tool_name=tool.metadata.name,
                duration_ms=duration_ms,
                retry_attempt=attempt,
                outcome=result.status.value,
                error_category=result.error_category,
            ))

            if result.status != ToolStatus.RETRYABLE_FAILURE:
                return ToolResult(
                    status=result.status,
                    output=result.output,
                    error_message=result.error_message,
                    error_category=result.error_category,
                    attempts=attempt,
                    metadata=result.metadata,
                )

        assert last_result is not None
        return ToolResult(
            status=ToolStatus.RETRYABLE_FAILURE,
            error_message=last_result.error_message or "Retry attempts exhausted",
            error_category=last_result.error_category or "retry_exhausted",
            attempts=max_attempts,
            metadata=last_result.metadata,
        )
