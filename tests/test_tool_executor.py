from __future__ import annotations

import unittest

from edge.harness.retry import RetryPolicy
from edge.tools.base import ToolMetadata, ToolRequest
from edge.tools.executor import ToolExecutor
from edge.tools.policies import Idempotency, RetrySafety, SideEffect
from edge.tools.results import ToolResult, ToolStatus


class NonIdempotentTool:
    metadata = ToolMetadata(
        name="non.idempotent",
        description="test",
        input_schema="{}",
        output_schema="{}",
        timeout_seconds=1,
        retry_safety=RetrySafety.UNSAFE,
        side_effect=SideEffect.EXTERNAL_CALL,
        idempotency=Idempotency.NON_IDEMPOTENT,
    )

    def __init__(self) -> None:
        self.calls = 0

    def execute(self, request: ToolRequest) -> ToolResult:
        self.calls += 1
        return ToolResult.success({"ok": True})


class ToolExecutorTests(unittest.TestCase):
    def test_prevents_duplicate_execution_for_non_idempotent_action(self) -> None:
        tool = NonIdempotentTool()
        executor = ToolExecutor(RetryPolicy(max_attempts=3, backoff_seconds=0.0))
        request = ToolRequest(run_id="run-1", iteration_id="iter-1", action_id="action-1")

        first = executor.execute(tool, request)
        second = executor.execute(tool, request)

        self.assertEqual(first.status, ToolStatus.SUCCESS)
        self.assertEqual(second.status, ToolStatus.CANCELLED)
        self.assertEqual(tool.calls, 1)


if __name__ == "__main__":
    unittest.main()
