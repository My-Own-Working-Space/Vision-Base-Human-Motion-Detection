from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from edge.harness.models import LocalImageTrigger, RunStatus
from edge.harness.retry import RetryPolicy
from edge.harness.runtime import HarnessRuntime
from edge.memory.file_store import FileCheckpointStore
from edge.providers.roboflow.fake import FakeVisionWorkflowProvider


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_IMAGE = ROOT / "alerts" / "alert_track_2_20260526_172904_540897.jpg"


class HarnessVerticalSliceTests(unittest.TestCase):
    def make_runtime(self, tmp: str, provider: FakeVisionWorkflowProvider | None = None, retry_policy: RetryPolicy | None = None) -> HarnessRuntime:
        return HarnessRuntime(
            provider=provider or FakeVisionWorkflowProvider(),
            checkpoint_store=FileCheckpointStore(Path(tmp) / "runs"),
            repository_root=ROOT,
            retry_policy=retry_policy or RetryPolicy(max_attempts=3, backoff_seconds=0.0),
        )

    def test_successful_end_to_end_fake_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp)
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(result.final_state, "COMPLETED")
            self.assertTrue(result.fake_provider)
            self.assertIn("predictions", result.outputs)
            self.assertTrue(result.checkpoint_path and result.checkpoint_path.exists())

    def test_missing_image_fails_and_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp)
            result = runtime.start_run(LocalImageTrigger(image_path=Path(tmp) / "missing.jpg"))

            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertIn("Image does not exist", result.error_summary or "")
            self.assertTrue(result.checkpoint_path and result.checkpoint_path.exists())

    def test_unsupported_image_type_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text_file = Path(tmp) / "not-image.txt"
            text_file.write_text("not an image")
            runtime = self.make_runtime(tmp)
            result = runtime.start_run(LocalImageTrigger(image_path=text_file))

            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertIn("Unsupported image type", result.error_summary or "")

    def test_retryable_tool_failure_followed_by_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeVisionWorkflowProvider(retryable_failures_before_success=2)
            runtime = self.make_runtime(tmp, provider=provider, retry_policy=RetryPolicy(max_attempts=3, backoff_seconds=0.0))
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunStatus.COMPLETED)
            self.assertEqual(provider.run_calls, 3)

    def test_retry_exhaustion_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeVisionWorkflowProvider(retryable_failures_before_success=5)
            runtime = self.make_runtime(tmp, provider=provider, retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.0))
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertEqual(provider.run_calls, 2)
            self.assertIn("Deterministic fake retryable failure", result.error_summary or "")

    def test_capability_unavailable_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeVisionWorkflowProvider(available=False)
            runtime = self.make_runtime(tmp, provider=provider)
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunStatus.BLOCKED)
            self.assertEqual(result.final_state, "BLOCKED")
            self.assertEqual(provider.run_calls, 0)

    def test_invalid_provider_response_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeVisionWorkflowProvider(invalid_response=True)
            runtime = self.make_runtime(tmp, provider=provider)
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertIn("invalid response", result.error_summary or "")

    def test_verification_failure_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = FakeVisionWorkflowProvider(empty_predictions=True)
            runtime = self.make_runtime(tmp, provider=provider)
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))

            self.assertEqual(result.status, RunStatus.FAILED)
            self.assertIn("predictions", result.error_summary or "")

    def test_checkpoint_creation_contains_resume_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp)
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            checkpoint = FileCheckpointStore(Path(tmp) / "runs").load(result.run_id)

            self.assertEqual(checkpoint.schema_version, "harness-checkpoint-v1")
            self.assertEqual(checkpoint.current_state, "COMPLETED")
            self.assertEqual(checkpoint.provider_capability_snapshot["image_path"], str(FIXTURE_IMAGE))
            self.assertNotIn("api_key", str(checkpoint.to_dict()).lower())

    def test_resume_from_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self.make_runtime(tmp)
            result = runtime.start_run(LocalImageTrigger(image_path=FIXTURE_IMAGE))
            resumed = runtime.resume_run(result.run_id)

            self.assertEqual(resumed.status, RunStatus.COMPLETED)
            self.assertEqual(resumed.run_id, result.run_id)
            self.assertEqual(resumed.iteration, 1)


if __name__ == "__main__":
    unittest.main()
