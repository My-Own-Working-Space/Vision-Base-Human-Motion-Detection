from __future__ import annotations

import base64
import binascii
import concurrent.futures
import imghdr
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from inference_sdk import InferenceHTTPClient

from edge.config import RoboflowWorkflowConfig
from edge.logging_config import get_logger

logger = get_logger(__name__)


class RoboflowWorkflowError(RuntimeError):
    """Base error for Roboflow workflow failures."""


class RoboflowWorkflowConfigError(RoboflowWorkflowError):
    """Raised when required Roboflow configuration is missing."""


class RoboflowWorkflowRequestError(RoboflowWorkflowError):
    """Raised when the workflow request fails after retries."""


class RoboflowWorkflowResponseError(RoboflowWorkflowError):
    """Raised when the workflow returns an unexpected response shape."""


class RoboflowImageOutputError(RoboflowWorkflowError):
    """Raised when an image-shaped workflow output cannot be decoded."""


@dataclass(frozen=True)
class RoboflowWorkflowResult:
    outputs: dict[str, Any]
    decoded_image_paths: dict[str, Path] = field(default_factory=dict)


WorkflowRunner = Callable[..., Any]


class RoboflowWorkflowClient:
    """Runs the configured Roboflow Workflow and validates its declared outputs."""

    def __init__(
        self,
        config: RoboflowWorkflowConfig,
        runner: WorkflowRunner | None = None,
    ) -> None:
        self._config = config
        self._runner = runner
        self._client: InferenceHTTPClient | None = None

    def run_evn_object_detection(
        self,
        image: str | bytes | Any,
        parameters: dict[str, Any] | None = None,
        output_directory: str | Path | None = None,
    ) -> RoboflowWorkflowResult:
        """Run the EVN object detection workflow on one image."""
        self._validate_config()
        raw_response = self._run_with_retries(image=image, parameters=parameters or {})
        outputs = self._extract_single_result(raw_response)
        self._validate_expected_outputs(outputs)

        decoded_paths = self._decode_image_outputs(
            outputs=outputs,
            output_directory=Path(output_directory or self._config.image_output_directory),
        )
        return RoboflowWorkflowResult(outputs=outputs, decoded_image_paths=decoded_paths)

    def _validate_config(self) -> None:
        if not self._config.api_key:
            raise RoboflowWorkflowConfigError("ROBOFLOW_API_KEY is required to run the Roboflow workflow")
        if not self._config.workspace_name:
            raise RoboflowWorkflowConfigError("ROBOFLOW_WORKSPACE_NAME is required")
        if not self._config.workflow_id:
            raise RoboflowWorkflowConfigError("ROBOFLOW_WORKFLOW_ID is required")
        if not self._config.image_input_name:
            raise RoboflowWorkflowConfigError("ROBOFLOW_IMAGE_INPUT_NAME is required")

    def _run_with_retries(self, image: str | bytes | Any, parameters: dict[str, Any]) -> Any:
        max_attempts = max(1, self._config.max_retries + 1)
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                return self._run_once(image=image, parameters=parameters)
            except Exception as exc:
                last_error = exc
                if attempt >= max_attempts:
                    break
                delay = self._config.retry_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Roboflow workflow attempt %d/%d failed; retrying in %.1fs: %s",
                    attempt,
                    max_attempts,
                    delay,
                    self._summarize_error(exc),
                )
                time.sleep(delay)

        raise RoboflowWorkflowRequestError(
            f"Roboflow workflow failed after {max_attempts} attempt(s): "
            f"{self._summarize_error(last_error)}"
        ) from last_error

    def _run_once(self, image: str | bytes | Any, parameters: dict[str, Any]) -> Any:
        runner = self._runner or self._get_client().run_workflow

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(
            runner,
            workspace_name=self._config.workspace_name,
            workflow_id=self._config.workflow_id,
            images={self._config.image_input_name: image},
            parameters=parameters,
        )
        try:
            return future.result(timeout=self._config.timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise RoboflowWorkflowRequestError(
                f"Roboflow workflow timed out after {self._config.timeout_seconds}s"
            ) from exc
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    def _get_client(self) -> InferenceHTTPClient:
        if self._client is None:
            self._client = InferenceHTTPClient(
                api_url=self._config.api_url,
                api_key=self._config.api_key,
            )
        return self._client

    def _extract_single_result(self, raw_response: Any) -> dict[str, Any]:
        if not isinstance(raw_response, list):
            raise RoboflowWorkflowResponseError(
                f"Expected workflow response to be a list, got {type(raw_response).__name__}"
            )
        if len(raw_response) != 1:
            raise RoboflowWorkflowResponseError(
                f"Expected one workflow result, got {len(raw_response)}"
            )
        result = raw_response[0]
        if not isinstance(result, dict):
            raise RoboflowWorkflowResponseError(
                f"Expected workflow result to be a dict, got {type(result).__name__}"
            )
        return result

    def _validate_expected_outputs(self, outputs: dict[str, Any]) -> None:
        missing = [key for key in self._config.expected_output_keys if key not in outputs]
        if missing:
            raise RoboflowWorkflowResponseError(
                f"Workflow response missing expected output key(s): {', '.join(missing)}"
            )

    def _decode_image_outputs(
        self,
        outputs: dict[str, Any],
        output_directory: Path,
    ) -> dict[str, Path]:
        decoded: dict[str, Path] = {}
        for key, value in outputs.items():
            image_data = self._extract_base64_image(value)
            if image_data is None:
                continue

            output_directory.mkdir(parents=True, exist_ok=True)
            image_bytes = self._decode_base64_image(image_data, key)
            extension = imghdr.what(None, h=image_bytes) or "png"
            path = output_directory / f"{key}.{extension}"
            path.write_bytes(image_bytes)
            decoded[key] = path
        return decoded

    def _extract_base64_image(self, value: Any) -> str | None:
        if isinstance(value, str) and self._looks_like_base64_image(value):
            return value
        if not isinstance(value, dict):
            return None

        value_type = str(value.get("type", "")).lower()
        if value_type == "base64" and isinstance(value.get("value"), str):
            return value["value"]

        for candidate_key in ("base64", "image", "value"):
            candidate = value.get(candidate_key)
            if isinstance(candidate, str) and self._looks_like_base64_image(candidate):
                return candidate
        return None

    def _looks_like_base64_image(self, value: str) -> bool:
        if value.startswith("data:image/"):
            return True
        if len(value) < 64:
            return False
        return value[:16].replace("+", "").replace("/", "").replace("=", "").isalnum()

    def _decode_base64_image(self, value: str, output_key: str) -> bytes:
        if "," in value and value.startswith("data:image/"):
            value = value.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(value, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise RoboflowImageOutputError(
                f"Workflow output {output_key!r} looked like base64 image data but could not be decoded"
            ) from exc

        if imghdr.what(None, h=image_bytes) is None:
            raise RoboflowImageOutputError(
                f"Workflow output {output_key!r} decoded to bytes that are not a known image format"
            )
        return image_bytes

    def _summarize_error(self, error: Exception | None) -> str:
        if error is None:
            return "unknown error"
        message = str(error)
        if len(message) > 300:
            return f"{message[:300]}..."
        return message
