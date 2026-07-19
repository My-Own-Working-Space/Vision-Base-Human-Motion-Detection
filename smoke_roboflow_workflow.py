from __future__ import annotations

import sys

from edge.clients.roboflow_workflow_client import RoboflowWorkflowClient
from edge.config import load_config
from edge.logging_config import setup_logging


SAMPLE_IMAGE_URL = "https://media.roboflow.com/notebooks/examples/dog.jpeg"


def main() -> int:
    setup_logging()
    config = load_config().roboflow_workflow
    client = RoboflowWorkflowClient(config)

    result = client.run_evn_object_detection(SAMPLE_IMAGE_URL)
    missing = [key for key in config.expected_output_keys if key not in result.outputs]
    if missing:
        raise AssertionError(f"Missing expected workflow output key(s): {', '.join(missing)}")

    print(f"Roboflow workflow smoke test passed: {sorted(result.outputs.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
