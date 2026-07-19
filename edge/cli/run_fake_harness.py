from __future__ import annotations

import argparse
from pathlib import Path

from edge.harness.models import LocalImageTrigger
from edge.harness.runtime import HarnessRuntime
from edge.memory.file_store import FileCheckpointStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local fake-provider harness on one image.")
    parser.add_argument("image_path", help="Path to a local jpg/png/webp image")
    parser.add_argument("--checkpoint-root", default="harness_runs")
    args = parser.parse_args()

    runtime = HarnessRuntime(checkpoint_store=FileCheckpointStore(args.checkpoint_root), repository_root=Path.cwd())
    result = runtime.start_run(LocalImageTrigger(image_path=Path(args.image_path)))
    print({
        "run_id": result.run_id,
        "status": result.status.value,
        "final_state": result.final_state,
        "checkpoint_path": str(result.checkpoint_path) if result.checkpoint_path else None,
        "fake_provider": result.fake_provider,
        "output_keys": sorted(result.outputs.keys()),
    })
    return 0 if result.status.value == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
