# Coding Agent Context

This document is the durable handoff for coding agents working in this repository. Read it before making architectural or integration changes.

## Project Snapshot

- Project: Vision-Based Human Motion and Suspicious Behavior Detection.
- Runtime: Python 3.10+, FastAPI edge service, OpenCV camera pipeline, YOLOv8 + ByteTrack person tracking, optional behavior classifier, alert dispatch.
- Main entrypoint: `edge/main.py`.
- Configuration source: environment variables loaded by `edge/config.py` through `python-dotenv`.
- Dependency file: `requirements.txt`.
- Primary docs: `README.md`, `README.docker.md`, and this file.

## Current Architecture

The edge runtime is intentionally split by responsibility:

- `edge/services/camera_service.py`: captures frames, throttles FPS, resizes input, and controls the processing/sleep state.
- `edge/services/detection_service.py`: wraps ML inference and returns domain-level `DetectionEvent` values.
- `edge/services/alert_service.py`: confirmation/cooldown state machine, evidence persistence, CSV ledger, retry behavior.
- `edge/clients/api_client.py`: transport-only client for dashboard alert upload.
- `edge/clients/pms_bridge_client.py`: optional PMS bridge transport.
- `edge/ui/stream_renderer.py`: frame annotation and MJPEG rendering.
- `inference/`: model-specific black boxes for YOLO, MobileNetV3, and ResNet18+LSTM.

Do not put alert policy, CSV persistence, or transport concerns inside detection code. Do not put ML inference details inside UI or API clients.

## Composition Rules

- `edge/main.py` is the composition root. Create dependencies there and inject them into services.
- Importing `edge.main` must remain side-effect light: do not initialize ML models, start alert retry workers, or start PMS heartbeat threads at import time. Use `create_app(...)`, `compose_services(...)`, and lifespan startup instead.
- Environment variables should be represented in frozen dataclasses in `edge/config.py`.
- Do not read secrets directly throughout the codebase; load them once through config.
- Existing clients use `requests`, request timeouts, and `edge.logging_config.get_logger`.
- Keep business logic out of client classes. Clients should translate data to/from external APIs and report clear failure modes.

## Local Commands

Set up:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run edge service:

```bash
./.venv/bin/python edge/main.py
```

Run via uvicorn when needed:

```bash
./.venv/bin/uvicorn edge.main:app --host 0.0.0.0 --port 8000
```

The browser dashboard is served from `/`; MJPEG stream is `/video`.

## Coding Standards

- Prefer small, focused modules that match existing boundaries.
- Use `logging.getLogger(__name__)` through `edge.logging_config.get_logger`; do not use `print`.
- Use typed dataclasses or explicit typed return objects for cross-module contracts.
- Keep edits scoped to the requested behavior.
- Never commit or hardcode `.env` secrets, Roboflow API keys, PMS credentials, or camera URLs.
- Do not change build, Docker, CI, or heavy/native dependencies unless the user explicitly approves.

## Harness Engineering Context

The repository now has a provider-independent harness slice in addition to the existing FastAPI camera runtime. Keep these boundaries separate:

- `edge/harness/`: run IDs, runtime execution, retries, checkpoint coordination, final run results, and structured event emission.
- `edge/loop/`: explicit loop state machine, planning, action selection, execution orchestration, verification, and stop states.
- `edge/context/`: per-iteration typed context assembly from trigger, repository instructions, history, capabilities, tools, and budget.
- `edge/prompts/`: provider-ready prompt/message construction only; no tool execution or state mutation.
- `edge/tools/`: provider-neutral tool metadata, result statuses, registry, executor, retry/idempotency policy.
- `edge/providers/`: provider interfaces and adapters. Generic code depends on provider protocols, not Roboflow, requests, MCP, or workflow-specific schemas.
- `edge/memory/`: durable checkpoints and stores. Do not serialize secrets, full env, raw base64, or large binary payloads.

The first vertical slice uses `edge.providers.roboflow.fake.FakeVisionWorkflowProvider`. It is deterministic and fake; never present its output as model inference.

Local harness example:

```bash
./.venv/bin/python -m edge.cli.run_fake_harness alerts/alert_track_2_20260526_172904_540897.jpg
```

Offline tests:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

Roboflow is one provider inside the harness. If Roboflow MCP is unavailable or a live run is blocked, continue developing and testing provider-independent harness code with fake/local adapters. Record live integration gaps in `docs/roboflow-integration-status.md`.

## Roboflow Workflow Integration Context

Requested workflow:

- Name: `EVN-Object-Detection vevn-object-detection-cnyo0-1-rfdetr-small-t1 Logic`
- Workspace slug: `les-workspace-ijdwd`
- Workflow slug: `evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`
- Serverless endpoint: `https://serverless.roboflow.com/les-workspace-ijdwd/workflows/evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`
- Declared user-known input: `image` of type `image`

Before implementing or changing Roboflow code, ground the integration through the Roboflow MCP server:

1. Call `workflows_get` with workflow id `evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`.
2. Read the exact workflow inputs, parameters, and outputs from the tool response.
3. Call `workflows_run` once with a representative image.
4. Treat the real response shape as source of truth. Workflow run results are expected to be a list with one dictionary per input image, keyed by the workflow own output names.
5. Do not hard-code output names before seeing the real workflow definition and sample response.

Current session status on 2026-07-19:

- Roboflow MCP `workflows_get` was called successfully.
- Published workflow definition source of truth:
  - Inputs: `image` with type `InferenceImage`.
  - Runtime parameters: none declared.
  - Outputs: `predictions` with type `JsonField`, selector `$steps.model.predictions`, coordinates system `own`.
  - Inner step: `roboflow_core/inner_workflow@v1`, child workflow `evn-object-detection-cnyo0`, model id binding `les-workspace-ijdwd/evn-object-detection-cnyo0-1-rfdetr-small-t1`.
- Roboflow MCP `workflows_run` was called with `https://media.roboflow.com/notebooks/examples/dog.jpeg`.
- The run did not produce a sample prediction response. Roboflow returned a server-side workflow configuration error: inner workflow step `model` binds unknown child input `model_id`; valid child inputs are `class_agnostic_nms`, `confidence`, `image`, `iou_threshold`, and `max_detections`.
- Integration code exists in `edge/clients/roboflow_workflow_client.py`, using `inference-sdk` and config from `edge/config.py`.
- Smoke test entrypoint: `smoke_roboflow_workflow.py`. It requires `ROBOFLOW_API_KEY` and asserts the configured expected output keys exist.

If the requested Roboflow path needs live video, webcam, RTSP, or stream processing, pause and ask the user. Roboflow live video workflows use a different path than single image workflow calls.
