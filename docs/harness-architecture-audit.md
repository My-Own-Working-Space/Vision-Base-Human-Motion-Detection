# Harness Architecture Audit

Date: 2026-07-19

## Scope Inspected

- Source tree excluding `.git`, `.venv`, and Python cache directories.
- Applicable repository instructions: `docs/AGENTS.md` only.
- Runtime entrypoint: `edge/main.py`.
- Configuration: `edge/config.py`, `.env.example`, `requirements.txt`.
- External clients: `edge/clients/api_client.py`, `edge/clients/pms_bridge_client.py`, `edge/clients/roboflow_workflow_client.py`.
- Image-processing and inference entrypoints: `edge/services/detection_service.py`, `inference/yolo_detector.py`, `inference/mobilenetv3_classifier.py`, `inference/resnet_lstm_classifier.py`, `/api/analyze` in `edge/main.py`.
- Logging: `edge/logging_config.py` and call sites in services/clients.
- State and persistence: `edge/services/alert_service.py`, `alerts/metadata.csv`, `alerts/*.jpg`, `alerts/received/*.jpg`.
- Scripts and service definitions: `scratch_test.py`, `smoke_roboflow_workflow.py`, `Dockerfile.edge`, `docker-compose.yml`, `api.http`, `README.md`, `README.docker.md`.

## Current Directory Shape

```text
edge/
  clients/                  # Backend/PMS/Roboflow transport clients
  services/                 # Camera, detection, alert state machine
  ui/                       # MJPEG renderer and dashboard HTML
  config.py                 # Env -> frozen dataclasses
  logging_config.py         # Root logging setup
  main.py                   # FastAPI app, composition, workers, streaming loop
  models.py                 # Domain dataclasses/enums
inference/                  # YOLO, MobileNetV3, ResNet+LSTM model wrappers
alerts/                     # Runtime evidence images and CSV ledger
models/                     # Local model weights
docs/AGENTS.md              # Agent instructions and Roboflow context
```

## Applicable AGENTS.md Instructions

`docs/AGENTS.md` is the only `AGENTS.md` found. Its scope is documentation context rather than Python package-local rules. Important constraints:

- Keep `edge/main.py` as the composition root for the current edge runtime.
- Represent environment settings as frozen dataclasses in `edge/config.py`.
- Use `edge.logging_config.get_logger`; avoid `print` in application code.
- Keep clients transport-focused and do not move alert policy, CSV persistence, or inference details into clients.
- Do not hardcode secrets.
- Do not change Docker/build/heavy dependencies without approval.
- Roboflow is currently documented as a workflow integration, but the new harness architecture should treat it as one provider behind an interface.

## Current Execution Flow

### Live Stream Runtime

1. Importing `edge/main.py` mutates `sys.path`, loads `.env`, configures logging, loads config, and constructs global singleton services.
2. `DetectionService.initialize()` loads YOLO and optional classifier model at module import time.
3. `AlertService.start_retry_worker()` starts a background retry thread at module import time.
4. If PMS is enabled, `edge/main.py` starts a nested PMS registration/heartbeat thread.
5. FastAPI serves `/`, `/video`, `/api/alerts-history`, and `/api/analyze`.
6. `/video` calls `generate_frames()`, which opens the camera, reads frames, updates camera processing/sleep state, invokes detection, sends anomaly events to `AlertService`, annotates frames, and streams MJPEG bytes.
7. `AlertService` confirms anomalies, saves evidence images under `alerts/`, writes `alerts/metadata.csv`, sends the payload to the dashboard API, optionally forwards to PMS, and queues failed dashboard uploads for background retry.

### Ad Hoc Analysis Runtime

1. `/api/analyze` accepts an uploaded image or video.
2. It performs content-type checks, decodes the file, resizes frames, and calls the same global `DetectionService`.
3. It returns summarized JSON directly from the endpoint.

### External Workflow Smoke Runtime

`smoke_roboflow_workflow.py` loads config, constructs `RoboflowWorkflowClient`, runs a hardcoded HTTPS sample image, and asserts expected output keys. It requires `ROBOFLOW_API_KEY` and internet access.

## Current Module Responsibilities

- `edge/config.py`: centralized env parsing into frozen dataclasses. Also contains provider-specific Roboflow workflow settings.
- `edge/logging_config.py`: root logging and noisy logger suppression.
- `edge/models.py`: core event/payload/frame dataclasses.
- `edge/main.py`: FastAPI app, dependency construction, model initialization, background workers, streaming pipeline, upload analysis, shutdown.
- `CameraService`: camera lifecycle, FPS throttling, resize, processing/sleep state, motion watchdog.
- `DetectionService`: YOLO tracking plus classifier orchestration, domain event conversion, evidence selection.
- `AlertService`: anomaly confirmation, cooldown, evidence persistence, dashboard/PMS dispatch, retry queue.
- `ApiClient`: multipart alert upload to dashboard backend.
- `PmsBridgeClient`: PMS registration, heartbeat, and detection upload.
- `RoboflowWorkflowClient`: direct Roboflow workflow SDK wrapper, typed errors, retry/timeout, output validation.
- `StreamRenderer`: frame annotation and MJPEG encoding.
- `inference/*`: stateful model wrappers.

## Coupling and Hidden Dependencies

- `edge/main.py` performs heavy side effects at import time: model loading, retry worker startup, global object creation, and optional PMS worker startup.
- `DetectionService` imports model classes lazily but owns concrete YOLO/classifier implementations rather than a provider/tool interface.
- `AlertService` depends directly on concrete dashboard and PMS clients; upload retry policy is embedded inside alert policy.
- External clients return booleans or raw dicts, losing error categories and retry semantics at the harness boundary.
- `PmsBridgeClient` can leak file descriptors when an exception occurs before explicit close in `send_detection`.
- `scratch_test.py` imports real PMS client, reads `.env`, and can call a remote default URL when executed.
- `/api/analyze` mixes input validation, media decoding, frame sampling, inference, response shaping, and temporary file cleanup in one endpoint.
- Inference modules read `BLUR_THRESHOLD` directly from env in classifier constructors, bypassing `edge/config.py`.
- `sys.path` mutation is used in `edge/main.py` and `inference/resnet_lstm_classifier.py`.
- Runtime outputs (`alerts/metadata.csv` absolute paths) couple local machine paths into persisted data.
- Roboflow client currently sits under `edge/clients/` as a concrete provider client rather than a provider-neutral adapter.

## Missing Abstractions

- No harness runtime abstraction for run IDs, deadlines, cancellation, checkpoints, resumability, or final run results.
- No generic tool abstraction with metadata, side-effect classification, idempotency, timeout, retry policy, redaction rules, or capability requirements.
- No provider-neutral vision workflow interface.
- No structured tool result taxonomy for retryable failure, permanent failure, invalid input, capability unavailable, timeout, or cancellation.
- No explicit loop state machine for trigger -> goal -> plan -> action -> verification -> memory/checkpoint -> stop/escalate.
- No typed per-iteration context object; context is implicit in globals and service state.
- No prompt construction boundary. This repo does not yet include agent prompt code, but future prompt code needs to be isolated from tool execution and state mutation.
- No durable checkpoint store separate from alert CSV persistence.
- No capability snapshot concept for MCP or provider availability.
- No event model for structured execution traces.

## Failure and Recovery Weaknesses

- Dashboard upload retry is local to `AlertService`, uses an in-memory deque, and is lost on process restart.
- PMS registration/heartbeat retry runs forever with broad exception logging and no checkpoint/state observability.
- External client failures are flattened to `False` or `None`, so callers cannot distinguish invalid input, timeout, network failure, capability gap, or provider bug.
- Many broad `except Exception` blocks log and continue without attaching run/action IDs or recovery categories.
- `DetectionService.process_frame()` returns `[]` if uninitialized, which can hide configuration/model-load failure.
- `/api/analyze` returns exception text to clients, which can expose internal details.
- Config parsing can raise raw `ValueError` on malformed env values during import.
- Roboflow smoke depends on internet and a secret, so it is not suitable as a normal test.
- No cancellation propagation exists for model inference, HTTP calls, or video processing.

## Testability Weaknesses

- No formal test suite exists; only `scratch_test.py` and `smoke_roboflow_workflow.py` scripts.
- Heavy side effects at import time make `edge.main` difficult to import in tests.
- Real model loading and camera capture are not abstracted behind test doubles.
- External services are not behind provider-neutral protocols.
- Retry behavior is embedded in services and hard to test deterministically without sleeping.
- No local fake provider exists for Roboflow or generic vision workflow behavior.
- Runtime state is spread across service instances, threads, CSV, and globals, limiting checkpoint/resume tests.

## Security and Configuration Risks

- `.env` is gitignored, but `.env.example` must remain secret-free.
- Roboflow API keys are correctly environment-loaded but the concrete client is provider-specific and not redaction-aware.
- `alerts/metadata.csv` persists absolute local paths, which can disclose usernames and directory structure.
- Logging sometimes includes response bodies and exception strings without centralized redaction.
- `scratch_test.py` contains a hardcoded public PMS URL fallback and should not be treated as a safe automated test.
- Prompt/harness logs must not serialize API keys, full env, large image payloads, or base64 outputs.

## Existing State and Persistence

- Alert evidence images live under `alerts/` and `alerts/received/`.
- `alerts/metadata.csv` is the only durable application ledger. It tracks timestamp, track ID, class, confidence, coordinates, image path, and upload status.
- There is no durable run state, checkpoint store, execution trace, retry ledger, or domain memory.

Representative alert fixtures:

- `alerts/alert_track_2_20260526_172904_540897.jpg`: JPEG 640x360.
- `alerts/alert_track_5_20260617_105930_583264.jpg`: JPEG 640x360.
- `alerts/alert_track_17_20260630_110752_085888.jpg`: JPEG 640x360.
- `alerts/received/alert_track_123_20260616_182638_948199.jpg`: JPEG 200x200.

## Components Suitable for Reuse

- `edge/config.py` dataclass style should be retained and extended for harness settings.
- `edge/logging_config.get_logger` can be reused as the base logger, with a structured event emitter added beside it.
- `edge/models.py` domain event types can remain for the current surveillance pipeline.
- `CameraService`, `DetectionService`, `AlertService`, `ApiClient`, `PmsBridgeClient`, and `StreamRenderer` should remain for backwards-compatible edge service behavior.
- `inference/*` should remain model-specific black boxes and later be wrapped as providers/tools when live model harnessing is needed.
- Representative images under `alerts/` are suitable local fixtures for fake-provider harness tests, without modifying the originals.
- The Roboflow client implementation contains useful retry/timeout/response validation ideas, but should be wrapped/migrated behind `edge/providers/roboflow/` rather than being a dependency of generic harness code.

## Architecture Direction

The repository should add a provider-independent harness stack alongside the existing edge service instead of rewriting the streaming pipeline in one step. The first vertical slice should:

1. Accept a local image trigger.
2. Validate file existence and allowed image type.
3. Create run and iteration IDs.
4. Assemble a typed iteration context from durable state and capability snapshots.
5. Select a vision workflow action.
6. Execute through a generic tool executor with retry policy.
7. Use a deterministic fake vision provider.
8. Verify typed workflow output.
9. Save a checkpoint without secrets or binary payloads.
10. Emit structured JSON-safe events.
11. Return a typed final result.

Roboflow should remain an adapter capability, not an architectural prerequisite. Live Roboflow MCP/schema work can be tracked as a capability gap while the generic harness is implemented and tested offline.
