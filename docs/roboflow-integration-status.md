# Roboflow Integration Status

Date: 2026-07-19

## Tools Searched For

Tool discovery query:

- `Roboflow workflows_get workflows_run workflow MCP`

## Tools Found

The Roboflow MCP namespace is available in this session. Relevant tools exposed:

- `workflows_get`: get details for a saved workflow.
- `workflows_run`: execute a saved workflow on one or more HTTPS URL or base64 images.
- `workflow_specs_run`: execute an inline workflow specification.
- `workflows_create`, `workflows_update`, `agent_workflow_publish`: mutating workflow tools, not used for this read-only readiness check.
- `agent_chat`: Roboflow Q&A/workflow planning, not needed for this provider-independent slice.

## Read-Only Workflow Description Result

`workflows_get` was called for workflow id:

`evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`

Observed workflow definition:

- Name: `EVN-Object-Detection vevn-object-detection-cnyo0-1-rfdetr-small-t1 Logic`
- URL slug: `evn-object-detection-vevn-object-detection-cnyo0-1-rfdetr-small-t1-logic`
- Document id: `EeryC4hKSXiZ0faQAhlM`
- Version: `1.0`
- Inputs: one `InferenceImage` named `image`
- Runtime parameters: none declared as top-level workflow inputs
- Steps: one `roboflow_core/inner_workflow@v1` step named `model`
- Inner workflow workspace: `les-workspace-ijdwd`
- Inner workflow id: `evn-object-detection-cnyo0`
- Parameter bindings:
  - `image`: `$inputs.image`
  - `model_id`: `les-workspace-ijdwd/evn-object-detection-cnyo0-1-rfdetr-small-t1`
- Outputs: one `JsonField` named `predictions`, selector `$steps.model.predictions`, coordinate system `own`

These are observations from Roboflow MCP and should be mapped inside `edge/providers/roboflow/adapter.py` when the live adapter is implemented.

## Minimal Run Attempt

A minimal `workflows_run` attempt was made using a small representative local image encoded as base64:

- Source fixture: `alerts/received/alert_track_123_20260616_182638_948199.jpg`
- File type: JPEG 200x200

The action was rejected by policy because uploading a local alert image from the workspace to an external Roboflow service could disclose sensitive repository/runtime data. This repository must not attempt to bypass that restriction.

A prior public-image run against the same published workflow returned a Roboflow server-side configuration error: the inner workflow step `model` binds unknown child input `model_id`; valid child inputs were reported as `class_agnostic_nms`, `confidence`, `image`, `iou_threshold`, and `max_detections`.

## Missing Capabilities / Gaps

- Live Roboflow run validation using local representative images is blocked by data-exfiltration policy.
- The published Roboflow workflow appears misconfigured server-side until the `model_id` child input binding is fixed or the workflow is republished.
- No live Roboflow adapter contract tests are added yet because the adapter must be implemented against the real schema while respecting data-handling policy.

## Code Complete Without MCP

The provider-independent harness is complete and tested offline:

- `edge/harness/`
- `edge/loop/`
- `edge/context/`
- `edge/prompts/`
- `edge/tools/`
- `edge/providers/base.py`
- `edge/providers/roboflow/fake.py`
- `edge/memory/`
- `edge/cli/run_fake_harness.py`
- Offline tests in `tests/`

The fake provider is deterministic and marks results as fake. It does not represent real Roboflow inference.

## Remaining Adapter Work

1. Implement `edge/providers/roboflow/adapter.py` against the allowed Roboflow path.
2. Keep raw MCP or SDK payloads inside the adapter boundary.
3. Convert real workflow descriptions into internal `WorkflowDescription`.
4. Convert real workflow run results into internal `WorkflowRunResult`.
5. Add adapter contract tests using non-sensitive fixtures or mocked MCP payloads.
6. If local alert images are needed for live validation, get an explicit approved data-handling path first; do not upload workspace alert images by default.
7. Re-run the published workflow only after the server-side `model_id` binding issue is resolved.

## How To Validate Once Safe

- Use a non-sensitive public HTTPS test image or an approved synthetic fixture.
- Call `workflows_get` first and assert the mapped internal description.
- Call the live adapter on the approved image.
- Assert the internal result contains the real output keys from the workflow description.
- Ensure logs/checkpoints contain only internal typed summaries, not raw base64 images or API keys.
