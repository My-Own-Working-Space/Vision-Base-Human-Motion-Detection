# Harness Refactor Plan

Date: 2026-07-19

## Proposed Directory Tree

```text
edge/
  harness/
    __init__.py
    runtime.py
    models.py
    errors.py
    events.py
    retry.py
    checkpoint.py
    logging.py
  loop/
    __init__.py
    controller.py
    state.py
    planner.py
    actions.py
    verifier.py
    policies.py
  context/
    __init__.py
    assembler.py
    models.py
    repository_context.py
    history.py
    retrieval.py
    capabilities.py
  prompts/
    __init__.py
    builder.py
    templates.py
    schemas.py
  tools/
    __init__.py
    base.py
    registry.py
    executor.py
    results.py
    policies.py
  providers/
    __init__.py
    base.py
    roboflow/
      __init__.py
      interface.py
      adapter.py
      models.py
      errors.py
      fake.py
  memory/
    __init__.py
    store.py
    models.py
    checkpoint_store.py
    in_memory.py
    file_store.py
  cli/
    __init__.py
    run_fake_harness.py
```

Test tree:

```text
tests/
  __init__.py
  test_harness_vertical_slice.py
  test_tool_executor.py
  test_checkpoint_and_resume.py
  test_redaction_and_state.py
```

Runtime artifact tree:

```text
harness_runs/             # ignored; checkpoints and traces for local harness runs
```

## Mapping From Current Modules

- `edge/config.py`: retain existing edge runtime config. Add harness config only if the slice needs defaults; avoid moving current config fields now.
- `edge/logging_config.py`: retain. Add `edge/harness/logging.py` for structured event emission and redaction.
- `edge/models.py`: retain for surveillance pipeline. New harness/provider models live in dedicated packages to avoid cross-domain coupling.
- `edge/clients/api_client.py`: retain. Future wrapper can expose dashboard upload as a `Tool` or provider operation.
- `edge/clients/pms_bridge_client.py`: retain. Future wrapper should classify PMS failures and close files safely.
- `edge/clients/roboflow_workflow_client.py`: retain as compatibility code. Future `edge/providers/roboflow/adapter.py` should wrap it or replace it with MCP/SDK-specific code behind `VisionWorkflowProvider`.
- `edge/services/*`: retain current behavior. Later migration can express detection and alert dispatch as tools or providers.
- `inference/*`: retain as model-specific black boxes. Later wrap as local vision providers.
- `scratch_test.py`: leave untouched, but do not use as harness test.
- `smoke_roboflow_workflow.py`: leave as live integration smoke script; not part of offline harness tests.

## Public Interfaces

### Provider Interface

```python
class VisionWorkflowProvider(Protocol):
    def describe_workflow(self, workflow_ref: str) -> WorkflowDescription: ...
    def run_workflow(self, request: WorkflowRunRequest) -> WorkflowRunResult: ...
```

Internal models:

- `WorkflowDescription`: provider name, workflow ref, declared inputs, parameters, outputs, capability status.
- `WorkflowRunRequest`: workflow ref, image path, parameters, idempotency key.
- `WorkflowRunResult`: provider name, workflow ref, outputs, artifacts, fake flag.

These are internal provider-neutral types and do not claim any Roboflow MCP schema.

### Tool Interface

```python
class Tool(Protocol):
    metadata: ToolMetadata
    def execute(self, request: ToolRequest) -> ToolResult: ...
```

`ToolMetadata` includes name, description, timeout, retry safety, side effects, idempotency, required capability, and redaction fields.

`ToolResult` has explicit statuses: success, retryable failure, permanent failure, invalid input, capability unavailable, timeout, cancelled.

### Harness Runtime Interface

```python
class HarnessRuntime:
    def start_run(self, trigger: LocalImageTrigger) -> RunResult: ...
    def resume_run(self, run_id: str) -> RunResult: ...
```

The runtime owns run IDs, deadlines, loop controller invocation, checkpoint writes, structured events, and final result construction.

### Loop Interface

```python
class LoopController:
    def run(self, state: LoopRunState) -> LoopRunState: ...
```

The loop uses explicit states and validates transitions. It delegates context assembly, planning, tool execution, verification, memory/checkpoint updates, and stop conditions.

### Context Interface

```python
class ContextAssembler:
    def assemble(self, request: ContextAssemblyRequest) -> IterationContext: ...
```

Context is rebuilt per iteration from durable state and capability snapshots.

### Prompt Interface

```python
class PromptBuilder:
    version: str
    def build(self, context: IterationContext) -> PromptBundle: ...
```

Prompt code has no side effects beyond message construction.

## Migration Sequence

1. Add documentation: audit and this plan.
2. Add provider-neutral models and fake provider.
3. Add tool metadata/result abstractions and generic executor with retry policy.
4. Add memory/checkpoint models and file/in-memory stores.
5. Add structured event emitter with redaction.
6. Add context assembler for local image triggers.
7. Add loop state machine, planner, action selector, verifier, and policies for one vertical slice.
8. Add harness runtime that composes the above.
9. Add CLI/example runner using a local image path.
10. Add offline tests for success, failures, retries, checkpointing, resume, redaction, and state transitions.
11. Check Roboflow MCP readiness and record status without blocking the fake-provider harness.

## Compatibility Approach

- Do not change `edge/main.py` behavior in this slice.
- Do not replace current alert or detection services.
- Keep existing clients in `edge/clients/` importable.
- Add new package boundaries beside existing code.
- Use fake provider in tests and examples to avoid network, API keys, MCP, or model weights.
- Put runtime checkpoints in an ignored `harness_runs/` directory.

## Testing Strategy

Use standard-library `unittest` to avoid introducing a new test framework. Tests must run offline.

Required coverage:

- Successful end-to-end fake run.
- Missing image.
- Unsupported image type.
- Retryable tool failure followed by success.
- Retry exhaustion.
- Capability unavailable.
- Invalid provider response.
- Verification failure.
- Checkpoint creation.
- Resume from checkpoint.
- Secret redaction.
- State-transition validity.
- Prevention of duplicate execution for non-idempotent actions.

## Risks

- `inference-sdk` changed the local OpenCV version constraint; keep this visible in docs and avoid further heavy dependency changes.
- The current `docs/` directory is untracked in this worktree, so new docs may appear as untracked until committed.
- Import-time side effects in `edge/main.py` remain a risk but are intentionally deferred.
- File checkpoint schemas need versioning from the start to avoid future migration ambiguity.
- Fake provider results must be clearly marked fake to avoid confusing them with actual model inference.

## Deferred Roboflow Work

- Inspect live Roboflow MCP tool descriptions when available at Phase 6.
- Keep raw Roboflow MCP/SDK payloads inside `edge/providers/roboflow/adapter.py`.
- Convert Roboflow payloads immediately into internal provider-neutral dataclasses.
- Add adapter contract tests once real schemas are known.
- Do not invent request/response schemas if MCP is unavailable.
- The published workflow currently has a known server-side configuration error: inner step binds `model_id`, while valid child inputs are `class_agnostic_nms`, `confidence`, `image`, `iou_threshold`, and `max_detections`.

## Definition of Done For This Slice

- Audit document exists.
- Refactor plan exists.
- Provider-independent harness vertical slice runs on a local image using a deterministic fake provider.
- Generic tool executor handles retry policy and structured result statuses.
- Checkpoint is saved without secrets or binary payloads.
- Structured events are emitted with redaction.
- Offline tests cover the required scenarios.
- Roboflow integration status is documented separately.
- Existing edge service modules remain import-compatible.
