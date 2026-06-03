# Request lifecycle: from user prompt to final multimodal output

## 1. Why lifecycle tracing matters

A staged multimodal request crosses many boundaries: Python user API, sampling params, prompt preprocessing, engine queues, orchestrator state, stage pools, scheduler states, model runners, connectors, output processors, and metrics.  Bugs often appear as “no output” or “wrong modality,” but the root cause may be an early final-stage decision, a processor hook, a waiting-state transition, or a missing connector signal.

This document follows the lifecycle from `Omni.generate()` to final `OmniRequestOutput`.

## 2. Initialization before any request

`OmniBase.__init__()` performs the common setup for synchronous and asynchronous entrypoints:

1. Resolve/download the model snapshot if needed.
2. Store user-level flags such as `log_stats`, `output_modalities`, `async_chunk`, and TTS batch settings.
3. Construct `OmniTransferMetrics` before creating the engine so the orchestrator can emit transfer metrics.
4. Create `AsyncOmniEngine`, which resolves stage configs, launches stages, constructs stage pools, and caches stage metadata.
5. Mirror the authoritative resolved `async_chunk` value from the engine.
6. Create request-state dictionaries and Prometheus/modality metrics.
7. Load default per-stage sampling params from the engine.
8. Build a local list of stage metadata.
9. Initialize prefill/decode disaggregation state if configured.

The key lesson is that the entrypoint is already graph-aware before request submission.  It knows the number of stages, their final output types, and default sampling params.

## 3. `Omni.generate()` normalizes the call

`Omni.generate()` accepts a single prompt or a sequence of prompts and either one sampling params object or per-stage sampling params.  It expands params for PD disaggregation when necessary, resolves missing params from defaults, and chooses between list-return mode and Python generator mode.

Before sending work to the engine, `_set_final_only_for_llm_stages()` forces LLM stages to use final-only output.  This prevents intermediate token streams from every LLM stage from leaking to the user in ordinary offline generation.  Diffusion stages are not treated the same because their output contract is not token streaming.

## 4. Request IDs and final-stage IDs

Inside `_run_generation()`, each prompt receives a unique request ID.  The entrypoint inspects prompt modalities and calls `_compute_final_stage_id()`.  That final-stage ID is stored in request metrics and sent to the engine.

This is one of the most important lifecycle values.  It determines whether a request should stop at a text stage or continue into audio/image stages.  If final-stage computation is wrong, the system may waste work by running unnecessary downstream stages or fail to produce the requested modality.

## 5. Engine submission message

`AsyncOmniEngine.add_request()` builds a `StageSubmissionMessage`.  The message includes:

- request ID;
- prompt or `EngineCoreRequest`;
- original prompt;
- output prompt text;
- sampling params for all stages;
- final stage ID;
- preprocessing latency;
- enqueue timestamp;
- message type (`add_request` or `streaming_update`).

The message is placed on the request queue consumed by the orchestrator.  At this point, the user thread no longer directly controls stage execution; it polls output queues.

## 6. Orchestrator intake and stage-0 submission

The orchestrator receives the submission, tracks request state, and submits the request to the appropriate Stage 0 pool.  Stage pools abstract over one or more local or remote replicas.  In distributed mode, load balancing and affinity determine which replica receives a request.  For local mode, the pool wraps local stage clients.

If prompt expansion is configured for CFG/KV workflows, the orchestrator can enqueue companion requests.  These companion requests are not user-visible outputs; they exist to produce matching cache/conditioning structures for downstream stages.

## 7. Stage execution and output polling

The orchestrator loop polls every live replica in every stage pool.  Diffusion stages are polled through diffusion output paths; LLM stages are polled for raw engine outputs.  Raw outputs may trigger KV-ready handling before being processed into logical stage outputs.

This loop is the central runtime heartbeat.  It decides whether an output should:

- emit final output to the user;
- emit only metrics;
- forward a transformed prompt/payload to a downstream stage;
- update KV readiness;
- register or satisfy chunk/full-payload waiting states;
- clean up finished request state.

## 8. Forwarding to the next stage

When a stage output is not the request’s final output, the orchestrator forwards work to downstream stages.  Forwarding can involve:

- applying a stage output processor;
- preserving original multimodal prompt data for stages that require it;
- building `EngineCoreRequest` objects from token IDs or processed payloads;
- selecting the correct sampling params for the next stage;
- respecting final-stage ID and early termination;
- emitting transfer metrics.

The central conceptual move is that downstream stage input is not necessarily the upstream stage’s user-visible output.  It may be a latent object, hidden state, token ID list, KV cache, CFG companion data, or a multimodal payload.

## 9. Receiving outputs in the entrypoint

The user-facing loop calls `self.engine.try_get_output()` repeatedly.  It handles `None` by continuing, handles error messages by raising, updates request metrics, and calls `_process_single_result()` to decide whether an output should be yielded.  When an `OutputMessage` has `finished=True`, the request is removed from the active set and summary cleanup runs.

The entrypoint’s output handling is intentionally decoupled from stage internals.  It relies on messages, `OmniRequestOutput`, stage metadata, and metrics rather than direct access to workers.

## 10. Abort and cleanup lifecycle

If an exception occurs while active requests exist, `Omni._run_generation()` aborts them.  `OmniBase` also installs a weak finalizer so engine shutdown occurs on garbage collection.  Explicit `close()`/shutdown paths are important because staged serving can leave background threads, subprocesses, queues, connectors, and GPU memory alive.

A single-stage LLM script might survive sloppy cleanup.  A staged omni runtime often will not.

## 11. Lifecycle invariants

The lifecycle is correct only if these invariants hold:

1. Every request ID is stable across stages and connector payloads.
2. Every stage uses the sampling params intended for that stage.
3. The final-stage ID is monotonically respected; no stage beyond it should be required.
4. Downstream stages are not scheduled until required input is available.
5. Companion requests are tied to parent requests and cleaned up correctly.
6. Metrics distinguish stage outputs from final user outputs.
7. Abort/shutdown reaches all relevant stage pools and waiting states.

When debugging, identify which invariant was violated.
