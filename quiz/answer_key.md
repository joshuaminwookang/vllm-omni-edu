# Answer key and grading notes

These are suggested answers.  Good student answers may use different wording if they identify the correct subsystem, invariant, and code path.

## Quiz 01 — Codebase map

1. B — `vllm_omni/config/`.
2. A — `vllm_omni/model_executor/stage_input_processors/`.
3. B — stage/runtime contracts provide the mental map.
4. B — `engine/` orchestrates stage startup/routing/messages; `worker/` owns model-runner and connector behavior.
5. Model declaration path: `pipeline_registry.py` → model-family `pipeline.py` → `stage_config.py` → `stage_init_utils.py` → `async_omni_engine.py`. Request execution path: `omni.py`/`omni_base.py` → `async_omni_engine.py` → `orchestrator.py` → `stage_pool.py` → workers/connectors → output messages.
6. Contracts: `StagePipelineConfig` as structural contract; messages as orchestration contract; connector output as scheduler contract. Violations can cause invalid graph materialization, unrouteable lifecycle events, or scheduler hangs/spurious scheduling.
7. Inspect transfer/readiness first because downstream input depends on processor hooks, connector readiness, and scheduler waiting-state transitions.
8. Read `pipeline_registry.py`, open the family `pipeline.py`, inspect `StagePipelineConfig` fields, inspect processor hooks, then trace `StageConfigFactory` merge/materialization.
9. Relevant directories: `metrics/`, `engine/`, `worker/`, `distributed/omni_connectors/`, and `core/sched/`, because transfer metrics depend on connector events and orchestrator forwarding.
10. Strong answer: model-family code declares graph topology and payload semantics; runtime code schedules, launches, routes, transfers, and finalizes requests through generic abstractions.

## Quiz 02 — Pipeline configuration and architecture support

1. B.
2. B.
3. B.
4. A.
5. A.
6. `StagePipelineConfig` defines one stage’s structural semantics; `PipelineConfig` groups stages and detection metadata for a model family. Both are needed because one describes a node and the other describes the graph/family.
7. Sampling constraints define whether intermediate outputs remain token IDs, are detokenized, or stop at model-specific IDs; wrong constraints can corrupt downstream payloads.
8. Detection uses HF config/model type, raw config fallback, nonstandard architecture fields, diffusers model index, architecture matching, predicates, and name fallback. This handles inconsistent multimodal repos.
9. Defaults can accidentally override deploy YAML or leak orchestrator-only fields into stage-local configs; explicitness matters.
10. It compiles model-family graph data plus deploy/CLI data into concrete `StageConfig`, scheduler, runtime, engine-args, and metadata objects.
11. Use `execution_type`, `input_sources`, `custom_process_*` hooks, `prompt_expand_func`/`cfg_kv_collect_func` as needed, `omni_kv_config` with transfer criteria, sampling constraints, final-output fields, deploy connector/TP settings.
12. Declare async and sync processor variants and let resolved `async_chunk` select the appropriate hooks through config logic.
13. Add registry entry with `hf_architectures` and possibly `hf_config_predicate`; avoid relying only on generic `model_type`.

## Quiz 03 — Request lifecycle and staged execution

1. A.
2. B.
3. A.
4. B.
5. B.
6. Strong trace: `Omni.generate()` normalizes params and final-stage ID → `AsyncOmniEngine.add_request()` builds `StageSubmissionMessage` → orchestrator submits to Stage 0 via `StagePool` → worker executes and connectors/processors handle intermediate transfer → orchestrator emits or forwards outputs → entrypoint receives `OutputMessage` and returns `OmniRequestOutput`.
7. Wrong final-stage computation can skip required stages, run unnecessary stages, emit wrong modality, or waste GPU/transfer resources.
8. Companion requests support CFG/KV alignment and hidden conditioning branches required by downstream diffusion stages.
9. Components: preprocessing, stage queueing, stage compute, transfer wait, transfer bandwidth/serialization, finalization. End-to-end hides the bottleneck.
10. Request IDs must remain stable across stages/connectors; sampling params must match the intended stage semantics.
11. Inspect final-stage routing: prompt modalities, `_compute_final_stage_id()`, stage final-output metadata, and request submission.
12. Metrics can be emitted for internal stage work even if no final user-visible output has been produced; inspect forwarding/final-stage handling.
13. Staged runtimes have background threads/processes/connectors/GPU memory; abort/close prevents orphaned work and leaks.

## Quiz 04 — Scheduler and connector internals

1. A.
2. A.
3. B.
4. B.
5. A.
6. Full payload: complete object handoff, useful for AR latent to diffusion. Async chunk: incremental streaming, useful for speech. KV transfer: cache/state handoff, useful for AR→DiT or cache-conditioned stages.
7. Chunk belongs to the same request, readiness is reported, terminal/intermediate status is correct, waiting state is cleared, and scheduler capacity is respected.
8. KV tensors are sharded differently across TP ranks; transfer requires rank-aware keying, slicing, merging, and topology injection.
9. It ensures full-payload producer behavior is enabled only when the stage structurally declares the needed downstream producer hook and mode.
10. It needs sets/handles indicating chunks ready/finished, full payload received, KV readiness, and request IDs to register for receive.
11. Check final-stage path, input sources, async mode, processor paths, connector init logs, producer payload emission, `OmniConnectorOutput`, coordinator state transition, scheduler capacity, and abort/finish state.
12. Inspect chunk index ordering, request ID matching, terminal chunk handling, processor chunk semantics, and payload merge logic.
13. The scheduler/coordinator readiness layer likely admitted the request too early or connector readiness was reported incorrectly.

## Quiz 05 — Diffusion and generation stages

1. A.
2. A.
3. A.
4. A.
5. `LLM_AR`: token/hidden autoregressive work; `LLM_GENERATION`: generation-like non-text renderer such as audio; `DIFFUSION`: denoising/image/video/audio tensor generation with diffusion clients/executors.
6. AR→diffusion may pass KV cache, hidden conditioning, CFG companion data, latent tensors, and original multimodal state; it is richer than caption text.
7. They isolate/debug subgraphs, benchmark stage costs independently, and support partial deployment.
8. Sequence/ring/Ulysses parallelism, CFG parallelism, VAE patch parallelism, offload, LoRA management, denoising cache acceleration, large activation memory.
9. Time to first audio, code generation latency, waveform rendering latency, transfer wait, chunk cadence, audio quality, and memory.
10. Ask whether to isolate diffusion, use offload, adjust parallelism, measure peak memory, transfer latency, queueing, and OOM rate.
11. Inspect denoising loop, diffusion scheduler, attention backend, VAE, offload/cache acceleration, image resolution, and per-step/stage profiling.

## Quiz 06 — Extending and testing model integrations

1. A.
2. B.
3. A.
4. A.
5. Describe graph, implement/wrap stages, write processors, declare pipeline, register/detect, add deploy defaults, examples, and tests.
6. Input types, required keys, shapes, dtypes, devices, request IDs, chunk semantics, empty/malformed behavior, and output schema.
7. Device placement is deployment-specific; hard-coding it in topology makes the model graph less portable.
8. Narrow waist: `StagePipelineConfig`, `StageConfig`, `StageMetadata`, `StageSubmissionMessage`, `OmniConnectorOutput`, `OmniRequestOutput`, and engine messages separate model declarations from runtime mechanisms.
9. Examples: generic orchestrator with model-specific conversions, scheduler calling connectors directly, device placement in pipeline, wrong detokenization defaults, assuming all requests traverse all stages.
10. Topology changes alter graph/stage declarations; scheduler changes alter request state/capacity policy and can create hangs or incorrect admission.
11. Request robust detection metadata (`hf_architectures`/predicate) to avoid misrouting generic `model_type` checkpoints.
12. Call out device mismatch, copy overhead, nondeterministic behavior, downstream assumptions, and missing ABI docs/tests.
13. Request text-only, multimodal inputs, batch, async/full-payload mode if supported, final-output routing tests, and processor/config tests.

## Quiz 07 — Research implications, observability, and experiments

1. A.
2. A.
3. A.
4. A.
5. Include text-only, audio-output, image-output, mixed modality input, multi-output/branching, streaming updates, and heterogeneous deployment workloads.
6. Report AR queue/compute, diffusion queue/compute, transfer wait/volume, KV/cache size, first latent/image timing, peak memory, quality, and utilization.
7. Cache placement involves correctness, lifetime, bandwidth, topology, heterogeneity, cancellation, and memory pressure across stages.
8. Include graph diagram, workload, deployment, per-stage metrics, transfer/memory analysis, correctness/quality, failure cases, code changes, and lessons.
9. Adaptive chunking could tune chunk size/window based on load or SLOs; it interacts with processors, connector outputs, scheduler coordinator, and metrics.
10. Example: degrade to text-only or lower-resolution output when downstream overloaded while preserving request IDs, final-output semantics, user-visible errors, cleanup, and correctness.
11. Strong capstone: define Thinker as text final, Talker/Code2Wav as audio branch, DiT as image branch; replicate bottleneck stages based on measured service times; try async chunks for speech, KV/full payload for diffusion; measure first-output, per-stage queue/compute, transfer, memory, quality, skipped work, and overload behavior.
