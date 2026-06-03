# Software architecture and abstractions: contracts that keep model code, runtime code, and deployment code separable

## 1. Architectural thesis

The repository’s software architecture is organized around a simple thesis: **multimodal inference needs stable contracts between fast-changing model families and slow-changing serving infrastructure**.  Model authors need to add new graph topologies and tensor conversions quickly; systems engineers need scheduler, connector, deployment, profiling, and distributed execution code to remain reusable.

The main separation is:

- `vllm_omni/config/`: stage topology, deployment merge, scheduler selection, and pipeline registry.
- `vllm_omni/entrypoints/`: user-facing synchronous/online APIs and request lifecycle.
- `vllm_omni/engine/`: orchestration, stage startup, stage pools, inter-process communication, output messages, and remote registration.
- `vllm_omni/core/sched/`: scheduler extensions for omni states, chunk readiness, full-payload readiness, and KV transfer decisions.
- `vllm_omni/worker/`: model-runner mixins and platform-specific worker behavior.
- `vllm_omni/model_executor/models/`: model-family implementation and pipeline declarations.
- `vllm_omni/model_executor/stage_input_processors/`: semantic conversion functions between stages.
- `vllm_omni/diffusion/`: diffusion executors, clients, parallel config, offload, LoRA, cache acceleration, and profiling.
- `examples/`: runnable examples that exercise end-to-end flows for specific model families.

## 2. Configuration as a layered contract

The config system has three layers:

1. **Frozen topology** in `PipelineConfig` and `StagePipelineConfig`.  This describes what the model family is.
2. **Deployment config** in YAML.  This describes how the topology should run on available hardware.
3. **CLI/API overrides**.  These describe what the user wants to change for this invocation.

`build_stage_runtime_overrides()` handles `stage_<id>_<field>` overrides while avoiding orchestrator-only and shared fields.  `strip_parent_engine_args()` prevents top-level engine arguments from leaking into stages where they would conflict or be meaningless.  `_apply_diffusion_parallel_runtime_overrides()` moves diffusion-specific parallel settings into nested diffusion parallel config.

This layering is important because omni models are deployed in many shapes.  The same Qwen3-Omni graph might be run as a local three-stage pipeline for research, as a text-only Thinker service, or with separate replicas for Talker/Code2Wav in production.  A rigid config format would force model declarations to include hardware assumptions; this repo avoids that.

## 3. Lazy registry and out-of-tree extensibility

`pipeline_registry.py` centralizes built-in pipelines, but the comments also preserve `register_pipeline(config)` for out-of-tree plugins and tests.  This is the right engineering compromise:

- Built-in support is discoverable in one file.
- Pipeline modules are imported lazily, reducing import-time cost and dependency entanglement.
- External researchers can register experimental pipelines without modifying the central table.

The registry pattern is especially useful for fast-moving multimodal research.  Many new model families differ only in stage topology and processor hooks; forcing them into a monolithic `if model_type == ...` block would create fragile runtime code.

## 4. Stage metadata and engine initialization

Stage startup is handled by `vllm_omni/engine/stage_init_utils.py` and `AsyncOmniEngine`.

The engine extracts metadata, resolves model/tokenizer subdirectories, applies CLI tokenizer forwarding, resolves worker classes from worker type, builds vLLM configs for non-diffusion stages, builds diffusion configs for diffusion stages, and computes device/replica layouts.  This startup layer is where abstract stage declarations become concrete process plans.

Important engineering details include:

- **Subdirectory resolution.** Multi-component Hugging Face repos may store stage configs/tokenizers under subdirectories; `_resolve_model_tokenizer_paths()` handles `model_subdir` and `tokenizer_subdir` while falling back to the base model path for tokenizer resolution.
- **Worker class resolution.** `resolve_worker_cls()` maps a high-level `worker_type` such as `ar` or `generation` to platform-specific worker classes.
- **KV topology injection.** `_inject_inferred_kv_tp_topology()` infers adjacent tensor-parallel sizes and injects rank mapping when stages send/receive KV cache.
- **Device locks.** The startup utilities acquire and release device locks so multiple stages initializing on the same machine do not race memory-intensive setup.

The startup code is therefore the “compiler backend” for the pipeline DSL.

## 5. Entry points: keeping user APIs simple

The synchronous `Omni` class hides the stage graph from most users.  Users pass prompts and a list of sampling params; the class resolves params, computes request-level final stage, submits requests, and yields final outputs.  Example scripts under `examples/offline_inference/qwen3_omni/` show how multimodal prompt dictionaries include `multi_modal_data` for video, image, and audio, and how `Omni.generate()` returns text or audio depending on stage output type.

The entrypoint deliberately does not expose connector handles, scheduler states, or process-launch details.  That is a valuable API design lesson: a research engine can expose rich controls but still keep the default path close to model semantics—prompts in, multimodal outputs out.

## 6. Orchestrator and message-driven execution

`AsyncOmniEngine` communicates with the orchestrator through queues and typed messages in `vllm_omni/engine/messages.py`.  The message vocabulary includes request submission, stage submission, abort, shutdown, remote replica registration, collective RPC, and error reporting.

Message-driven design has several benefits:

- The caller thread is insulated from stage process failures.
- Local and remote stage engines can share a common control protocol.
- The orchestrator can record per-stage timing and output state without being coupled to model forward code.
- Future distributed deployments can evolve transport independently of the user API.

In multimodal systems, this indirection is not optional.  Diffusion stages, AR stages, and generation stages can have different process models and lifecycle requirements.

## 7. Worker mixins: keeping transfer out of model forward code

`OmniConnectorModelRunnerMixin` is one of the most important abstractions in the repo.  It centralizes connector creation, custom processor loading, async chunk bookkeeping, full-payload send/recv, KV-cache delegation, chunk registration, and rank-aware payload handling.

This is good software architecture because model forward implementations should not directly manage background I/O threads or scheduler readiness flags.  Instead:

- model runners call mixin methods;
- the mixin sends/receives through connectors;
- the mixin reports `OmniConnectorOutput`;
- schedulers consume readiness metadata;
- connectors remain swappable.

The result is a narrow waist between model execution and distributed transfer.  Researchers can add a new connector transport or a new stage processor without rewriting every model runner.

## 8. Schedulers as policy modules

`_resolve_scheduler()` in `stage_config.py` selects `OmniARScheduler`, `OmniARAsyncScheduler`, or `OmniGenerationScheduler` based on stage execution type and async scheduling.  The coordinator in `omni_scheduling_coordinator.py` manages extra states not present in vanilla vLLM scheduling.

This policy separation is central to maintainability:

- AR stages need token scheduling and possibly chunk-aware downstream readiness.
- Generation stages need a different output contract.
- Diffusion stages do not use vLLM’s token scheduler in the same way.
- Async chunk mode changes downstream readiness without changing model topology.

The repo therefore treats scheduling as a per-stage policy selected from metadata, not as a global engine behavior.

## 9. Diffusion subsystem as a parallel architecture

The diffusion subtree is extensive because diffusion serving has its own concerns:

- executor abstractions;
- process-backed and inline clients;
- parallelism configs;
- attention backends;
- cache acceleration such as TeaCache and Cache-DiT;
- LoRA management;
- offload hooks;
- profiling.

Rather than hiding diffusion behind a generic “postprocess image” function, vLLM-Omni elevates it to a peer subsystem.  The config layer can choose `StageExecutionType.DIFFUSION`, while the engine layer initializes diffusion-specific clients and configs.  This is the correct boundary for image/video generation models whose compute profile is closer to DiT serving than token decoding.

## 10. Engineering for observability

The entrypoint creates `OrchestratorMetrics` per request, records wall-clock start time, stage first timestamps, final-stage expectations, and logs summaries during cleanup.  The repo also includes `vllm_omni/profiler/`, diffusion profiling, and examples with `--profiler-stages` flags.

For multimodal inference, observability must be stage-aware.  Aggregate latency hides whether the bottleneck is multimodal preprocessing, AR prefill, Talker decoding, Code2Wav rendering, diffusion denoising, connector transfer, or queueing.  The code’s per-stage metrics design is therefore not just instrumentation; it is necessary for systems research.

## 11. Example-driven validation

The examples directory is not just user documentation.  It is a compatibility matrix of architectural patterns:

- `examples/offline_inference/qwen3_omni/end2end.py` exercises text, video, image, audio, mixed modalities, audio list inputs, and streaming-like async-chunk options.
- `examples/offline_inference/qwen2_5_omni/` exercises a predecessor Thinker/Talker stack.
- `examples/offline_inference/bagel/` exercises AR↔diffusion image behavior.
- `examples/offline_inference/text_to_speech/` covers many TTS model families with different intermediate representations.
- `examples/online_serving/` shows how the same stage abstractions are exposed through serving APIs.

Researchers should read examples after reading pipeline declarations.  The examples reveal which prompt schemas and sampling params the topology expects.

## 12. Failure containment and cleanup

A staged system has more failure modes than a single-process model:

- one stage may fail during initialization;
- a remote replica may not register;
- a connector may not deliver a chunk;
- a request may be aborted while downstream stages are waiting;
- a diffusion process may die independently of AR stages.

The engine layer includes failure callbacks, process termination helpers, abort messages, shutdown messages, and timeout-aware waiting states.  This is a core lesson for multimodal serving research: once computation is split across semantic stages, reliability becomes a graph problem, not a function-call problem.

## 13. Design patterns worth reusing

1. **Declarative topology, imperative runtime.** Keep graph facts in data classes; keep execution machinery generic.
2. **Dotted hooks for model-specific semantics.** Use import paths for processors so model families own tensor conversion.
3. **Per-stage execution type.** Select schedulers and clients by stage metadata.
4. **Per-request final stage.** Avoid executing unnecessary downstream modalities.
5. **Connector readiness as scheduler input.** Do not let schedulers call transports directly.
6. **Deployment merge layer.** Separate model topology from hardware layout.
7. **Stage-aware metrics.** Measure queueing, execution, and transfer per stage.
8. **Variant pipelines.** Represent AR-only, DiT-only, and full graphs as separate topology declarations.

These patterns are portable to other multimodal serving engines, especially those that must support rapidly evolving research models.

## 14. The “narrow waist” of the system

The repo has a useful narrow-waist architecture:

- Above the waist: user APIs, model-family declarations, examples, and deployment knobs.
- At the waist: stage metadata, messages, connector outputs, and processor hooks.
- Below the waist: vLLM engine cores, diffusion executors, GPU workers, transport implementations, and platform-specific workers.

A narrow waist is valuable because it lets both sides evolve.  Researchers can add new model families by declaring stages and processors; systems engineers can improve scheduling, transfer, or diffusion execution without changing every model family.  The cost is that the waist must be carefully specified.  In this repo, the specification is distributed across `StagePipelineConfig`, `StageConfig`, `StageMetadata`, `StageSubmissionMessage`, `OmniConnectorOutput`, and `OmniRequestOutput`.

## 15. Message structures as architectural documentation

`vllm_omni/engine/messages.py` is worth reading as a design document.  The message types define what the orchestrator believes can happen:

- `StageSubmissionMessage` means a request or streaming update is ready for a stage.
- `AddCompanionRequestMessage` means a hidden companion request is needed, commonly for CFG/KV scenarios.
- `AbortRequestMessage` and `ShutdownRequestMessage` define lifecycle control.
- `RegisterRemoteReplicaMessage` and `UnregisterRemoteReplicaMessage` define dynamic distributed membership.
- `OutputMessage` carries user-visible or stage-visible outputs with metrics and completion state.
- `StageMetricsMessage` decouples metrics emission from final output emission.
- `CollectiveRPCRequestMessage` and `CollectiveRPCResultMessage` provide a control-plane path for operations across stage replicas.

When extending the runtime, ask whether your feature fits an existing message.  If it does not, you are likely adding a new orchestration concept rather than only a model feature.

## 16. Object ownership: who is allowed to know what?

A reliable mental model is:

- **Pipeline declarations know model semantics** but should not know process handles, queues, or GPU locks.
- **Stage config factory knows how to merge topology and deployment** but should not run models.
- **Async engine knows how to start and connect stages** but should not interpret model-family tensors.
- **Orchestrator knows request graph state** but should not implement tensor conversion logic.
- **Stage pools know replica membership and load balancing** but should not know model architecture details.
- **Schedulers know readiness and capacity** but should not perform connector I/O.
- **Connector mixins know transfer mechanics** but should not decide global request routing.
- **Stage processors know tensor semantics** but should not manage queues or scheduling.

Most maintainability bugs violate one of these ownership rules.  For example, putting Qwen-specific tensor conversion into the orchestrator would make future audio models harder to add; making a scheduler poll a connector directly would make testing and transport replacement harder.

## 17. Code review heuristics for this repository

When reviewing a change, classify it by layer:

1. **Topology change:** Does it alter `PipelineConfig`, `StagePipelineConfig`, registry entries, or deploy YAML?  Verify stage IDs, input sources, final outputs, and processor paths.
2. **Processor change:** Does it alter intermediate representation?  Verify shape/dtype/device assumptions and async-vs-sync compatibility.
3. **Scheduler change:** Does it alter request states or capacity?  Verify no request can be stranded in a waiting state and no KV blocks are freed incorrectly.
4. **Connector change:** Does it alter payload transfer?  Verify request IDs, chunk indices, rank mapping, and cleanup semantics.
5. **Engine/orchestrator change:** Does it alter lifecycle or routing?  Verify abort, shutdown, metrics, remote replicas, and companion requests.
6. **Diffusion change:** Does it alter denoising, parallelism, offload, or LoRA?  Verify stage config conversion and memory behavior.

This checklist turns a large codebase into a set of local invariants.
