# Codebase map: where the important abstractions live

## 1. Why start with a map?

vLLM-Omni is not a small “model wrapper.”  It is a multimodal serving system built around staged execution, vLLM integration, diffusion execution, distributed connectors, model-family processors, and deployment-time graph synthesis.  Students often get lost because they begin with a model file and then jump directly into scheduler internals.  A better strategy is to first identify the layers and the contracts between them.

The central organizing idea is:

> model-family code declares **what** stages exist and how adjacent stages transform data; runtime code decides **when**, **where**, and **how** those stages execute.

## 2. Top-level directories to know

| Directory | What it teaches | Typical files to open first |
| --- | --- | --- |
| `vllm_omni/config/` | Stage graph declarations, deploy merge, scheduler selection, model detection | `stage_config.py`, `pipeline_registry.py`, `model.py`, `yaml_util.py` |
| `vllm_omni/entrypoints/` | User-facing APIs and request lifecycle | `omni.py`, `omni_base.py`, `utils.py`, `cli/serve.py` |
| `vllm_omni/engine/` | Orchestration, stage startup, message routing, stage pools, remote replicas | `async_omni_engine.py`, `orchestrator.py`, `stage_init_utils.py`, `stage_pool.py`, `messages.py` |
| `vllm_omni/core/sched/` | Omni-aware scheduler policy and waiting-state coordination | `omni_ar_scheduler.py`, `omni_generation_scheduler.py`, `omni_scheduling_coordinator.py` |
| `vllm_omni/worker/` | GPU/model runners, connector mixin, platform-specific execution | `omni_connector_model_runner_mixin.py`, `gpu_model_runner.py`, `gpu_ar_model_runner.py`, `gpu_generation_model_runner.py` |
| `vllm_omni/distributed/` | Connector implementations, KV transfer, coordinator/load balancing | `omni_connectors/`, `omni_coordinator.py` |
| `vllm_omni/model_executor/models/` | Model-family implementations and per-family `pipeline.py` files | `qwen3_omni/pipeline.py`, `bagel/pipeline.py`, `hunyuan_image3/pipeline.py` |
| `vllm_omni/model_executor/stage_input_processors/` | Tensor/payload ABI between stages | `qwen3_omni.py`, `qwen2_5_omni.py`, `bagel.py`, `hunyuan_image3.py` |
| `vllm_omni/diffusion/` | Diffusion engine, clients, schedulers, LoRA, offload, profiling | `diffusion_engine.py`, `stage_diffusion_client.py`, `inline_stage_diffusion_client.py`, `data.py` |
| `examples/` | Concrete modality flows and CLI/API usage | `offline_inference/qwen3_omni/end2end.py`, `offline_inference/bagel/end2end.py`, `online_serving/*` |
| `tests/` | Executable expectations for config, scheduler, engine, diffusion, examples | `tests/config/`, `tests/core/`, `tests/engine/`, `tests/diffusion/` |

## 3. The most important cross-directory paths

### 3.1 Model declaration path

1. `vllm_omni/config/pipeline_registry.py` maps a model type to a pipeline module.
2. `vllm_omni/model_executor/models/<family>/pipeline.py` declares stages.
3. `vllm_omni/config/stage_config.py` merges the pipeline with deploy/CLI overrides.
4. `vllm_omni/engine/stage_init_utils.py` extracts metadata and builds engine args.
5. `vllm_omni/engine/async_omni_engine.py` builds stage init plans and launches clients.

If you are adding model support, this is your main path.

### 3.2 Request execution path

1. `vllm_omni/entrypoints/omni.py` receives prompts and sampling params.
2. `vllm_omni/entrypoints/omni_base.py` owns common initialization, final-stage metadata, metrics, and shutdown behavior.
3. `vllm_omni/engine/async_omni_engine.py` serializes request submission into queues.
4. `vllm_omni/engine/orchestrator.py` routes stage submissions and outputs.
5. `vllm_omni/engine/stage_pool.py` chooses a replica and forwards work.
6. Stage workers execute and produce raw/processed outputs.
7. The orchestrator forwards intermediate data or emits `OutputMessage`.
8. The entrypoint converts messages into `OmniRequestOutput` objects.

If you are debugging latency or missing outputs, this is your main path.

### 3.3 Transfer and readiness path

1. A pipeline stage declares processor hooks and possibly `omni_kv_config`.
2. Stage initialization injects connector and KV topology into per-stage engine args.
3. `OmniConnectorModelRunnerMixin` initializes connectors and background I/O.
4. The model runner reports readiness through `OmniConnectorOutput`.
5. `OmniSchedulingCoordinator` transitions requests among `WAITING`, `WAITING_FOR_INPUT`, `WAITING_FOR_CHUNK`, and running states.
6. The stage scheduler admits work only when the required upstream data is available.

If a downstream stage hangs, this is your main path.

## 4. Three “contracts” that explain most code

### Contract A: `StagePipelineConfig` is a structural contract

A stage declaration says which execution type is needed, which upstream stage feeds it, whether it is final, which modality it emits, and which processor hooks bridge semantic gaps.  It should not contain runtime process handles or queue state.

### Contract B: messages are the orchestration contract

`messages.py` defines the event vocabulary between the parent engine/orchestrator and the rest of the runtime.  If a feature cannot be described as a stage submission, output, metrics event, abort, remote registration, or collective RPC, it is probably a new orchestration concept.

### Contract C: connector output is the scheduler contract

Schedulers should not call transport code.  Connector/mixin code reports facts—chunk ready, full payload received, KV ready—and scheduler/coordinator code updates request state.  This makes transports replaceable and scheduler tests possible.

## 5. A productive first-week study plan

Day 1: Read `pipeline_registry.py` and three `pipeline.py` files.  Draw the graphs.

Day 2: Read `StagePipelineConfig`, `PipelineConfig`, `StageConfig`, and `StageConfigFactory` in `stage_config.py`.  Write down what fields are structural versus deploy-time.

Day 3: Trace `Omni.generate()` and `AsyncOmniEngine.add_request()`.  Identify the request ID and final-stage ID at every boundary.

Day 4: Read `orchestrator.py` and `stage_pool.py` at a high level.  Do not chase every async call; focus on routing and output handling.

Day 5: Read `omni_connector_model_runner_mixin.py` and `omni_scheduling_coordinator.py`.  Explain why `WAITING_FOR_INPUT` and `WAITING_FOR_CHUNK` exist.

Day 6: Read one stage processor file, preferably `qwen3_omni.py` or `bagel.py`.  Identify the exact payload fields exchanged between stages.

Day 7: Run or inspect an example script and map its prompt schema to the stage graph.

## 6. What not to do first

Do not start by reading every model implementation line-by-line.  Many model files contain architecture-specific neural network details that are less important for understanding the serving system.  First learn the stage graph and runtime contracts; then model internals will have a place in your mental map.

Do not assume upstream vLLM concepts are unchanged.  vLLM-Omni inherits vLLM’s engine and scheduler foundations but adds new state for multimodal transfer, final-output modality, diffusion stages, and multi-stage orchestration.
