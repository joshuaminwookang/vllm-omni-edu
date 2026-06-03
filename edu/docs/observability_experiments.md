# Observability, experiments, and study assignments

## 1. Why observability is a first-class topic

Efficient multimodal inference cannot be understood from end-to-end latency alone.  A request may spend time in multimodal preprocessing, AR prefill, AR decode, connector transfer, full-payload waiting, chunk waiting, KV slicing/merging, diffusion denoising, waveform rendering, output packaging, or queueing behind another modality.  If the runtime reports only final latency, optimization becomes guesswork.

vLLM-Omni includes stage-aware metrics, transfer metrics, modality metrics, profiling hooks, and examples with profiler flags.  Students should treat observability as part of the system design, not as an afterthought.

## 2. Metrics to collect

For each experiment, record:

- request modality and final stage;
- end-to-end latency;
- first-output latency for text/audio/image where applicable;
- per-stage queueing time;
- per-stage compute time;
- transfer wait time;
- transfer payload size;
- KV/cache transfer volume;
- peak GPU memory per stage;
- utilization per stage replica;
- output quality or correctness signal;
- abort/failure/hang counts.

The key is to correlate modality with stage path.  A text-only request to an omni model should not be compared directly to an audio-output request without noting skipped stages.

## 3. Experiment 1: final-stage routing efficiency

Goal: quantify how much work is saved when requests stop early.

Procedure:

1. Choose a pipeline with multiple final-output stages, such as Qwen3-Omni.
2. Prepare text-only prompts and audio-output prompts of similar text length.
3. Run with the same stage deployment.
4. Measure which stages receive submissions and outputs.
5. Compare end-to-end latency, stage queueing, and downstream utilization.

Expected lesson: final-stage computation is an efficiency mechanism, not merely output formatting.

## 4. Experiment 2: async chunk versus full payload

Goal: understand first-audio latency and throughput tradeoffs.

Procedure:

1. Use a speech pipeline with async chunk processors.
2. Run once with `async_chunk=True` and once with `async_chunk=False` if supported.
3. Measure first-audio latency, final audio latency, stage utilization, and number of waiting-state transitions.
4. Inspect logs for processor selection and connector mode.

Expected lesson: async chunking can reduce first-output latency but increases scheduler/connector state and may affect batching efficiency.

## 5. Experiment 3: AR→diffusion cache handoff

Goal: study cache/conditioning transfer cost.

Procedure:

1. Choose BAGEL or a similar AR→diffusion pipeline.
2. Compare variants that transfer after prefill versus after reasoning if available.
3. Record diffusion start time, image completion time, cache payload size, and quality differences.
4. Vary prompt length and resolution.

Expected lesson: the best transfer criterion depends on both quality semantics and systems cost.

## 6. Experiment 4: per-stage replica allocation

Goal: capacity-plan a heterogeneous stage graph.

Procedure:

1. Measure average service time for each stage under a fixed workload.
2. Identify the bottleneck stage.
3. Increase replicas for that stage only.
4. Compare throughput, queueing, and GPU utilization.
5. Repeat under a different modality mix.

Expected lesson: the optimal replica allocation depends on workload modality distribution, not just model size.

## 7. Experiment 5: heterogeneous tensor parallelism

Goal: evaluate cross-stage TP mismatch overhead.

Procedure:

1. Configure adjacent stages with equal TP size.
2. Configure them with different TP sizes.
3. Use a pipeline with KV/cache transfer.
4. Measure startup injection logs, transfer latency, and correctness.
5. Inspect rank-aware slicing/merging paths if results differ.

Expected lesson: heterogeneous TP can improve resource allocation but makes transfer semantics more complex.

## 8. Experiment 6: diffusion colocated versus isolated

Goal: study memory and latency tradeoffs for diffusion stages.

Procedure:

1. Run an AR→diffusion model with colocated stages if supported.
2. Run with diffusion isolated in its own process/device allocation.
3. Measure peak memory, diffusion queueing, transfer latency, and failure/OOM rate.
4. Repeat with LoRA/offload/cache acceleration options if available.

Expected lesson: colocating stages can reduce communication but may create memory pressure that reduces throughput or reliability.

## 9. Debugging lab: downstream stage never starts

Given a request that reaches Stage 0 but not Stage 1, answer:

1. Did final-stage routing intentionally stop at Stage 0?
2. Does Stage 1 have `input_sources=(0,)` or equivalent?
3. Was the correct processor selected for async/full-payload mode?
4. Did the connector initialize with a non-null custom processor?
5. Did the upstream stage produce payload/chunk/KV output?
6. Did the scheduler coordinator register the downstream receive?
7. Did the request leave `WAITING_FOR_INPUT` or `WAITING_FOR_CHUNK`?
8. Was the request aborted or marked finished early?

This lab teaches students to debug the system rather than the neural network.

## 10. Reading assignments

### Assignment A: draw a stage graph

Pick three pipelines from `pipeline_registry.py`.  Draw their stage graphs and annotate execution type, final-output type, processor hooks, and KV/cache behavior.

### Assignment B: write a lifecycle trace

For one example under `examples/offline_inference/`, trace how its prompt dictionary becomes a stage submission and final output.

### Assignment C: processor ABI audit

Open one processor file and document every key in the payload it consumes and returns.  Identify whether missing keys cause warnings, empty payloads, or hard errors.

### Assignment D: scheduler state proof

Using `OmniSchedulingCoordinator`, prove informally that a request receiving a full payload eventually becomes schedulable, assuming the connector reports arrival and no abort occurs.

### Assignment E: propose an optimization

Choose one bottleneck and propose an optimization.  Your proposal must state which files change, which metrics improve, and which invariants could break.

## 11. Reporting template for student projects

A good project report should include:

1. model/pipeline studied;
2. stage graph diagram;
3. workload description and modality mix;
4. deployment configuration;
5. per-stage metrics table;
6. transfer and memory analysis;
7. quality/correctness checks;
8. failure cases;
9. code changes or patches;
10. lessons for future omni serving systems.

This format mirrors how serious systems papers explain performance: architecture first, methodology second, measurement third, interpretation last.
