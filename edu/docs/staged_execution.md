# Efficient staged execution: orchestration, scheduling, transfer, and latency control

## 1. Why staged execution is the natural form for omni inference

Text-only LLM serving is usually described as two phases: prefill and decode.  Recent systems such as DistServe argue that these phases have different compute/memory profiles and should sometimes be disaggregated across GPU pools.  Sarathi-Serve shows that chunking prefill can reduce stalls and improve latency/throughput tradeoffs.  vLLM’s PagedAttention shows that KV-cache memory layout is central to high-throughput serving.

Omni inference generalizes all three ideas.  A single user request may require:

- a multimodal AR stage to ingest text, images, video frames, or audio features;
- a second AR stage to transform hidden states into codec or semantic tokens;
- a non-AR stage to synthesize waveform or image tensors;
- a diffusion stage whose computational profile differs radically from token decoding;
- optional early termination when the request asks only for text, or continued execution when it asks for audio/image.

Thus vLLM-Omni’s staged execution is not merely pipeline parallelism.  It is **semantic disaggregation**: each stage owns a modality-specific computation and communicates through declared transfer interfaces.

## 2. Request submission through the synchronous `Omni` entrypoint

The offline entrypoint `vllm_omni/entrypoints/omni.py` shows the high-level loop:

1. `Omni.generate()` normalizes per-stage sampling params, expands params when prefill/decode disaggregation is active, and delegates to `_run_generation()`.
2. `_set_final_only_for_llm_stages()` forces non-diffusion LLM stages to return `FINAL_ONLY`, avoiding token-level streaming from intermediate stages unless explicitly needed.
3. `_run_generation()` converts one prompt or a sequence of prompts into request IDs.
4. For each request, it reads prompt modalities, computes the request’s final stage via `_compute_final_stage_id()`, creates per-request metrics, applies PD-specific sampling changes, and calls `self.engine.add_request()` with `sampling_params_list` and `final_stage_id`.
5. The loop then repeatedly calls `self.engine.try_get_output()`, validates output/error messages, processes stage-level results, yields externally visible outputs, and cleans up completed requests.

The key detail is that `final_stage_id` is per request.  A text-only request to a Qwen3-Omni pipeline may finish after the Thinker; an audio request may continue through Talker and Code2Wav.  This per-request final-stage computation is what makes a single deployed graph usable for multiple output modalities.

## 3. The asynchronous engine and orchestrator boundary

`vllm_omni/engine/async_omni_engine.py` is the parent engine.  It is intentionally a proxy and coordinator rather than a model runner.  It loads and resolves stage configs, builds logical stage initialization plans, starts local or remote stage engines, creates diffusion clients when needed, configures connectors/KV transfer, and exposes a queue-based API to the caller.

A central method is `_build_logical_stage_init_plans()`:

- It extracts metadata for each stage.
- It resolves connector specifications from the omni transfer config.
- It resolves `omni_kv_config` for the stage.
- For non-diffusion stages, it builds per-stage engine args and vLLM configs.
- It injects connector config and inferred KV tensor-parallel topology.
- It creates per-replica init plans, including launch mode and device assignment.

This design makes orchestration explicitly separate from execution.  The parent engine knows which stage exists, where it lives, and how to connect it, but the stage worker still owns forward passes and local scheduling.

## 4. Stage pools and load balancing

The staged runtime has to answer a production-serving question: if a stage has multiple replicas, where should a request go?  The `StagePool` and `StagePoolClient` abstractions under `vllm_omni/engine/stage_pool.py` encapsulate a group of stage clients.  `AsyncOmniEngine` imports `LoadBalancer` and `build_load_balancer_factory()` from `vllm_omni/distributed/omni_coordinator.py`, then uses stage pools during orchestration.

This matters for multimodal inference because stage costs are highly asymmetric.  A Talker may be lighter than a Thinker; a diffusion DiT may dominate image latency; a Code2Wav stage may be optimized for streaming.  Per-stage replica counts allow researchers to scale bottleneck stages independently, which is a systems analogue of mixture-of-experts routing: different work types should be served by different pools.

## 5. Transfer modes: full payload, async chunk, and KV cache

`vllm_omni/worker/omni_connector_model_runner_mixin.py` is the unified data-plane mixin for model runners.  Its docstring states the core contract: connector `put()`/`get()` calls live here; background I/O threads handle async-chunk and full-payload transfers; KV cache is delegated to the KV transfer manager; transfer results are reported as `OmniConnectorOutput` so schedulers can make decisions without touching connector implementations.

The mixin supports three transfer modes:

1. **Full payload mode.** A producer accumulates a complete intermediate object and sends it to the next stage.  This fits non-streaming handoffs such as AR latent payload to diffusion input.
2. **Async chunk mode.** The producer sends incremental chunks while the upstream stage is still generating.  This fits streaming speech, where waiting for a full utterance would increase first-audio latency.
3. **KV-cache transfer.** The producer sends attention KV blocks or cache-like structures to downstream stages.  This fits AR→DiT conditioning and CFG-style reuse.

The helper `should_accumulate_full_payload_output()` is a useful example of robust engineering: it only enables producer-side full-payload accumulation when the stage declares a downstream full-payload processor, async chunking is off, the stage is not a final output, and `model_stage` exists.  This avoids accidentally turning a consumer-side helper name into producer behavior.

## 6. Scheduler coordination with `WAITING_FOR_CHUNK` and `WAITING_FOR_INPUT`

`vllm_omni/core/sched/omni_scheduling_coordinator.py` provides the scheduling-side state machine for downstream stages.  It tracks:

- requests with ready chunks;
- finished requests;
- full-payload inputs that have arrived;
- pending chunk registrations that the model runner should start polling;
- pending full-payload input registrations;
- monotonic timestamps for timeout detection.

For async chunking, `process_pending_chunks()` transitions requests whose chunks have arrived back to schedulable states.  It also handles terminal-ready requests and trims the running queue back to `scheduler_max_num_seqs` without freeing KV blocks as if the request were truly preempted.

For full payload mode, `process_pending_full_payload_inputs()` moves fresh non-stage-0 requests into `WAITING_FOR_INPUT`, registers them for background polling, and returns them to `WAITING` once data arrives.  The scheduler therefore does not spin on requests whose upstream data is absent; it parks them in an explicit state.

This is the staged analogue of prefill/decode scheduling.  Instead of “prefill complete, now decode,” the scheduler reasons over “upstream chunk arrived,” “full payload arrived,” or “cache ready.”  The abstraction is general enough for text, speech, image, and video pipelines.

## 7. Model-family stage processors as the semantic ABI

A stage graph is only useful if adjacent stages agree on intermediate formats.  vLLM-Omni makes that contract explicit through dotted processor functions:

- Qwen3-Omni’s pipeline points to `vllm_omni.model_executor.stage_input_processors.qwen3_omni` functions such as `thinker2talker_full_payload`, `thinker2talker_async_chunk`, `talker2code2wav_full_payload`, and `talker2code2wav_async_chunk`.
- BAGEL points to processors such as `expand_cfg_prompts`, `expand_cfg_prompts_think`, and `collect_cfg_kv_caches`.
- HunyuanImage3 points to `ar2diffusion` for AR→DiT conversion.

This processor layer is effectively an **application binary interface for tensors and multimodal payloads**.  It defines whether the next stage sees token IDs, embeddings, hidden-state tensors, codec codes, CFG-expanded prompts, or diffusion conditioning structures.  The engine imports these functions dynamically, so a new model family can define its own intermediate representation without changing the connector or scheduler APIs.

## 8. Early finalization and multimodal routing

The repo’s final-output design matters for efficiency.  In a pipeline where Stage 0 can produce text and Stage 2 can produce audio, the system should not execute Stage 1 and Stage 2 for every request.  The `Omni` entrypoint computes a final stage using prompt modalities and sends that value with `add_request()`.  The scheduler and worker code then use request metadata to decide when KV transfer or downstream payload transfer can be omitted.

`vllm_omni/core/sched/omni_ar_scheduler.py` contains logic for requests that omit transfer to the next stage, for example when a text-only request finalizes at Stage 0.  That is critical: without this optimization, a mixed text/audio service would pay audio-pipeline costs even for text-only traffic.

## 9. Heterogeneous tensor parallelism and KV topology

Different stages may run with different tensor-parallel sizes.  For example, a large Thinker could use more GPUs than a lightweight Talker or Code2Wav stage.  `vllm_omni/engine/stage_init_utils.py` includes `_inject_inferred_kv_tp_topology()`, which infers adjacent-stage TP topology from loaded stage configs and injects rank mapping into `omni_kv_config` when KV send/recv is enabled.

This is a nontrivial systems feature.  KV-cache transfer is easy when both sides have identical TP degree and rank partitioning; it is harder when stage A shards hidden states or KV tensors differently from stage B.  The repo’s approach is to infer topology at startup and centralize rank-aware slicing/merging in connector/worker utilities.  For researchers, this is an example of how deployment heterogeneity leaks into what might initially look like a model-only interface.

## 10. Diffusion stages as first-class citizens

Diffusion stages do not fit AR token schedulers.  `AsyncOmniEngine` imports diffusion-specific data structures and clients such as `DiffusionParallelConfig`, `InlineStageDiffusionClient`, `StageDiffusionClient`, and diffusion process startup helpers.  Stage config generation routes diffusion parallel overrides into nested diffusion parallel config.  Diffusion stages can be inline or process-backed, and they use their own parallelism knobs such as sequence parallel, Ulysses degree, ring degree, CFG parallelism, and VAE patch parallelism elsewhere in the repo.

The important architectural principle is that diffusion is not bolted on as postprocessing.  It is a stage execution type with its own executor path, deployment config, output modality, and input processors.  This is the right abstraction for modern multimodal models where generation often combines language, vision encoders, and diffusion transformers.

## 11. Relationship to recent inference systems research

- **PagedAttention/vLLM.** The base vLLM insight is that KV memory management determines serving throughput.  vLLM-Omni inherits vLLM’s serving substrate and extends KV-awareness across stages through `omni_kv_config` and connector-managed transfer.
- **DistServe.** DistServe separates prefill and decode because they have different latency/resource profiles.  vLLM-Omni generalizes separation to modality stages: Thinker/Talker/Code2Wav, AR/DiT, or tokenizer/diffusion subgraphs.
- **Sarathi-Serve.** Sarathi-Serve’s chunking idea appears conceptually in async chunk transfer: instead of waiting for a full upstream output, downstream stages can begin when chunks arrive.
- **Qwen2.5-Omni and Qwen3-Omni.** These reports motivate the engine’s need for staged speech generation.  The repo turns Thinker/Talker/codec decomposition into a scheduler-visible pipeline.

## 12. Practical reading checklist

When analyzing a staged pipeline, answer these questions:

1. Which stages are `LLM_AR`, `LLM_GENERATION`, or `DIFFUSION`?
2. Which stages are final outputs, and for which modalities?
3. Which stages require original multimodal data?
4. Does the handoff use full payload, async chunks, KV cache, or a combination?
5. Are there model-family processor hooks for input conversion or next-stage conversion?
6. Are sampling constraints used to enforce detokenization, stop IDs, or output kind?
7. Can the request terminate early for some modality choices?
8. Are stages expected to have heterogeneous tensor parallelism or replica counts?
9. How would first-token, first-audio, or first-image latency be measured from the orchestrator metrics?
10. What part of the graph is the true bottleneck under expected traffic?

The answer to those questions is usually more informative than a raw parameter count.

## 13. Timeline view: one request through a three-stage speech pipeline

For a Qwen3-Omni-style audio request, an approximate staged timeline is:

1. **Client submission.** The user passes prompt text plus optional image/audio/video data to `Omni.generate()`.
2. **Final-stage computation.** The entrypoint determines that audio is requested, so the request should not stop at the Thinker text output.
3. **Stage-0 enqueue.** The parent engine constructs a `StageSubmissionMessage` carrying the prompt, original prompt, sampling params for all stages, final stage ID, preprocessing latency, and enqueue timestamp.
4. **Thinker execution.** Stage 0 performs multimodal understanding and generates text/latent information.  If async chunking is enabled, it may emit chunks toward Talker before its full output is complete.
5. **Stage-0 output handling.** The orchestrator observes raw or processed outputs, emits metrics, and decides whether to forward to Stage 1 or return a final text output.
6. **Talker scheduling.** Stage 1 waits until its input is available: either full payload arrives, chunks arrive, or KV/cache readiness is signaled.
7. **Codec/code generation.** Stage 1 produces audio-code-like intermediate data and forwards it to Code2Wav.
8. **Code2Wav generation.** Stage 2 produces waveform data as a final audio output.
9. **Finalization.** The entrypoint receives an `OutputMessage`, updates per-stage metrics, observes modality metrics, and yields an `OmniRequestOutput` with audio in the multimodal output field.

The important systems property is overlap.  In a non-streaming implementation, Stage 1 cannot start until Stage 0 fully finishes and transfers a complete object.  In async chunk mode, downstream work can become schedulable earlier.  That improves first-audio latency but creates new scheduler states and transfer bookkeeping.

## 14. Timeline view: one request through an AR→diffusion image pipeline

For BAGEL/HunyuanImage3-style image generation, the timeline differs:

1. The AR stage may consume multimodal inputs and create text, hidden, or latent conditioning.
2. If CFG is used, prompt expansion can create companion requests so positive/negative or conditional/unconditional branches have matching cache structures.
3. The AR stage may transfer KV cache after prefill or after a longer reasoning sequence, depending on the pipeline variant.
4. The diffusion stage waits for the correct conditioning payload or cache before entering its denoising loop.
5. The diffusion executor produces an image output, and the orchestrator emits it as a final modality.

This path is less like token streaming and more like cross-engine conditioning.  The latency bottleneck may be diffusion denoising rather than AR decode; memory bottlenecks may involve VAE/DiT weights rather than KV-cache growth.  Therefore the runtime cannot use one global LLM scheduler for every stage.

## 15. Queueing theory lens: where latency hides

A staged omni system has at least six latency components:

1. **Frontend preprocessing:** prompt normalization, multimodal loading, tokenization, and `EngineCoreRequest` construction.
2. **Stage queueing:** time between stage submission and scheduler admission.
3. **Stage compute:** prefill, decode, generation, diffusion denoising, or waveform synthesis.
4. **Transfer wait:** time parked in `WAITING_FOR_INPUT` or `WAITING_FOR_CHUNK` before downstream data arrives.
5. **Transfer compute/bandwidth:** serialization, connector send/recv, KV slicing/merging, and device movement.
6. **Finalization:** output processing, detokenization, waveform/image packaging, metrics, and cleanup.

Research reports often collapse these into one end-to-end number.  The repo’s stage-aware design encourages a better methodology: measure each component and compare how the breakdown changes under different graph shapes and modality mixes.

## 16. Why stage-local schedulers need global context

Each stage scheduler makes local decisions, but those decisions depend on global graph facts:

- A Stage-0 request may not need downstream transfer if its final stage is 0.
- A Stage-1 request should not occupy scheduler capacity until upstream payload/chunk readiness exists.
- A diffusion stage may need cache from an AR stage whose TP topology differs.
- A companion CFG request may be invisible to the user but necessary for downstream image quality.
- A downstream stage may have more or fewer replicas than the upstream producer.

vLLM-Omni resolves this tension by passing global facts as metadata: final stage ID, connector outputs, `omni_kv_config`, stage input sources, and per-stage runtime metadata.  The scheduler remains stage-local, but its readiness conditions are graph-aware.

## 17. Failure-mode study: how staged execution can hang

The staged design is powerful but unforgiving.  A missing producer hook can leave a downstream request waiting forever.  A processor that returns an empty payload may park a consumer in `WAITING_FOR_INPUT`.  A wrong async/sync processor selection can send chunks while the downstream scheduler expects a full payload.  A wrong KV transfer criterion can make a diffusion stage start with incomplete conditioning or never start at all.

When debugging a hang, inspect in this order:

1. Does the pipeline declaration have matching producer and consumer processor hooks?
2. Did deploy/CLI resolution choose `async_chunk=True` or `False` as expected?
3. Did the model runner log connector initialization with the expected custom process function?
4. Did `OmniConnectorOutput` report ready chunks, finished chunks, full-payload recv, or KV readiness?
5. Did `OmniSchedulingCoordinator` move the request out of `WAITING_FOR_INPUT` or `WAITING_FOR_CHUNK`?
6. Did the orchestrator receive a processed output or only a stage metrics message?

This debugging checklist is often more useful than inspecting model logits.
