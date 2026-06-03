# Diffusion and generation stages: serving non-token workloads as first-class stages

## 1. Why diffusion is not postprocessing

Many multimodal demos treat image generation as a final Python function after text generation.  vLLM-Omni treats diffusion as a stage execution type.  That is a major architectural choice.  A diffusion stage has its own config, process/client lifecycle, parallelism settings, scheduler behavior, output modality, and profiling.  It may receive AR-produced conditioning, KV cache, CFG companion data, or original multimodal prompt data.

This makes diffusion a peer of LLM execution, not a helper function.

## 2. Stage execution types revisited

`StageExecutionType.DIFFUSION` indicates that the stage should not be initialized as an ordinary vLLM AR worker.  `StageExecutionType.LLM_GENERATION` covers generation-style LLM stages that are not ordinary text AR stages, such as waveform/code rendering stages.  `StageExecutionType.LLM_AR` covers classical autoregressive stages.

The distinction matters because each execution type has different assumptions:

| Execution type | Typical output | Scheduler/client style | Example role |
| --- | --- | --- | --- |
| `LLM_AR` | tokens, text, hidden/latent payload | omni AR scheduler | Thinker, Talker, AR image conditioner |
| `LLM_GENERATION` | generated non-text payload | generation scheduler/worker | Code2Wav or audio renderer |
| `DIFFUSION` | image/audio/video tensor output | diffusion client/executor | DiT image generation, image/video pipeline |

## 3. Diffusion config and clients

`vllm_omni/diffusion/` contains the diffusion runtime.  Important pieces include:

- `diffusion_engine.py`: request submission and execution for diffusion workloads;
- `stage_diffusion_client.py`: process-backed client for stage-level diffusion;
- `inline_stage_diffusion_client.py`: inline client path for colocated execution;
- `stage_diffusion_proc.py`: process spawning and handshake;
- `data.py`: diffusion config and sampling/parallel data structures;
- `sched/`: diffusion request schedulers;
- `lora/`: diffusion LoRA management;
- `offloader/`: layerwise/offload support;
- `profiler/`: diffusion profiling support.

The engine layer imports these components in `async_omni_engine.py` and chooses diffusion startup paths during stage plan construction.

## 4. AR→diffusion handoff semantics

In BAGEL-like pipelines, an AR stage prepares context for a DiT stage.  That handoff can involve:

- prompt expansion for CFG;
- companion requests to produce parallel cache/conditioning branches;
- KV cache transfer at prefill completion or after reasoning;
- `cfg_kv_collect_func` to collect and organize cache for diffusion;
- custom input processors to convert AR outputs into diffusion requests.

This is not equivalent to passing a text caption into Stable Diffusion.  The AR stage can produce rich model-internal state, and the diffusion stage may rely on exact cache alignment.

## 5. HunyuanImage3-style AR→DiT handoff

HunyuanImage3 demonstrates a clean two-stage AR→DiT pipeline and also AR-only/DiT-only variants.  The full pipeline declares an AR stage that emits latent output and a DiT diffusion stage with a custom `ar2diffusion` input processor.  The variants are useful for debugging, partial deployment, and benchmarking stage costs independently.

A student exercise is to compare full, AR-only, and DiT-only variants and ask: what runtime code is reused, and what graph facts differ?

## 6. Diffusion parallelism and memory

Diffusion workloads have parallelism and memory concerns distinct from text decode:

- attention backend choices;
- sequence/ring/Ulysses parallelism;
- CFG parallelism;
- VAE patch parallelism;
- LoRA adapter cache management;
- layerwise offload;
- cache acceleration such as TeaCache/Cache-DiT;
- large activation tensors for image/video resolutions.

`_apply_diffusion_parallel_runtime_overrides()` in config code routes diffusion parallel overrides into nested diffusion config, reflecting that diffusion stages have their own parallel schema.  A global LLM-only config would be insufficient.

## 7. Generation stages for audio

`LLM_GENERATION` stages, such as Code2Wav-like components, occupy a middle ground.  They are not diffusion stages, but their output is not ordinary detokenized text either.  They may consume codec IDs or latent tokens and emit audio tensors.  Their sampling params and output processing must preserve audio-specific semantics.

This is why Qwen3-Omni’s pipeline marks Talker as `LLM_AR` with intermediate latent output and Code2Wav as `LLM_GENERATION` with final audio output.  The architecture separates semantic token/code generation from waveform rendering.

## 8. Evaluation questions for non-token stages

For diffusion and audio generation stages, ask:

- What is the time to first perceptible output, not only final completion?
- Is the stage compute-bound, memory-bound, or transfer-bound?
- How much queueing does the stage accumulate relative to AR stages?
- Does colocating it with upstream stages reduce transfer latency or cause memory pressure?
- Does cache/conditioning transfer dominate small requests?
- How do sampling quality knobs change scheduler occupancy?
- Are output tensors copied unnecessarily between CPU/GPU/processes?

These questions are central to multimodal efficient inference and are invisible in tokens/sec metrics.
