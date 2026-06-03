# Omni architecture support: from model families to executable stage graphs

## 1. The problem: omni models are not one architecture

A multimodal inference engine cannot assume that every model is an autoregressive decoder with a prompt and a token stream.  This repository supports at least four broad architectural patterns:

1. **Multimodal understanding plus speech generation**, such as Qwen2.5-Omni and Qwen3-Omni, where a language-centric “Thinker” consumes text/image/audio/video and a speech-oriented “Talker” or codec stage emits audio tokens or waveforms.
2. **Autoregressive reasoning followed by diffusion generation**, such as BAGEL and HunyuanImage3, where a discrete/latent AR component prepares context and a DiT or diffusion pipeline renders images.
3. **Single-stage diffusion or unified models**, where the diffusion engine encapsulates tokenizer, visual encoder, VAE, and transformer behavior internally.
4. **Specialized TTS/audio generation stacks**, where the output is audio and the intermediate representation may be semantic tokens, residual vector quantization codes, hidden states, or waveform tensors.

The core architectural insight in vLLM-Omni is that the runtime should not hard-code those families.  Instead, it should represent each family as a **stage graph** with a small fixed contract: stage identity, execution type, input sources, final-output behavior, modality requirements, processor hooks, sampling constraints, and transfer metadata.  The topological layer is in `vllm_omni/config/stage_config.py`, while the model-family registry is in `vllm_omni/config/pipeline_registry.py`.

This mirrors the way vLLM made paged KV management mostly independent of model architecture: PagedAttention separates serving memory management from the model’s transformer math.  vLLM-Omni extends that philosophy from “many text models” to “many multimodal execution graphs.”

## 2. The declarative topology layer

### 2.1 `StageExecutionType`: one enum, multiple execution semantics

`StageExecutionType` in `vllm_omni/config/stage_config.py` distinguishes the runtime behavior of a stage:

- `LLM_AR` means autoregressive token-by-token execution using omni-aware AR schedulers.
- `LLM_GENERATION` means a generation stage that is still LLM-like but does not follow the same scheduler contract as a normal AR stage; Qwen3-Omni’s Code2Wav uses this path.
- `DIFFUSION` means a diffusion pipeline stage, usually driven through the diffusion executor/client path rather than vLLM’s normal token scheduler.

The `_resolve_scheduler()` helper maps these execution types to scheduler classes.  This is a crucial abstraction: model-family code can say “this is an AR stage” or “this is a diffusion stage” without importing scheduler classes directly.

### 2.2 `StagePipelineConfig`: the minimum semantic unit

`StagePipelineConfig` captures the model-independent facts about a stage:

- `stage_id`: the logical order in the graph.
- `model_stage`: a model-family name such as `thinker`, `talker`, `code2wav`, `dit`, or `AR`.
- `execution_type`: the execution backend and scheduler family.
- `input_sources`: upstream stage IDs.
- `final_output` and `final_output_type`: whether this stage may be externally visible and what modality it produces.
- `owns_tokenizer`: whether the stage owns tokenizer responsibilities.
- `requires_multimodal_data`: whether the original multimodal prompt data must remain available.
- `hf_config_name`, `model_arch`, `model_subdir`, and `tokenizer_subdir`: indirections needed for multi-component Hugging Face repositories.
- `custom_process_input_func` and `custom_process_next_stage_input_func`: dotted import paths for model-family-specific conversion functions.
- `async_chunk_process_next_stage_input_func` and `sync_process_input_func`: alternate conversion functions selected by deployment mode.
- `prompt_expand_func` and `cfg_kv_collect_func`: hooks for classifier-free guidance and diffusion/KV handoff.
- `omni_kv_config`: declarative KV-transfer behavior.

The important design choice is that **model-specific transformation logic is referenced, not embedded**.  A Qwen stage can point at Qwen processors; a BAGEL stage can point at BAGEL processors; the orchestrator remains generic.

### 2.3 `PipelineConfig`: a model family as a frozen graph

`PipelineConfig` groups stages into a topology and adds detection metadata:

- `model_type` is the registry key.
- `model_arch` is the model class name used when a stage does not override it.
- `hf_architectures` resolves Hugging Face architecture-name collisions.
- `hf_config_predicate` handles cases where two related checkpoints share an architecture string but differ in a config field.
- `diffusers_pipeline_cls` supports diffusion-family detection.

This is especially important for omni models, because `model_type` is often not enough.  For example, a checkpoint may report a generic `qwen2` type while its architecture list indicates a specific multimodal lineage.  The factory therefore has both a model-type path and an architecture fallback path.

## 3. Central registration: one table for many families

`vllm_omni/config/pipeline_registry.py` is the repository’s architectural catalog.  It maps `model_type` strings to `(module_path, variable_name)` pairs and imports the pipeline lazily.  It currently includes Qwen2.5-Omni, Qwen3-Omni, Qwen3-TTS, Covo Audio, BAGEL variants, Lance, GLM Image, HunyuanImage3 variants, VoxCPM2, CosyVoice3, MiMo Audio, Voxtral TTS, GLM-TTS, Fish Speech, Ming Flash Omni variants, MOSS-TTS, MiniCPM-o 4.5, and Higgs Audio V2.

This table is more than convenience.  It makes the repo inspectable: researchers can see which architectural forms the engine treats as first-class and can compare their stage decompositions by opening each family’s `pipeline.py`.

A useful reading exercise:

1. Open `vllm_omni/config/pipeline_registry.py` and list the registered model families.
2. Open each family’s `vllm_omni/model_executor/models/<family>/pipeline.py`.
3. Classify the family by graph shape: AR-only, AR→AR→generation, AR→diffusion, diffusion-only, or variant selection.
4. Identify where the graph exposes text, audio, or image as a final output.

## 4. Case study: Qwen3-Omni as a three-stage speech graph

`vllm_omni/model_executor/models/qwen3_omni/pipeline.py` declares:

- Stage 0, `thinker`: `LLM_AR`, no input sources, owns tokenizer, requires multimodal data, final text output, and emits latent representations for downstream speech.
- Stage 1, `talker`: `LLM_AR`, consumes Stage 0, emits latent audio-code information, uses stage processors for thinker→talker and talker→code2wav conversion, and disables detokenization while stopping on a codec stop token.
- Stage 2, `code2wav`: `LLM_GENERATION`, consumes Stage 1, final audio output, and uses a talker→code2wav processor.

This maps directly onto the Thinker/Talker/codec decomposition described by Qwen-Omni technical reports.  The Qwen2.5-Omni report describes a Thinker that performs language-centric reasoning and a Talker that uses Thinker hidden states to produce speech-related outputs.  The Qwen3-Omni report goes further by emphasizing a lightweight causal Code2Wav renderer for low-latency waveform reconstruction.  In this repo, that architectural story becomes executable topology: three stages, two intermediate transfer hooks, and two final-output possibilities.

Key inference implication: **text and audio are not mutually exclusive final outputs**.  Stage 0 can be final for text-only or text-visible behavior, while Stage 2 is final for audio.  The orchestrator therefore needs request-level final-stage computation rather than a single hard-coded final stage.

## 5. Case study: BAGEL as AR-to-diffusion with KV transfer

`vllm_omni/model_executor/models/bagel/pipeline.py` demonstrates a different pattern:

- The default `BAGEL_PIPELINE` has Stage 0 `thinker` as `LLM_AR`, final text output, multimodal input, prompt expansion, and `omni_kv_config` indicating that KV cache should be sent after `prefill_finished`.
- Stage 1 `dit` is `DIFFUSION`, final image output, receives KV cache, and has a `cfg_kv_collect_func` for classifier-free guidance cache collection.
- `BAGEL_THINK_PIPELINE` modifies the handoff criteria so transfer happens after reasoning tokens rather than immediately after prefill.
- `BAGEL_SINGLE_STAGE_PIPELINE` represents a self-contained diffusion stage.

The architecture lesson is that staged execution is not just about splitting layers.  It can split **semantic roles**: a language model builds conditioning context; a diffusion model renders.  The transfer object may be tokens, latent tensors, or KV cache.  The engine’s generic stage contract covers all three by separating stage metadata from transfer mechanism.

## 6. Case study: HunyuanImage3 as AR-to-DiT with selectable subgraphs

`vllm_omni/model_executor/models/hunyuan_image3/pipeline.py` declares:

- `HUNYUAN_IMAGE3_PIPELINE`: two stages, AR then DiT, final image output.
- `HUNYUAN_IMAGE3_AR_PIPELINE`: AR-only variant, final text output.
- `HUNYUAN_IMAGE3_DIT_PIPELINE`: DiT-only variant, final image output.

This is a useful model of **subgraph serving**.  A production system may want to deploy a full multimodal model, only the AR component for debugging or routing, or only the diffusion component behind another upstream service.  vLLM-Omni expresses those as separate `PipelineConfig` objects with shared constants.  The implication is that a stage graph can be a packaging/deployment artifact, not merely a faithful representation of training-time modules.

## 7. Runtime detection and config synthesis

`StageConfigFactory.create_from_model()` in `vllm_omni/config/stage_config.py` performs pipeline selection:

1. Load or infer the Hugging Face config.
2. Match `model_type` against the central pipeline registry.
3. If needed, match `architectures` against `hf_architectures` declared by registered pipelines.
4. If a predicate exists, evaluate it to resolve same-architecture sibling models.
5. Merge the frozen pipeline topology with deployment YAML and CLI overrides.
6. Return concrete `StageConfig` objects consumed by engine startup.

This is the bridge from model identity to executable runtime.  Researchers should notice how many error-prone details are handled here rather than in model implementations: async-chunk mode, per-stage overrides, stage-specific runtime fields, scheduler class injection, and deployment defaults.

## 8. Why this architecture scales to new model families

To add a new omni architecture, the clean path is:

1. Implement or wrap model execution code under `vllm_omni/model_executor/models/<family>/`.
2. Write stage input processors under `vllm_omni/model_executor/stage_input_processors/` if intermediate formats are family-specific.
3. Declare `PipelineConfig` and `StagePipelineConfig` instances in `<family>/pipeline.py`.
4. Register the pipeline in `vllm_omni/config/pipeline_registry.py`.
5. Add a deployment YAML if runtime defaults differ from stage topology defaults.
6. Add examples under `examples/offline_inference/` or `examples/online_serving/`.

This is a strong separation-of-concerns pattern for research codebases.  It lets model researchers change graph topology without editing scheduler internals, and systems researchers change transfer/scheduling policies without rewriting model-family declarations.

## 9. Conceptual summary

vLLM-Omni treats an omni model as a **typed, partially observable, multimodal stage graph**:

- **Typed** because each stage has an execution type and output modality.
- **Partially observable** because multiple stages can be final outputs for different request modalities.
- **Multimodal** because prompts and outputs may include text, audio, images, and video.
- **A stage graph** because execution order, data dependencies, and transfer hooks are explicit.

That abstraction is the foundation for every efficiency mechanism described in the staged-execution document.

## 10. Deeper comparative anatomy of supported architecture patterns

The first pass through `pipeline_registry.py` shows a list of names.  The second pass should classify why those names need different runtime contracts.  A useful taxonomy is:

| Pattern | Example files | Runtime pressure point | Why a plain text LLM server is insufficient |
| --- | --- | --- | --- |
| AR→AR→generation speech graph | `vllm_omni/model_executor/models/qwen3_omni/pipeline.py` | low-latency intermediate transfer, codec stopping rules, waveform finalization | the final output is not a token string, and the middle stage should consume hidden/codec state rather than text |
| AR→diffusion image graph | `vllm_omni/model_executor/models/bagel/pipeline.py`, `vllm_omni/model_executor/models/hunyuan_image3/pipeline.py` | KV or latent conditioning, CFG companions, diffusion process scheduling | diffusion has denoising loops and parallelism knobs that do not match decode scheduling |
| Single-stage diffusion/unified graph | BAGEL single-stage and DiT-only variants | diffusion stage owns tokenizer/vision/VAE internals | the “stage” is a complete multimodal program, not a downstream renderer |
| TTS/audio-specialized graph | `qwen3_tts`, `higgs_audio_v2`, `cosyvoice3`, `voxcpm2`, `moss_tts_nano` registry entries | audio token/code semantics, speaker conditioning, custom voices | the input/output ABI is not equivalent to chat completion text |
| Research/variant graph | AR-only, DiT-only, thinker-only variants | partial graph deployment and debugging | a serving system must expose subgraphs without duplicating engine code |

This taxonomy is more useful than a “supported models” list because it predicts which subsystem you should study.  For example, if a new model is AR→diffusion, study `prompt_expand_func`, `cfg_kv_collect_func`, `omni_kv_config`, and diffusion clients.  If a model is streaming speech, study async chunk processors and `OmniSchedulingCoordinator`.

## 11. Architecture support is also capability routing

A multimodal model family often exposes more than one user-visible capability.  Qwen-style omni models can answer textually, produce speech, or analyze audio/video/image prompts.  BAGEL-like systems can reason in text and render images.  HunyuanImage3 variants can serve a full graph or only an AR/DiT subgraph.  vLLM-Omni encodes this with two mechanisms:

1. `final_output` and `final_output_type` on stages, which declare externally visible outputs.
2. request-level final-stage selection in the entrypoint/runtime, which decides how far a specific request should travel.

The architectural implication is that model support is not a single binary question: “can this checkpoint load?”  A higher-quality support layer answers: which graph suffixes are executable, which modalities can terminate early, and which intermediate representations are safe to skip?  This is why stage metadata includes final-output properties and why stage processors are not optional glue code.

## 12. Reading exercise: reconstruct a pipeline without running it

Choose one registered model family and reconstruct the following table from code only:

| Field | Question to answer |
| --- | --- |
| `model_type` | Which registry key selects this pipeline? |
| `hf_architectures` / predicate | Can the factory detect this model by architecture if `model_type` collides? |
| stage IDs | Are stages sequential, or can they be subgraphs/variants? |
| `execution_type` | Which scheduler/client family should each stage use? |
| `input_sources` | Which upstream stage provides data? |
| final outputs | Which stages can be returned to users, and in which modality? |
| processor hooks | Which functions define the semantic ABI between stages? |
| `omni_kv_config` | Is cross-stage cache transfer part of the architecture? |
| sampling constraints | Does the topology force stop tokens, detokenization behavior, or max-token behavior? |

After filling the table, open `vllm_omni/config/stage_config.py` and trace how that declaration becomes concrete `StageConfig` objects.  The goal is to see topology declarations as a small domain-specific language for multimodal serving.

## 13. Common mistakes when interpreting architecture support

- **Mistake: treating `model_stage` as cosmetic.** In practice it influences connector mode, processor selection, logs, and model-family behavior.
- **Mistake: assuming all final outputs are at the last stage.** Multiple stages can be externally visible, and the final stage can be request-dependent.
- **Mistake: assuming processor hooks are preprocessing only.** Some hooks transform next-stage input, collect CFG/KV structures, or select async-vs-sync handoff behavior.
- **Mistake: assuming diffusion is a terminal postprocessor.** Diffusion stages are configured, launched, scheduled, and profiled as runtime stages.
- **Mistake: assuming architecture detection is just `model_type`.** The factory has model-type, architecture, predicate, diffusers-class, and name-fallback paths because real model repositories are inconsistent.
