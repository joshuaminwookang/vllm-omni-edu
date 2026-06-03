# vLLM-Omni educational reading sequence

This directory is a guided, code-grounded reading sequence for LLM researchers who want to understand **multimodal efficient inference systems** through this repository.  It is written as a set of technical reports rather than quick-start notes: each document connects model architecture choices, staged execution, scheduler/connector mechanics, and engineering abstraction boundaries to concrete code paths in `vllm_omni/`.

The sequence is intentionally redundant in a productive way: the same runtime idea is revisited from several perspectives—architecture, request lifecycle, scheduler state, deployment configuration, diffusion execution, and experimental methodology—because efficient omni inference is a cross-layer problem.  A student should be able to move from a model-family `pipeline.py` declaration to the exact engine, scheduler, connector, and worker code that makes the declaration executable.

## Suggested reading order

### Part I — Build the mental model

1. [Codebase map: where the important abstractions live](codebase_map.md)
   - A directory-level map of the repo, emphasizing the boundaries between config, entrypoints, engine, schedulers, workers, diffusion, model executors, examples, tests, and docs.
2. [Omni architecture support: from model families to executable stage graphs](architecture_support.md)
   - How this repo represents very different omni-model families—Qwen-Omni speech stacks, AR→DiT image models, single-stage diffusion, and TTS pipelines—using one declarative topology layer.
3. [Pipeline configuration deep dive: the DSL that compiles model families into runtime stages](pipeline_config_deep_dive.md)
   - A field-by-field study of `StagePipelineConfig`, `PipelineConfig`, deploy config, CLI override merge, architecture detection, scheduler selection, and stage materialization.

### Part II — Follow a request through the runtime

4. [Request lifecycle: from user prompt to final multimodal output](request_lifecycle.md)
   - A trace of `Omni.generate()` and `AsyncOmniEngine.add_request()` through orchestrator messages, per-stage submission, output handling, and finalization.
5. [Efficient staged execution: orchestration, scheduling, transfer, and latency control](staged_execution.md)
   - How requests move across stages, how final output is selected, how payload/KV transfer is coordinated, and why this resembles but generalizes prefill/decode disaggregation.
6. [Scheduler and connector deep dive: WAITING states, chunks, full payloads, and KV transfer](scheduler_connector_deep_dive.md)
   - How model-runner connector outputs drive omni scheduler state transitions without letting scheduler code call transport code directly.

### Part III — Study the major subsystems

7. [Software architecture and abstractions: contracts that keep model code, runtime code, and deployment code separable](software_architecture.md)
   - The major object boundaries: `PipelineConfig`, `StageConfig`, `AsyncOmniEngine`, `StagePool`, schedulers, workers, connectors, processors, and diffusion clients.
8. [Diffusion and generation stages: serving non-token workloads as first-class stages](diffusion_and_generation_stages.md)
   - How diffusion stages and generation-style LLM stages differ from ordinary AR decode stages, and why image/audio/video generation require separate execution policies.
9. [Extending vLLM-Omni: adding a new omni model family without breaking the runtime](extending_and_testing_models.md)
   - A practical model-integration guide covering pipeline declarations, processor hooks, deploy YAML, examples, tests, debugging, and anti-patterns.

### Part IV — Research and evaluation

10. [Research implications and design patterns for next-generation multimodal serving](research_implications.md)
    - What this repo teaches about future efficient inference systems: modality-specialized execution, streaming chunk interfaces, KV reuse, deployment knobs, observability, and open research questions.
11. [Observability, experiments, and study assignments](observability_experiments.md)
    - Concrete experiments, reading exercises, profiling questions, and metrics that students can use to turn the codebase into a systems research lab.

## What to look for while reading the code

- **Topology is data, execution is code.** Model families enter through small `pipeline.py` files under `vllm_omni/model_executor/models/*/`, while common runtime behavior lives in `vllm_omni/engine/`, `vllm_omni/core/sched/`, and `vllm_omni/worker/`.
- **A stage is more than a model.** A stage has model weights, execution type, scheduler semantics, tokenizer ownership, modality requirements, transfer hooks, output type, sampling constraints, and deployment overrides.
- **The central inference problem is not just tokens/sec.** Omni systems also optimize first-audio latency, image diffusion handoff, hidden-state transfer, KV-cache transfer, heterogeneous tensor-parallel layouts, modality-specific memory, and graceful degradation when a request only needs a prefix of the full graph.
- **The repo is a living map of the research frontier.** Its abstractions echo recent systems work such as vLLM/PagedAttention, DistServe-style stage disaggregation, Sarathi/Sarathi-Serve chunked scheduling, and recent omni-model technical reports such as Qwen2.5-Omni and Qwen3-Omni.
- **Intermediate representations are the hard part.** The stage graph only works because model-family processors agree on precise payload semantics: hidden states, codec IDs, prompt expansions, diffusion conditioning, CFG companions, or KV caches.
- **Deployment choices change algorithms.** `async_chunk`, per-stage TP size, replica counts, diffusion parallel degrees, and connector settings do not merely tune throughput; they change when downstream stages become schedulable and what data they receive.

## External references used across the reading sequence

The documents cite these sources for research context, while the implementation claims are grounded in repository code paths:

- Kwon et al., **Efficient Memory Management for Large Language Model Serving with PagedAttention**, SOSP 2023 / arXiv:2309.06180, <https://arxiv.org/abs/2309.06180>.
- Zhong et al., **DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**, OSDI 2024 / arXiv:2401.09670, <https://arxiv.org/abs/2401.09670>.
- Agrawal et al., **Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve**, arXiv:2403.02310, <https://arxiv.org/abs/2403.02310>.
- Xu et al., **Qwen2.5-Omni Technical Report**, arXiv:2503.20215, <https://arxiv.org/abs/2503.20215>.
- Qwen Team, **Qwen3-Omni Technical Report**, arXiv:2509.17765, <https://arxiv.org/abs/2509.17765>.

## How to study this material

For each document, do three passes:

1. **Concept pass:** read the prose and draw the stage graph or dataflow in your own words.
2. **Code pass:** open the referenced files and locate the named class/function.  Write down what invariants the code enforces.
3. **Experiment pass:** choose one knob—`async_chunk`, final output modality, stage replica count, sampling constraint, processor hook, or diffusion parallel option—and predict which code paths change before running anything.

A strong student outcome is the ability to answer: “If I add a new model with AR audio understanding, a codec-token speech decoder, and optional image generation, which files must I touch, which runtime states matter, and which metrics prove that my integration is efficient?”
