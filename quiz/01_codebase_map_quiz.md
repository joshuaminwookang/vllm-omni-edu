# Quiz 01 — Codebase map and subsystem boundaries

Paired reading: `edu/docs/codebase_map.md`

## Multiple choice

1. Which directory is the best first stop for understanding how model-family stage graphs are declared and compiled into stage configs?
   - A. `vllm_omni/assets/`
   - B. `vllm_omni/config/`
   - C. `vllm_omni/metrics/`
   - D. `vllm_omni/platforms/`

2. Which directory contains the model-family-specific functions that define the semantic ABI between adjacent stages?
   - A. `vllm_omni/model_executor/stage_input_processors/`
   - B. `vllm_omni/entrypoints/`
   - C. `vllm_omni/profiler/`
   - D. `requirements/`

3. In the reading sequence, why should a student avoid starting with every neural-network model implementation line-by-line?
   - A. The model files are unused.
   - B. The runtime contracts and stage graph provide the mental map needed to understand model internals productively.
   - C. The codebase has no model implementations.
   - D. The model files are generated and cannot be read.

4. Which pair most accurately describes the relationship between `vllm_omni/engine/` and `vllm_omni/worker/`?
   - A. `engine/` stores static assets; `worker/` stores Markdown docs.
   - B. `engine/` orchestrates stage startup/routing/messages; `worker/` implements GPU/model-runner behavior and connector mixins.
   - C. `engine/` contains only diffusion code; `worker/` contains only CLI code.
   - D. They are aliases for the same subsystem.

## Short answer

5. Explain the difference between the “model declaration path” and the “request execution path.” Name at least four files or directories involved in each.

6. What are the three cross-layer contracts highlighted in the codebase map? For each, state one bug that could occur if the contract is violated.

7. A downstream stage never receives input. Which cross-directory path should you inspect first: model declaration, request execution, or transfer/readiness? Explain why.

## Applied tracing

8. You are given a new registered model name and asked to draw its stage graph without running weights. List the exact code-reading sequence you would follow.

9. A user reports that final outputs appear but per-stage transfer metrics look wrong. Which directories are likely relevant, and why?

10. In one paragraph, explain the phrase: “model-family code declares what stages exist; runtime code decides when, where, and how those stages execute.”
