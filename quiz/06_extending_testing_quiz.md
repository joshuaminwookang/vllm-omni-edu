# Quiz 06 — Extending and testing model integrations

Paired readings: `edu/docs/extending_and_testing_models.md`, `edu/docs/software_architecture.md`

## Multiple choice

1. Where should model-family tensor conversion between stages usually live?
   - A. `vllm_omni/model_executor/stage_input_processors/`
   - B. `vllm_omni/engine/orchestrator.py`
   - C. `.git/hooks/`
   - D. `requirements/common.txt`

2. Which change most likely violates layer ownership?
   - A. Adding a model-family `pipeline.py`
   - B. Putting Qwen-specific payload conversion directly inside the generic orchestrator
   - C. Adding an example script
   - D. Adding config tests

3. A minimal model-integration PR should include tests for which category even if weights are too large for CI?
   - A. Config and processor behavior on synthetic payloads
   - B. GPU thermal paste
   - C. Browser CSS only
   - D. Random file names

4. Which file is the central built-in registry for pipeline modules?
   - A. `vllm_omni/config/pipeline_registry.py`
   - B. `quiz/README.md`
   - C. `requirements/cuda.txt`
   - D. `docs/assets/`

## Short answer

5. List the recommended steps for adding a new omni model family.

6. What should a processor hook document or validate about its inputs and outputs?

7. Why should device placement usually live in deploy config rather than `pipeline.py`?

8. Explain the “narrow waist” architecture in the repo. Name at least four objects or message types that participate in the waist.

9. What are three anti-patterns when adding a new model family?

10. What is the difference between a topology change and a scheduler change in code review?

## PR review scenarios

11. A contributor adds a registry entry but no `hf_architectures` for a checkpoint family known to use a generic `model_type`. What review comment would you leave?

12. A new stage processor returns CPU tensors for one path and GPU tensors for another without documentation. What risks should the reviewer call out?

13. A new model supports text-only and audio-output requests, but its examples only cover one audio prompt. What additional examples/tests would you request?
