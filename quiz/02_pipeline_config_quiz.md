# Quiz 02 — Pipeline configuration and architecture support

Paired readings: `edu/docs/architecture_support.md`, `edu/docs/pipeline_config_deep_dive.md`

## Multiple choice

1. `StageExecutionType.LLM_AR` most directly indicates which kind of runtime behavior?
   - A. Diffusion denoising only
   - B. Autoregressive LLM-style execution using omni-aware AR scheduling
   - C. Static dataset preprocessing
   - D. Prometheus metrics export

2. Which `StagePipelineConfig` field tells the runtime that a stage can produce externally visible user output?
   - A. `input_sources`
   - B. `final_output`
   - C. `model_subdir`
   - D. `hf_config_name`

3. Why does the pipeline factory use `hf_architectures` and optional predicates in addition to `model_type`?
   - A. To randomly select stages for load balancing
   - B. To disambiguate real-world checkpoint configs whose `model_type` is generic, missing, or shared by multiple architectures
   - C. To disable all tokenizer loading
   - D. To enforce one-stage-only pipelines

4. What is the purpose of `_select_processor_funcs()` in the configuration layer?
   - A. It chooses async or sync stage processor functions based on resolved `async_chunk` mode.
   - B. It chooses CUDA device IDs.
   - C. It computes benchmark scores.
   - D. It downloads model weights.

5. Which item is primarily structural rather than deployment/runtime configuration?
   - A. `stage_id`
   - B. number of replicas
   - C. device placement
   - D. dtype override

## Short answer

6. Define `StagePipelineConfig` and `PipelineConfig` in your own words. Why does the repo need both?

7. Explain why sampling constraints are part of the stage ABI. Use a Talker or codec-style stage as an example.

8. Describe the model-detection fallback ladder. Why is it important for multimodal repositories?

9. What is dangerous about argparse defaults or top-level CLI overrides in a staged serving system?

10. In what sense is `stage_config.py` a “compiler” for the model-family graph?

## Design questions

11. You are adding an AR→diffusion model whose AR stage should transfer KV cache after prefill. Which `StagePipelineConfig` fields and related deploy/config concepts should you think about?

12. A new speech model can run in full-payload or streaming mode. How should its pipeline declaration expose both modes without changing orchestrator code?

13. A checkpoint reports `model_type="qwen2"` but is actually a new audio-omni architecture. How should you make pipeline selection robust?
