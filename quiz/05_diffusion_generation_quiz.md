# Quiz 05 — Diffusion and generation stages

Paired reading: `edu/docs/diffusion_and_generation_stages.md`

## Multiple choice

1. Why does vLLM-Omni treat diffusion as a stage execution type rather than simple postprocessing?
   - A. Diffusion has its own lifecycle, config, parallelism, scheduler/client behavior, and output modality.
   - B. Diffusion stages are always text-only.
   - C. Diffusion cannot run on GPUs.
   - D. Diffusion does not need inputs.

2. Which execution type best matches a Code2Wav-like stage that consumes codec-like information and emits audio?
   - A. `LLM_GENERATION`
   - B. `DIFFUSION` only
   - C. `README`
   - D. `StageDeployConfig` only

3. Which stage pattern best describes BAGEL-style image generation?
   - A. AR reasoning/conditioning followed by diffusion rendering
   - B. Static file copy only
   - C. Metrics export followed by logging only
   - D. Tokenizer-only execution

4. Which diffusion concern is not typically captured by tokens/sec alone?
   - A. VAE/DiT memory pressure and denoising latency
   - B. Markdown heading count
   - C. Git branch name
   - D. Python package metadata only

## Short answer

5. Compare `LLM_AR`, `LLM_GENERATION`, and `DIFFUSION` stage execution types using typical inputs, outputs, and scheduling/client behavior.

6. Explain AR→diffusion handoff semantics. Why is it richer than passing a caption string to a renderer?

7. Why are AR-only and DiT-only variants useful for HunyuanImage3-like systems?

8. List diffusion-specific parallelism or memory concepts that a text-only LLM server would not fully cover.

9. For an audio generation stage, what metrics would you report besides final waveform latency?

## Design scenarios

10. You need to serve an image model with a large DiT stage that OOMs when colocated with the AR stage. What deployment and measurement questions should you ask?

11. A diffusion stage starts quickly but final images are slow. Which parts of the diffusion subsystem and metrics would you inspect?
