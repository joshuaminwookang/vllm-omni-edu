# Quiz 07 — Research implications, observability, and experiments

Paired readings: `edu/docs/research_implications.md`, `edu/docs/observability_experiments.md`

## Multiple choice

1. Why is tokens/sec insufficient for evaluating omni inference?
   - A. Omni requests can involve non-token outputs, transfer waits, diffusion denoising, waveform rendering, and modality-specific first-output latency.
   - B. Tokens/sec is always zero.
   - C. Tokens/sec measures Markdown formatting only.
   - D. Omni systems do not use GPUs.

2. Which experiment best tests final-stage routing efficiency?
   - A. Compare text-only and audio-output requests on a multi-final-stage pipeline and measure skipped downstream work.
   - B. Count README words only.
   - C. Rename a directory.
   - D. Disable all logging and never inspect stages.

3. Which metric is especially important for streaming speech systems?
   - A. Time to first audio chunk
   - B. Number of Markdown headings
   - C. Local timezone
   - D. Git author email

4. Which research question maps to `StagePool` and per-stage metrics?
   - A. Which stage should be replicated under a fixed GPU budget?
   - B. Which license header is longest?
   - C. Which shell prompt is used?
   - D. Which docs file has the shortest title?

## Short answer

5. Design a benchmark suite with at least five workload classes for multimodal efficient inference.

6. What metrics should be reported for an AR→diffusion pipeline beyond end-to-end image latency?

7. Explain why cross-stage cache placement is a research problem rather than a simple implementation detail.

8. What should a student include in a project report about vLLM-Omni serving performance?

9. How could adaptive chunking become a research contribution? What code abstractions would it interact with?

10. Describe a fault-tolerant degradation policy for an overloaded audio or diffusion stage. What invariants must it preserve?

## Capstone design prompt

11. You have 8 GPUs and must serve a workload mix of 60% text-only, 30% audio-output, and 10% image-output requests on an omni model with a Thinker, Talker, Code2Wav, and DiT branch. Propose:
    - a stage graph;
    - which stages can be final outputs;
    - which stages should be replicated or isolated first;
    - what transfer modes you would try;
    - what metrics would prove your deployment is efficient;
    - what failure modes you would test.
