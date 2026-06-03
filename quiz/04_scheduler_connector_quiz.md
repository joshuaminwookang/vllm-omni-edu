# Quiz 04 — Scheduler and connector internals

Paired readings: `edu/docs/scheduler_connector_deep_dive.md`, `edu/docs/staged_execution.md`

## Multiple choice

1. Which component centralizes connector creation, background I/O, custom processor loading, and KV transfer delegation for model runners?
   - A. `OmniConnectorModelRunnerMixin`
   - B. `README.md`
   - C. `PipelineConfig`
   - D. `OmniPrometheusMetrics`

2. What does `WAITING_FOR_INPUT` represent?
   - A. A downstream request is waiting for a full payload to arrive.
   - B. A request has finished successfully.
   - C. A tokenizer vocabulary is missing.
   - D. A stage is compiling CUDA kernels only.

3. What does async chunking primarily optimize?
   - A. Git history size
   - B. First-output latency for streaming-like downstream execution
   - C. Markdown link formatting
   - D. Hugging Face config naming

4. Why should scheduler code not call connector transport code directly?
   - A. Because scheduler code is never executed.
   - B. To preserve testability, transport flexibility, and layer separation.
   - C. Because all connectors are synchronous strings.
   - D. Because all requests are single-stage.

5. `omni_kv_config` is most relevant when:
   - A. A stage needs to send or receive KV/cache-like state across stage boundaries.
   - B. A quiz needs an answer key.
   - C. A static image is embedded in docs.
   - D. No stages exist.

## Short answer

6. Compare full payload, async chunk, and KV transfer. For each, name a workload pattern where it fits.

7. What invariants must hold for a request transitioning out of `WAITING_FOR_CHUNK`?

8. Explain how heterogeneous tensor parallelism complicates KV transfer.

9. Why is `should_accumulate_full_payload_output()` described as a structural gate?

10. What information does a scheduler need from connector outputs to avoid spinning on unavailable downstream inputs?

## Debugging exercises

11. A Stage 1 request remains in `WAITING_FOR_INPUT` forever. List a step-by-step debugging plan.

12. A request receives two chunks with the same request ID but downstream output is corrupted. What alignment issues should you inspect?

13. A downstream request runs before its upstream data has arrived. Which layer likely violated its responsibility?
