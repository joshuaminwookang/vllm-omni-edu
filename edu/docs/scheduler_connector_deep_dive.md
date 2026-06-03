# Scheduler and connector deep dive: WAITING states, chunks, full payloads, and KV transfer

## 1. The core problem

In vanilla LLM serving, a request is usually schedulable once tokenized input is available and KV/cache capacity exists.  In vLLM-Omni, a downstream stage may be known to the scheduler before its actual input has arrived.  That input might arrive as a full payload, a stream of chunks, or a KV-cache transfer.  Scheduling it too early wastes capacity or fails; scheduling it too late hurts latency.

The repo solves this with a separation:

- connector/model-runner code performs transport and reports readiness;
- scheduler coordinator code tracks waiting states;
- stage schedulers consume readiness without knowing transport internals.

## 2. The connector mixin

`vllm_omni/worker/omni_connector_model_runner_mixin.py` centralizes data-plane communication for model runners.  It owns connector creation, background I/O threads, custom processor loading, rank-aware slicing/merging, chunk registration, full-payload send/recv, and KV transfer delegation.

The mixin supports three families of transfer:

1. **Full payload:** producer accumulates a complete object and sends it once.
2. **Async chunk:** producer sends incremental chunks; consumer can start before full completion.
3. **KV cache:** producer/consumer exchange cache-like state through the KV transfer manager.

A model runner should not open sockets or mutate scheduler queues directly.  It should use the mixin and report `OmniConnectorOutput`.

## 3. Full payload mode

Full payload mode is conceptually simple: upstream produces an object, downstream waits until it arrives.  The complexity is lifecycle correctness.

`should_accumulate_full_payload_output()` protects the producer side.  It returns true only when:

- a custom downstream producer hook exists;
- async chunking is disabled;
- the stage is not itself a final output;
- a `model_stage` is present.

This prevents accidental full-payload behavior when only a consumer-side helper exists.  The docstring explicitly notes this structural gate because naming conventions alone are too fragile.

On the scheduler side, `process_pending_full_payload_inputs()` moves non-stage-0 waiting requests into `WAITING_FOR_INPUT`, registers minimal receive handles for the model runner, and moves requests back to `WAITING` once the connector reports arrival.

## 4. Async chunk mode

Async chunking is designed for latency-sensitive streaming pipelines, especially speech.  Instead of waiting for a full upstream output, the producer sends chunks as they become available.  The downstream scheduler tracks requests in `WAITING_FOR_CHUNK` and returns them to schedulable queues when chunks are ready.

Important details:

- The coordinator distinguishes terminal-ready chunks from intermediate chunks.
- The running queue may be trimmed back to `scheduler_max_num_seqs` without treating this as ordinary preemption that frees KV blocks.
- Chunk receive registration is explicit; background I/O must know which request/chunk to poll.
- Finished chunk state must be tracked so downstream stages do not wait for data that will never arrive.

Async chunking improves first-output latency but increases state complexity.  It should be introduced only when the model-family processor and downstream scheduler agree on chunk semantics.

## 5. KV transfer mode

KV transfer is neither full payload nor chunking.  It is cache/state movement, often with tensor-parallel rank semantics.  `omni_kv_config` declares whether a stage needs to send or receive cache and can include transfer criteria such as `prefill_finished`.

Stage startup injects connector config and inferred TP topology.  The connector mixin configures rank-aware key builders and payload slicers/mergers for heterogeneous TP.  The scheduler can then reason about KV readiness without owning the low-level tensor transfer.

KV transfer is especially important for AR→diffusion models where the diffusion stage consumes conditioning derived from AR cache or hidden state.  It is also a research frontier: cache placement and lifetime policies become cross-stage decisions.

## 6. Waiting-state invariants

A request in a downstream stage should be in exactly one meaningful state:

- `WAITING`: input is present and scheduler may admit it when capacity allows.
- `WAITING_FOR_INPUT`: full payload has not arrived.
- `WAITING_FOR_CHUNK`: async chunk has not arrived.
- `RUNNING`: stage is executing.
- terminal/finished/aborted: request should not be scheduled again.

Violations cause classic distributed-systems bugs:

- duplicate scheduling if a request is returned to `WAITING` twice;
- hangs if a request remains in `WAITING_FOR_INPUT` after payload arrival;
- memory leaks if finished requests are not removed from coordinator sets;
- incorrect output if a terminal chunk is treated as an intermediate chunk;
- capacity collapse if waiting requests occupy running slots.

## 7. Why the scheduler does not call connectors

This design has three benefits:

1. **Testability.** Scheduler tests can provide synthetic readiness sets without creating sockets or GPU payloads.
2. **Transport flexibility.** Connectors can change from local queues to ZMQ, RDMA-like transport, or custom distributed connectors without rewriting scheduling policy.
3. **Layering.** Schedulers reason about request states and capacity; connector code reasons about I/O, serialization, chunk indices, and rank-aware tensors.

This is the same style of abstraction that made PagedAttention useful: isolate the hard resource-management problem behind a stable interface.

## 8. Debugging guide

For a downstream request that never runs:

1. Confirm the stage graph says this stage should receive the request (`input_sources`, final-stage ID).
2. Confirm processor selection matches deployment mode (`async_chunk` true/false).
3. Confirm connector initialization logs show the expected custom processor path.
4. Confirm the producer emitted payload/chunk/KV data for the same request ID.
5. Confirm `OmniConnectorOutput` contains the readiness signal.
6. Confirm `OmniSchedulingCoordinator` moved the request back to `WAITING`.
7. Confirm the scheduler has capacity and no unrelated request is blocking the running queue.

For a downstream request that runs with wrong data:

1. Inspect processor return fields and tensor shapes.
2. Inspect request ID and chunk index alignment.
3. Check TP rank mapping and cache slicing.
4. Check whether async and sync processors have equivalent semantics.
5. Check whether original multimodal data was preserved for stages that require it.
