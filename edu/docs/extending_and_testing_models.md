# Extending vLLM-Omni: adding a new omni model family without breaking the runtime

## 1. Integration philosophy

A new model family should enter vLLM-Omni by declaring a graph and implementing model-specific semantics at the edges.  Avoid editing generic orchestrator/scheduler code unless the model truly introduces a new runtime concept.  Most model integrations require:

1. model executor code;
2. stage input/output processors;
3. a `pipeline.py` declaration;
4. a registry entry;
5. deploy defaults;
6. examples;
7. tests.

## 2. Step 1: describe the semantic graph

Before writing code, draw the graph:

- What are the stages?
- Which stages are `LLM_AR`, `LLM_GENERATION`, or `DIFFUSION`?
- Which stage owns the tokenizer?
- Which stages require original multimodal data?
- Which outputs are user-visible, and in which modality?
- Does any request stop early?
- Does a downstream stage require hidden states, token IDs, codec IDs, KV cache, or full tensors?
- Does the model require streaming chunks?

If the graph is unclear, the code will be unclear.

## 3. Step 2: implement or wrap model stages

Model code belongs under `vllm_omni/model_executor/models/<family>/`.  Keep serving-specific topology in `pipeline.py`; keep neural-network implementation in model files; keep inter-stage conversion in `stage_input_processors`.  This separation makes it possible to change graph topology without editing model layers.

If the model repository has subdirectories for components, use `model_subdir` and `tokenizer_subdir` in `StagePipelineConfig` rather than hard-coding paths in runtime code.

## 4. Step 3: write processor hooks

Processor hooks are the semantic ABI.  They should be explicit about:

- input object type;
- required keys/fields;
- tensor shape and dtype;
- device assumptions;
- whether returned payload is full, chunked, or cache-like;
- how request IDs and chunk indices are preserved;
- what happens on empty or malformed input.

Name hooks consistently.  If you provide both async and sync variants, verify they produce semantically equivalent downstream inputs, modulo chunking.

## 5. Step 4: declare `PipelineConfig`

Create `vllm_omni/model_executor/models/<family>/pipeline.py` with one or more `PipelineConfig` objects.  Include variants when useful:

- full graph;
- stage-only debug graph;
- text/thinker-only graph;
- diffusion-only graph;
- low-latency streaming graph;
- quality-oriented full-payload graph.

For each `StagePipelineConfig`, set:

- `stage_id` and `input_sources`;
- `model_stage`;
- `execution_type`;
- final-output fields;
- tokenizer and multimodal requirements;
- processor hook paths;
- sampling constraints;
- KV/cache config if needed.

## 6. Step 5: add registry and detection metadata

Add the pipeline to `vllm_omni/config/pipeline_registry.py`.  If the model’s Hugging Face config has a generic or colliding `model_type`, set `hf_architectures` and possibly `hf_config_predicate` in `PipelineConfig`.  If it is a diffusers-style model, use diffusers class metadata where appropriate.

Do not rely only on naming conventions unless no better metadata exists.

## 7. Step 6: deploy defaults and CLI behavior

Add deploy YAML if the model needs nontrivial defaults: devices, dtype, TP size, async chunk, default sampling params, connector edges, or diffusion parallel settings.  Then verify CLI overrides do not accidentally erase deploy choices.  Pay special attention to explicit `False` values such as `async_chunk=False`; they must not fall through as if unspecified.

## 8. Step 7: examples

Add examples that cover every graph branch:

- text-only request if supported;
- target non-text modality request;
- multimodal input request;
- batch request;
- async chunk mode if supported;
- custom voice/speaker/image/video assets if relevant;
- online serving if the model should be served via API.

Examples are integration tests for humans.  They reveal prompt schema expectations that are not obvious from model code.

## 9. Step 8: tests

At minimum, add tests for:

- pipeline registry resolution;
- stage config materialization;
- deploy/CLI override precedence;
- processor hook behavior on synthetic payloads;
- final-stage computation for modality combinations;
- scheduler waiting-state behavior if using chunks/full payload;
- KV config injection if using cache transfer;
- example smoke tests where feasible.

If the model is too large for CI, write tests around config and processor units that do not require weights.

## 10. Anti-patterns

- Putting model-family tensor conversion inside the orchestrator.
- Making scheduler code call connector transport directly.
- Encoding device placement in `pipeline.py` instead of deploy config.
- Treating detokenization defaults as harmless for intermediate stages.
- Assuming every request must execute every stage.
- Ignoring heterogeneous TP when transferring KV/cache.
- Adding a registry entry without architecture detection metadata for colliding configs.
- Providing only one example that covers the happy path.

## 11. Minimal pull request checklist

Before submitting an integration:

1. Can `StageConfigFactory.create_from_model()` select the right pipeline?
2. Do all processor dotted paths import successfully?
3. Do stage IDs and input sources form the intended graph?
4. Do default sampling params match intermediate-output semantics?
5. Does final-stage routing skip unnecessary work?
6. Does the model work with both configured deploy defaults and explicit CLI overrides?
7. Do logs identify each stage and connector mode clearly?
8. Are tests small enough to run without downloading massive weights where possible?

This checklist keeps model research code compatible with serving-system invariants.
