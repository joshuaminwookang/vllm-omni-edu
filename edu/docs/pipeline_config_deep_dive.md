# Pipeline configuration deep dive: the DSL that compiles model families into runtime stages

## 1. The configuration layer as a DSL

`vllm_omni/config/stage_config.py` is best read as a small domain-specific language compiler.  Its input is a mixture of:

- a model-family `PipelineConfig` from `vllm_omni/model_executor/models/*/pipeline.py`;
- deploy YAML describing runtime choices;
- CLI/API overrides;
- Hugging Face config metadata used for detection.

Its output is a list of concrete `StageConfig` objects that stage startup code can turn into engine arguments, runtime settings, scheduler classes, and stage metadata.

This is why the file contains both declarative data classes and operational helper functions.  It is not merely parsing YAML; it is compiling a model-family graph into executable stage plans.

## 2. Structural fields versus runtime fields

A reliable way to read stage config code is to distinguish structural model facts from runtime deployment choices.

Structural fields live in `StagePipelineConfig`:

- `stage_id`, `model_stage`, `execution_type`, `input_sources`;
- `final_output`, `final_output_type`;
- `owns_tokenizer`, `requires_multimodal_data`;
- `hf_config_name`, `model_arch`, `model_subdir`, `tokenizer_subdir`;
- processor hooks and KV/cache declarations.

Runtime fields live in deploy config or overrides:

- device placement;
- number of replicas;
- dtype, quantization, executor backend;
- tensor/data/pipeline parallel sizes;
- `async_chunk`;
- default sampling params;
- connector settings;
- diffusion parallel options.

The distinction matters because changing a structural field can alter correctness, while changing a runtime field should usually alter performance or placement.  There are exceptions: `async_chunk` is a deployment field that changes the selected processor hook and scheduler readiness behavior.

## 3. Execution type compilation

`StageExecutionType` collapses model-level intent into runtime policy:

- `LLM_AR` maps to `StageType.LLM` plus worker type `ar` and an omni AR scheduler.
- `LLM_GENERATION` maps to `StageType.LLM` plus worker type `generation` and a generation scheduler.
- `DIFFUSION` maps to `StageType.DIFFUSION` and bypasses vLLM token scheduler selection.

The helper `_resolve_execution_mode()` provides the legacy `(stage_type, worker_type)` tuple, while `_resolve_scheduler()` chooses scheduler classes for LLM-like stages.  This is a compact but important “compiler pass”: it turns a semantic stage category into concrete runtime fields.

## 4. Processor selection is deployment-sensitive

`_select_processor_funcs()` chooses `(input_proc, next_stage_proc)` based on the stage declaration and resolved `async_chunk` mode:

- if async chunking is enabled and an async next-stage processor exists, use that next-stage processor;
- if async chunking is disabled and a sync input processor exists, use that consumer input processor;
- otherwise fall back to the default custom processor paths.

This is one of the most important details in the repo.  The same model-family graph can have different dataflow semantics depending on `async_chunk`.  For speech systems, this can distinguish first-audio-optimized streaming behavior from simpler full-payload behavior.  For students, it is a concrete example of deployment configuration affecting algorithmic execution, not just GPU placement.

## 5. Engine args construction

`_build_engine_args()` assembles the flat `yaml_engine_args` dictionary for each stage.  It starts with model architecture and stage output type, includes selected next-stage processor hooks, applies model/tokenizer subdirectory indirections, propagates pipeline-wide deploy fields, overlays per-stage deploy fields, materializes resolved `async_chunk`, and copies `omni_kv_config`.

The result is intentionally flat because downstream vLLM engine args expect many top-level fields.  However, the source of each field is layered.  When debugging a stage initialization problem, ask:

1. Did the field originate from the frozen pipeline declaration?
2. Did deploy YAML override it?
3. Did CLI/API override it globally?
4. Did a `stage_<id>_<field>` override replace it only for this stage?
5. Did stage startup mutate it further, for example by injecting connector config or inferred KV topology?

## 6. Extras construction

`_build_extras()` combines sampling constraints and connector-like extras.  Sampling constraints from a pipeline declaration are more than defaults; they can enforce stage correctness.  Qwen3-Omni’s Talker disables detokenization and sets stop token IDs because its output is an intermediate codec-like sequence, not user text.  BAGEL uses prompt expansion and KV/cache hooks to prepare diffusion conditioning.

Students should treat default sampling params as part of the stage ABI.  If a stage processor expects raw token IDs but detokenization is accidentally enabled, downstream payload semantics may break.

## 7. Deploy merge and platform overlays

The deploy merge path allows pipeline-wide settings and per-stage settings to coexist.  Platform overlays can adjust devices, env variables, and selected runtime fields.  Dict-valued deep-merge behavior prevents a platform override from accidentally erasing sibling keys such as sampling parameters.

This reflects a production-serving reality: the same model graph may run differently on CUDA, ROCm, XPU, NPU, single-node, multi-node, or colocated deployment.  Encoding hardware assumptions directly into `pipeline.py` would make research iteration brittle.

## 8. Model detection fallback ladder

`StageConfigFactory._auto_detect_model_type()` is robust because real model repositories are inconsistent.  The detection ladder includes:

1. standard Hugging Face config loading and `model_type`;
2. raw `config.json` fallback;
3. singular `architecture` fallback for nonstandard configs;
4. diffusers-style `model_index.json` `_class_name` matching;
5. basename matching against registered pipeline keys.

Then `create_from_model()` can match registry entries by `model_type` or by `hf_architectures`, optionally using `hf_config_predicate` to disambiguate sibling generations.

The research lesson is that serving systems need “messy world” compatibility layers.  Architecture support is not only neural-network code; it is also robust model identification.

## 9. CLI override hygiene

`build_stage_runtime_overrides()` prevents orchestrator-only fields and shared fields from leaking into per-stage runtime configs.  It also supports `stage_<id>_<param>` overrides.  This is a subtle but important safety feature.  Without it, top-level CLI defaults could silently override deploy YAML or inject invalid stage-local engine args.

The entrypoint deprecation note around `from_cli_args()` also reflects this problem: parser defaults can accidentally look like explicit user choices.  Correct omni deployment requires preserving the difference between “user typed this” and “argparse filled a default.”

## 10. Study assignment: write a pseudo-compiler trace

Pick `QWEN3_OMNI_PIPELINE` and write the output of each conceptual compiler pass:

1. structural pipeline stages;
2. resolved async mode;
3. selected processor functions;
4. stage type and worker type;
5. scheduler class;
6. engine args;
7. extras/default sampling params;
8. runtime overrides;
9. final `StageConfig` list;
10. `StageMetadata` extracted at startup.

If you can do this without running the model, you understand the configuration layer.
