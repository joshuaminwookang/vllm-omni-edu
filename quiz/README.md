# vLLM-Omni educational quizzes

This directory contains quizzes that pair with the `edu/docs/` reading sequence.  The questions are designed for LLM researchers, systems students, and contributors who want to test whether they can connect vLLM-Omni concepts to concrete repository structure and runtime behavior.

## How to use these quizzes

1. Read the matching document in `edu/docs/`.
2. Answer the quiz without looking at the answer key.
3. Re-open the code paths named in the question and revise your answer with file/function evidence.
4. Use `answer_key.md` only after attempting the questions.

## Quiz map

| Quiz | Paired reading | Focus |
| --- | --- | --- |
| [`01_codebase_map_quiz.md`](01_codebase_map_quiz.md) | `edu/docs/codebase_map.md` | Directory structure, layer boundaries, cross-directory paths |
| [`02_pipeline_config_quiz.md`](02_pipeline_config_quiz.md) | `edu/docs/architecture_support.md`, `edu/docs/pipeline_config_deep_dive.md` | Stage graph DSL, registry, deploy merge, model detection |
| [`03_request_lifecycle_quiz.md`](03_request_lifecycle_quiz.md) | `edu/docs/request_lifecycle.md`, `edu/docs/staged_execution.md` | Prompt submission, final-stage routing, orchestrator messages, output handling |
| [`04_scheduler_connector_quiz.md`](04_scheduler_connector_quiz.md) | `edu/docs/scheduler_connector_deep_dive.md`, `edu/docs/staged_execution.md` | Full payload, async chunks, KV transfer, waiting states |
| [`05_diffusion_generation_quiz.md`](05_diffusion_generation_quiz.md) | `edu/docs/diffusion_and_generation_stages.md` | Diffusion stages, generation stages, AR→DiT and Code2Wav-style execution |
| [`06_extending_testing_quiz.md`](06_extending_testing_quiz.md) | `edu/docs/extending_and_testing_models.md`, `edu/docs/software_architecture.md` | Adding new model families, ownership rules, test strategy |
| [`07_research_experiments_quiz.md`](07_research_experiments_quiz.md) | `edu/docs/research_implications.md`, `edu/docs/observability_experiments.md` | Experimental design, metrics, benchmark methodology, open research questions |
| [`answer_key.md`](answer_key.md) | All quizzes | Suggested answers and grading notes |

## Suggested grading rubric

- **Conceptual correctness (40%)**: the answer identifies the right subsystem and invariant.
- **Code grounding (30%)**: the answer cites the correct directory, class, function, or field from the repo.
- **Systems reasoning (20%)**: the answer explains performance, correctness, or reliability consequences.
- **Clarity (10%)**: the answer is concise and unambiguous.

For short-answer questions, a strong response should usually mention both the conceptual role and the relevant code path.  For design questions, a strong response should identify at least one invariant that could break.
