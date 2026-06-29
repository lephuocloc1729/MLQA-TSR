# Week 5 Issues - VLSP Post-Submission, Real Benchmark, And Free-Form QA

This issue pack extends the four-week course prototype into a stronger
post-submission benchmark and product demo phase.

The goal is no longer only to show a working scaffold. The goal is:

```text
locked internal validation
  -> real VLM benchmark
  -> VLSP post-submission package for Subtask 1 + Subtask 2
  -> official/post-submission score artifact
  -> free-form upload QA demo
  -> report update with honest metric separation
```

Keep the reporting boundary strict:

- Subtask 1 score = retrieval benchmark.
- Subtask 2 score = constrained QA benchmark with `A/B/C/D` or `Đúng/Sai`.
- Free-form QA = product extension/demo, evaluated with qualitative cases and
  citation inspection, not VLSP official accuracy.
- QLoRA remains optional unless it beats the base VLM on the same validation
  split with a comparable artifact.

Replace `M1`-`M4` with real names before creating GitHub issues.

## Entry Gate

Before starting these issues, confirm:

- `release/v0.1.0-course` is merged or the working tree is clean.
- `make release-check` passes in the project `.venv`.
- Qdrant can start locally with `make qdrant-up`.
- Raw public/private test files exist under `data/raw/`.
- Real VLM backend or GPU plan is agreed before any expensive full run.
- Codabench post-submission is actually available to the team account.

## W5-01 - VLSP Submission Packager For Task 1 And Task 2

Labels: `week-5`, `P0`, `submission`, `evaluation`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M1` / `M4`

Branch: `feat/w5-01-vlsp-submission-packager`

PR title: `feat(eval): package VLSP task1 and task2 submissions`

Depends on: `W4-03`, `W4-06`

### Description

Implement the exact VLSP post-submission packaging format. The system must
write:

```text
submission_task1.json
submission_task2.json
submission.zip
```

The zip must contain exactly the two JSON files above at the top level. This
issue is about format safety and validation, not improving model quality.

### Project Files To Change

- `src/competition_submission.py` (new)
- `src/submission.py` if reusable Task 2 helpers should be refactored
- `configs/config.yaml` only if path names need aliases
- `scripts/evaluate.sh`
- `README.md`
- `docs/experiments.md`
- `tests/test_competition_submission.py` (new)

### Reference Files To Study

- `src/submission.py`: current Task 2 answer converter and validation style.
- `configs/config.yaml`: public/private task input paths.
- `data/raw/README.txt`: local dataset notes.
- `data/raw/public_test/vlsp_2025_public_test_task1.json`: Task 1 input shape.
- `data/raw/public_test/vlsp_2025_public_test_task2.json`: Task 2 input shape.
- `data/raw/private_test/Task 1 Submission File/vlsp2025_submission_task1.json`:
  private Task 1 template shape.
- `data/raw/private_test/Task 2 Submission File/vlsp2025_submission_task2.json`:
  private Task 2 template shape.
- `../LexiSignVQA-main/src/eval/sub_task_1.py`: citation/retrieval output
  expectations.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: answer normalization and
  valid-label expectations.

### Implementation Notes

- Support `--set-name public_test` and `--set-name private_test`.
- Support `--task task1`, `--task task2`, and `--task both`.
- Task 1 rows must preserve `id`, `image_id`, `question`, and
  `relevant_articles`.
- Task 2 rows must preserve `id`, `image_id`, `question`, `question_type`,
  `choices` when present, `relevant_articles`, and `answer`.
- `relevant_articles` must be a list of `{law_id, article_id}` objects.
- Multiple-choice answers must be exactly `A/B/C/D`.
- Yes/No answers must be exactly `Đúng/Sai`.
- Missing predictions must fail validation unless `--allow-missing --dry-run`
  is explicitly used.
- Do not silently fill missing answers with `A`, `Đúng`, or any default label.
- Do not include extra files inside `submission.zip`.
- Keep generated submissions under `data/outputs/submissions/` and ignored by
  Git.

### Expected Output

Example commands:

```bash
python -m src.competition_submission \
  --set-name private_test \
  --task both \
  --task1-predictions data/outputs/competitions/private_task1_predictions.jsonl \
  --task2-predictions data/outputs/competitions/private_task2_predictions.jsonl \
  --output-dir data/outputs/submissions/vlsp_private

python -m src.competition_submission \
  --pack data/outputs/submissions/vlsp_private \
  --output data/outputs/submissions/submission.zip
```

Generated files:

```text
data/outputs/submissions/vlsp_private/submission_task1.json
data/outputs/submissions/vlsp_private/submission_task2.json
data/outputs/submissions/submission.zip
```

### Acceptance Criteria

- [ ] Task 1 JSON rows include `relevant_articles` and never include `answer`.
- [ ] Task 2 JSON rows include `answer` and `relevant_articles`.
- [ ] All required public/private IDs are present unless dry-run missing is
  explicitly allowed.
- [ ] Invalid answers fail validation.
- [ ] Invalid or malformed citations fail validation.
- [ ] `submission.zip` contains exactly `submission_task1.json` and
  `submission_task2.json`.
- [ ] The command prints a validation summary with row counts and missing IDs.
- [ ] No generated submission files are committed.

### Tests

- Unit test valid Task 1 conversion.
- Unit test valid Task 2 conversion.
- Unit test missing Task 1 citation fails.
- Unit test invalid Task 2 answer fails.
- Unit test zip file contains exactly the required file names.
- Unit test private-test path resolution with spaces in directory names.
- Unit test `--allow-missing` is accepted only with `--dry-run`.

### Verification

```bash
python -m pytest tests/test_competition_submission.py tests/test_submission.py -q
python -m src.competition_submission --help
python -m src.competition_submission \
  --set-name public_test \
  --task both \
  --task1-predictions tests/fixtures/tiny_task1_predictions.jsonl \
  --task2-predictions tests/fixtures/tiny_predictions.jsonl \
  --output-dir data/outputs/submissions/smoke \
  --allow-missing \
  --dry-run
git status --ignored --short data/outputs/submissions
git diff --check
```

## W5-02 - Public/Private Test Runner For VLSP Subtasks

Labels: `week-5`, `P0`, `benchmark`, `retrieval`, `vlm`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M4` / `M2`

Branch: `feat/w5-02-vlsp-test-runner`

PR title: `feat(pipeline): run VLSP public and private test predictions`

Depends on: `W5-01`, `W3-03`, `W4-02`

### Description

Add a runner that produces prediction JSONL artifacts for VLSP public/private
test inputs. These files do not have gold labels, so the runner must not depend
on `answer` or gold `relevant_articles`.

Task 1 should run retrieval only. Task 2 should run retrieval plus the real VLM
or a clearly marked mock smoke mode.

### Project Files To Change

- `src/pipeline.py`
- `src/data_utils.py`
- `src/retrieval.py` if test image-path resolution needs helper changes
- `configs/experiments/vlsp_task1_retrieval.yaml` (new)
- `configs/experiments/vlsp_task2_structured_real.yaml` (new)
- `scripts/evaluate.sh`
- `README.md`
- `tests/test_vlsp_test_runner.py` (new)

### Reference Files To Study

- `src/pipeline.py`: benchmark runtime, output artifact shape, and model error
  rows.
- `src/data_utils.py`: split/sample loading and image path attachment.
- `src/retrieval.py`: evidence retrieval and fusion logic.
- `src/submission.py`: required-sample path resolution.
- `data/raw/public_test/` and `data/raw/private_test/`: actual test input
  layout.
- `../LexiSignVQA-main/src/core/sub_task_1.py`: retrieval-stage orchestration.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: answer-generation
  orchestration.

### Implementation Notes

- Add a mode such as:
  `python -m src.pipeline --mode vlsp-test --set-name private_test --task task1`.
- Supported set names: `public_test`, `private_test`.
- Supported tasks: `task1`, `task2`.
- Task 1 output rows should include query fields and predicted
  `relevant_articles`.
- Task 2 output rows should include query fields, evidence, prediction answer,
  citations, parse status, latency, and model metadata.
- For Task 2 real runs, `mock=false` and `include_image=true` must be written
  into artifacts.
- If the real model call fails, write explicit error rows instead of fake
  answers.
- Keep `--limit` support for smoke runs.
- Keep output paths configurable and ignored under `data/outputs/competitions/`.

### Expected Output

```bash
python -m src.pipeline \
  --mode vlsp-test \
  --set-name private_test \
  --task task1 \
  --config configs/experiments/vlsp_task1_retrieval.yaml \
  --output data/outputs/competitions/private_task1_predictions.jsonl

python -m src.pipeline \
  --mode vlsp-test \
  --set-name private_test \
  --task task2 \
  --config configs/experiments/vlsp_task2_structured_real.yaml \
  --output data/outputs/competitions/private_task2_predictions.jsonl
```

### Acceptance Criteria

- [ ] Public and private test inputs can be loaded without gold labels.
- [ ] Image paths resolve for public/private images.
- [ ] Task 1 runner does not call the VLM.
- [ ] Task 2 runner calls the configured VLM only when `mock=false`.
- [ ] Missing backend credentials fail clearly before a full expensive run.
- [ ] Per-sample model errors are recorded as invalid/error rows.
- [ ] `--limit` smoke runs work for both tasks.
- [ ] Output artifacts can be passed to W5-01 packager.

### Tests

- Unit test public/private sample loading with synthetic templates.
- Unit test Task 1 runner returns citations and no answer.
- Unit test Task 2 mock runner returns valid answer shape.
- Unit test missing image path error message.
- Unit test missing live backend credentials fail before generating fake
  answers.
- Unit test output row IDs match input order.

### Verification

```bash
python -m pytest tests/test_vlsp_test_runner.py tests/test_pipeline.py -q
make qdrant-up
make preprocess
make index
python -m src.pipeline \
  --mode vlsp-test \
  --set-name public_test \
  --task task1 \
  --limit 5 \
  --output data/outputs/competitions/public_task1_smoke.jsonl
python -m src.competition_submission \
  --set-name public_test \
  --task task1 \
  --task1-predictions data/outputs/competitions/public_task1_smoke.jsonl \
  --allow-missing \
  --dry-run
git diff --check
```

## W5-03 - Real VLM Backend Benchmark With Qwen2.5-VL

Labels: `week-5`, `P0`, `vlm`, `benchmark`, `needs-gpu`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M3` / `M4`

Branch: `experiment/w5-03-real-vlm-benchmark`

PR title: `experiment: run real Qwen2.5-VL benchmark for VLSP QA`

Depends on: `W5-02`, `W3-03`

### Description

Run a real, non-mock VLM benchmark with image input. The preferred model is
`Qwen/Qwen2.5-VL-7B-Instruct` through an OpenAI-compatible endpoint. If 7B is
not stable on the available GPU, fall back to `Qwen/Qwen2.5-VL-3B-Instruct` and
record the reason.

The output must be suitable for Task 2 submission and for internal validation
comparison.

### Project Files To Change

- `configs/experiments/vlsp_task2_qwen25vl_7b.yaml` (new)
- `configs/experiments/vlsp_task2_qwen25vl_3b.yaml` (new fallback)
- `.env.example`
- `docs/experiments.md`
- `docs/model-card.md`
- `scripts/evaluate.sh`
- `tests/test_experiment_config.py` if config validation needs extension

### Reference Files To Study

- `src/vlm.py`: OpenAI-compatible backend and image-message construction.
- `src/pipeline.py`: `mock=false` artifact behavior and error rows.
- `configs/experiments/w4_structured_rag.yaml`: current real structured RAG
  config.
- `docs/report.md`: current limitations around missing real metrics.
- Official Qwen2.5-VL model card/serving docs used by the team GPU host.

### Implementation Notes

- Keep benchmark temperature `0.0`.
- Use `include_image: true`.
- Use `max_new_tokens` high enough to avoid truncating JSON; start with `512`.
- Run in stages:
  1. `--limit 1` connectivity smoke;
  2. `--limit 10` parse/citation smoke;
  3. full internal validation;
  4. public/private test prediction only after validation looks stable.
- Save backend/model name, GPU host, dtype/quantization, max tokens, latency,
  parse success, invalid JSON, truncated output, and error IDs.
- Do not tune on private-test post-submission score repeatedly if submission
  attempts are limited.
- Keep API keys only in `.env` or shell environment variables.

### Expected Output

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://<gpu-host>:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export OPENAI_COMPATIBLE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"

python -m src.pipeline \
  --mode benchmark \
  --config configs/experiments/vlsp_task2_qwen25vl_7b.yaml \
  --limit 10

python -m src.evaluate \
  --config configs/experiments/vlsp_task2_qwen25vl_7b.yaml \
  --predictions data/outputs/experiments/vlsp_task2_qwen25vl_7b.jsonl
```

### Acceptance Criteria

- [ ] A `mock=false` internal validation artifact exists.
- [ ] Artifact records `include_image=true`.
- [ ] Parse success, invalid JSON, truncation, and latency are recorded.
- [ ] At least one successful image+question model call is documented.
- [ ] Full validation run is attempted only after `--limit 10` smoke is stable.
- [ ] If 7B fails, fallback to 3B is documented with exact error reason.
- [ ] No API secrets, raw outputs with private keys, or model weights are
  committed.

### Tests

- Unit test config loading for 7B and 3B configs.
- Unit test missing credential error remains clear.
- Unit test `include_image=true` appears in benchmark metadata.
- Manual GPU smoke evidence: command, model name, hardware, latency, and one
  sanitized output row.

### Verification

```bash
python -m pytest tests/test_experiment_config.py tests/test_vlm_backend.py -q
python -m src.pipeline \
  --mode benchmark \
  --config configs/experiments/vlsp_task2_qwen25vl_7b.yaml \
  --limit 1
python -m src.evaluate \
  --config configs/experiments/vlsp_task2_qwen25vl_7b.yaml \
  --predictions data/outputs/experiments/vlsp_task2_qwen25vl_7b.jsonl
git diff --check
```

## W5-04 - Retrieval Fusion And Reranking For Subtask 1

Labels: `week-5`, `P0`, `retrieval`, `evaluation`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M2` / `M1`

Branch: `experiment/w5-04-retrieval-reranking`

PR title: `experiment: improve VLSP subtask1 retrieval with fusion and reranking`

Depends on: `W2-02`, `W3-05`, `W5-02`

### Description

Improve Task 1 retrieval because it directly determines Subtask 1 score and
also affects Subtask 2 QA quality. This issue should compare retrieval
strategies on the locked validation split before producing public/private test
submissions.

### Project Files To Change

- `src/retrieval.py`
- `src/evaluate.py` if retrieval diagnostics need more fields
- `configs/experiments/retrieval_bm25.yaml` (new if implemented)
- `configs/experiments/retrieval_fusion_rerank.yaml` (new)
- `docs/experiments.md`
- `docs/error-analysis.md`
- `tests/test_retrieval.py`
- `tests/test_evidence_fusion.py`
- `tests/test_retrieval_rerank.py` (new if separate)

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  modality/top-k ablation patterns.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/text_embedding.py`: text
  embedding wrapper idea.
- `../LexiSignVQA-main/src/eval/sub_task_1.py`: retrieval F2 scoring.
- `docs/error-analysis.md`: hard cases and error categories.
- `src/retrieval.py`: current direct and example-fusion retrieval.

### Implementation Notes

Start with low-risk retrieval improvements before adding detector/OCR work:

- Compare direct dense retrieval vs example-fused retrieval.
- Add or simulate BM25/keyword retrieval over article title + content.
- Fuse dense and keyword scores with configurable weights.
- Add query expansion:
  - question only;
  - question + choices;
  - question + normalized sign-code hints;
  - optional image caption/sign description when a real VLM captioner is
    available.
- Rerank top-20 candidates to top-k using article title match, sign-code match,
  and citation votes.
- Tune `top_k` values: `3`, `5`, `8`, `10`.
- Do not use public/private test feedback for tuning unless post-submission
  rules explicitly allow repeated attempts.

### Expected Output

An ablation table in `docs/experiments.md`:

| Config | Top-k | Dense | BM25 | Example fusion | Rerank | Retrieval P/R/F2 |
| --- | ---: | --- | --- | --- | --- | --- |

The best validation config becomes the submission retrieval config for Task 1.

### Acceptance Criteria

- [ ] At least three retrieval configs are compared on the same locked split.
- [ ] Best config is chosen by validation Retrieval F2, not intuition.
- [ ] Retrieval results are deterministic for the same seed/config.
- [ ] Reranker never uses gold answer or gold citations for test samples.
- [ ] Failure cases are updated with representative examples.
- [ ] Public/private submission uses the frozen best config.

### Tests

- Unit test BM25/keyword scoring with synthetic articles if implemented.
- Unit test score fusion ordering.
- Unit test reranker respects top-k and unique article UIDs.
- Unit test sign-code query hints do not introduce malformed citations.
- Unit test retrieval config loading.

### Verification

```bash
python -m pytest tests/test_retrieval.py tests/test_evidence_fusion.py tests/test_retrieval_rerank.py -q
make qdrant-up
make preprocess
make index
python -m src.pipeline \
  --mode benchmark \
  --config configs/experiments/retrieval_fusion_rerank.yaml \
  --limit 20
python -m src.evaluate \
  --config configs/experiments/retrieval_fusion_rerank.yaml \
  --predictions data/outputs/experiments/retrieval_fusion_rerank.jsonl
git diff --check
```

## W5-05 - Prompt And Parser Optimization For Subtask 2

Labels: `week-5`, `P1`, `vlm`, `evaluation`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M3` / `M4`

Branch: `experiment/w5-05-subtask2-prompt-optimization`

PR title: `experiment: improve constrained QA prompting and parsing`

Depends on: `W5-03`, `W5-04`

### Description

Improve Subtask 2 QA accuracy by reducing invalid outputs and making the model
compare answer choices more reliably. This issue should not change the
submission format. It should produce a controlled prompt/parser ablation.

### Project Files To Change

- `src/prompts.py`
- `src/vlm.py`
- `src/pipeline.py` if retry behavior belongs there
- `configs/experiments/subtask2_prompt_ablation/` (new)
- `docs/experiments.md`
- `tests/test_vlm_output.py`
- `tests/test_citation_output.py`
- `tests/test_prompt_ablation.py` (new if useful)

### Reference Files To Study

- `../LexiSignVQA-main/src/prompts/answer_prompt.py`: final-answer constraint
  style.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: answer extraction and
  question-type branching.
- `src/prompts.py`: current structured legal RAG prompt.
- `src/vlm.py`: parser and citation validation.
- `docs/error-analysis.md`: reasoning/output-format failures.

### Implementation Notes

- Keep JSON output fields unchanged:
  `answer`, `citations`, `explanation`, `confidence`, `abstained`.
- Add a stricter Subtask 2 prompt variant such as `vlsp_subtask2_rag`.
- For multiple choice:
  - require a one-line comparison of choices in the short explanation;
  - final `answer` must be one of `A/B/C/D`.
- For Yes/No:
  - require a direct legal conclusion;
  - final `answer` must be `Đúng` or `Sai`.
- Add one retry on invalid JSON or invalid label:
  - retry prompt should include the raw invalid output and ask only for valid
    JSON;
  - do not change evidence or sample order during retry.
- Do not store hidden chain-of-thought.
- Do not map arbitrary invalid text to a label unless it exactly matches a
  deterministic normalization rule.

### Expected Output

A prompt ablation table:

| Config | Model | Retrieval config | Prompt | Parse success | QA accuracy | Invalid |
| --- | --- | --- | --- | ---: | ---: | ---: |

### Acceptance Criteria

- [ ] New prompt variant is selectable from config.
- [ ] Parser still rejects hallucinated citations outside retrieved evidence.
- [ ] Invalid JSON retry is implemented or explicitly documented as skipped.
- [ ] Prompt ablation uses the same locked validation split.
- [ ] Winning prompt is chosen by validation QA accuracy and parse success.
- [ ] No private-test output is used to tune prompt wording.

### Tests

- Unit test multiple-choice prompt includes strict answer constraints.
- Unit test Yes/No prompt includes strict answer constraints.
- Unit test retry prompt keeps evidence unchanged.
- Unit test invalid answer `E` remains invalid after parsing.
- Unit test valid markdown-fenced JSON still parses.

### Verification

```bash
python -m pytest tests/test_vlm_output.py tests/test_citation_output.py tests/test_prompt_ablation.py -q
python -m src.pipeline \
  --mode benchmark \
  --config configs/experiments/subtask2_prompt_ablation/vlsp_subtask2_rag.yaml \
  --limit 10
python -m src.evaluate \
  --config configs/experiments/subtask2_prompt_ablation/vlsp_subtask2_rag.yaml \
  --predictions data/outputs/experiments/vlsp_subtask2_rag.jsonl
git diff --check
```

## W5-06 - Free-Form Upload Legal QA Demo

Labels: `week-5`, `P1`, `demo`, `vlm`, `retrieval`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M2` / `M3`

Branch: `feat/w5-06-free-form-upload-demo`

PR title: `feat(demo): add free-form upload legal QA mode`

Depends on: `W4-04`, `W5-03`

### Description

Add the product-facing free-form mode: a user uploads a traffic image, types a
natural-language legal question, and receives a short answer with citations,
evidence, confidence, abstention, and disclaimer.

This is separate from VLSP Subtask 2. Subtask 2 remains constrained
multiple-choice/Yes-No; free-form QA is a demo/product capability.

### Project Files To Change

- `app/streamlit_app.py`
- `src/pipeline.py`
- `src/schemas.py` only if uploaded-image metadata needs a small extension
- `src/prompts.py`
- `README.md`
- `docs/model-card.md`
- `tests/test_demo_contract.py`
- `tests/test_free_form_demo.py` (new)

### Reference Files To Study

- `app/streamlit_app.py`: current sample-based demo modes.
- `src/schemas.py`: `QuestionType.FREE_FORM` and validation behavior.
- `src/prompts.py`: free-form answer instruction.
- `src/pipeline.py`: demo-facing contract and retrieval-only fallback.
- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: evidence display pattern.

### Implementation Notes

- Add a Streamlit tab: `Free-form upload`.
- Inputs:
  - uploaded image file;
  - free-form question text;
  - top-k evidence setting;
  - output mode: retrieval-only, live VLM, cached if later supported.
- Store uploaded images under an ignored temp/output directory, for example
  `data/outputs/uploads/`.
- Build `Query(question_type="Free form", choices=None, answer=None)`.
- Retrieve LawDB evidence from the free-form question.
- If live backend is configured, call VLM with `include_image=true`.
- If live backend is missing, show retrieval evidence and a clear message.
- Always show:
  - answer;
  - citations;
  - retrieved evidence;
  - confidence;
  - abstention;
  - disclaimer that it is research assistance, not official legal advice.
- Do not expose `.env`, API keys, raw local absolute paths, or hidden
  chain-of-thought.

### Expected Output

`bash scripts/demo.sh` opens a demo with:

- sample inspection mode;
- cached prediction mode;
- live benchmark sample mode;
- free-form upload QA mode.

### Acceptance Criteria

- [ ] User can upload an image and enter a natural-language question.
- [ ] Free-form query retrieves legal evidence.
- [ ] Live VLM mode returns a paragraph answer when configured.
- [ ] Missing live backend falls back to retrieval-only mode without crashing.
- [ ] Citations and evidence are visible.
- [ ] Uploaded files are stored only in ignored output/temp paths.
- [ ] Tests do not require real model calls.
- [ ] Demo still starts without credentials.

### Tests

- Unit test free-form demo request contract.
- Unit test uploaded image path is stored under an ignored output directory.
- Unit test free-form query has no choices and no gold answer.
- Unit test missing live backend returns retrieval-only response.
- Unit test fake VLM free-form answer parses and displays citations.

### Verification

```bash
python -m pytest tests/test_free_form_demo.py tests/test_demo_contract.py tests/test_vlm_output.py -q
make qdrant-up
make preprocess
make index
bash scripts/demo.sh
git status --ignored --short data/outputs/uploads
git diff --check
```

## W5-07 - Post-Submission Result Report And Defense Update

Labels: `week-5`, `P0`, `documentation`, `evaluation`

Milestone: `W5 - Post-submission benchmark`

Owner/reviewer: `M4` / `M3`

Branch: `docs/w5-07-post-submission-report`

PR title: `docs: report VLSP post-submission benchmark results`

Depends on: `W5-01`, `W5-02`, `W5-03`, `W5-04`, `W5-05`, `W5-06`

### Description

Update the final report and slides with post-submission benchmark evidence.
The report must clearly separate official/post-submission results from internal
validation and free-form demo results.

### Project Files To Change

- `docs/report.md`
- `docs/experiments.md`
- `docs/final-slides.md`
- `docs/model-card.md`
- `docs/error-analysis.md`
- `docs/assets/` for Codabench screenshots
- `README.md`
- release notes if the team creates a new tag

### Reference Files To Study

- `docs/report.md`: current final course report.
- `docs/experiments.md`: metric-source rules.
- `docs/release-v0.1.0-course.md`: release checklist.
- Codabench result page/screenshot exported by the team.
- Generated `submission.zip` and validation summaries.

### Implementation Notes

- Add a section: `VLSP Post-Submission Results`.
- Include:
  - submission timestamp;
  - Codabench phase name;
  - model/backend;
  - retrieval config;
  - prompt config;
  - Task 1 score;
  - Task 2 score;
  - screenshot/path to result artifact.
- If the run is post-submission and not official competition ranking, say so.
- Do not compare against private leaderboard unless Codabench explicitly shows
  comparable ranking.
- Keep internal validation table separate.
- Keep free-form QA demo as qualitative case study.
- Add at least 5 free-form demo examples with citations and screenshots if
  possible.

### Expected Output

The final report can say:

```text
We evaluated the system using both an internal leakage-safe validation split
and the VLSP 2025 MLQA-TSR post-submission system. The post-submission result
is reported separately from internal validation and free-form QA demo cases.
```

### Acceptance Criteria

- [ ] Report includes post-submission Task 1 and Task 2 results if available.
- [ ] Report clearly labels post-submission vs official ranking vs internal
  validation.
- [ ] Every number links to a metrics JSON, Codabench screenshot, or saved
  result artifact.
- [ ] Free-form QA examples are not reported as official accuracy.
- [ ] Slides match the report numbers.
- [ ] Known limitations and next steps are updated.

### Tests

- Manual review that every metric has a source.
- Manual review that no mock metrics are presented as real model quality.
- Manual review that QLoRA is not presented as final unless proven.
- Manual check that screenshots do not expose secrets or private data beyond
  intended benchmark result metadata.

### Verification

```bash
python -m src.competition_submission --help
python -m src.evaluate --help
make verify
git diff --check
```
