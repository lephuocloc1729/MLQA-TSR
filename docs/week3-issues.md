# Week 3 Issues - QLoRA And Controlled Experiments

This issue pack continues from the week-1 and week-2 baseline:

```text
flattened LawDB + grouped split
  -> Qdrant LawDB retrieval
  -> fused evidence + retrieved examples
  -> B2/B3/B4 smoke ablations
  -> real VLM backend, SFT data, QLoRA smoke, and controlled W3 experiments
```

Week 3 should turn the week-2 engineering smoke runs into controlled model
experiments. The main technical milestone is a small, reproducible PEFT/QLoRA
experiment and a fair comparison against the base VLM using the same locked
validation split. Do not add OCR, traffic-sign detection, or detector-heavy
cropped-sign retrieval in week 3 unless all required W3 gates are already
green.

Replace `M1`-`M4` with real member names before creating GitHub issues.

## Entry Gate From Week 1 And Week 2

Before merging week-3 experiment PRs, the team should confirm:

- LawDB preprocessing still produces 402 article records.
- The grouped split remains locked with 421 train samples and 109 validation
  samples unless the split is intentionally regenerated and documented.
- B2/B3/B4 configs exist and use the same validation split.
- Week-2 smoke metrics are clearly marked `mock: true` and are not reported as
  final model accuracy.
- The retrieval layer can return direct LawDB evidence, fused evidence, and
  retrieved training examples.
- The VLM parser still rejects invalid answers and citations outside retrieved
  evidence.

If GPU access is unavailable, do not fake QLoRA results. Merge CPU-safe data,
collator, config, and trainer tests, then mark the GPU smoke run as blocked
with owner and expected resolution date.

## W3-01 - Build Leakage-Safe SFT Data And Collator

Labels: `week-3`, `P0`, `data`, `training`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M1` / `M4`

Branch: `feat/w3-01-sft-data-collator`

PR title: `feat(training): build leakage-safe SFT data and collator`

Depends on: `W1-03`, `W1-06`, `W2-03`, `W2-04`

### Description

Create supervised fine-tuning records for the multimodal legal QA task. The SFT
data must follow the same prompt and output contract used by the week-2
pipeline, while preventing validation leakage.

### Project Files To Change

- `src/data_utils.py`
- `src/collator.py`
- `src/prompts.py` only if a training-specific prompt helper is needed
- `configs/qlora.yaml`
- `tests/test_sft_data.py` (new)
- `tests/test_collator.py` (new)
- Generated locally: `data/processed/sft_train.jsonl`
- Generated locally: `data/processed/sft_val.jsonl`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: learn how
  image paths, answers, and relevant articles are attached to samples.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn answer normalization and
  question-type branching.
- `../LexiSignVQA-main/src/prompts/answer_prompt.py`: learn output
  constraints, but keep our stricter JSON target format.

### Implementation Notes

- Build SFT records only from the train split by default.
- Validation records may be generated for evaluation/loss only; never mix them
  into training.
- Each record should contain image path, question, choices when available,
  retrieved or oracle legal evidence, target answer, target citations, and a
  short target explanation.
- Prefer oracle gold `relevant_articles` for SFT targets when available, then
  resolve them to LawDB content.
- The target assistant response must be valid JSON compatible with
  `Prediction`: `answer`, `citations`, `explanation`, `confidence`,
  `abstained`.
- The collator should be processor-aware but testable with a fake processor.
- Do not commit generated large tensors or tokenized caches.

### Expected Output

Example SFT JSONL row:

```json
{
  "id": "train_1",
  "image_id": "train_1_3",
  "image_path": "data/raw/train_data/train_images/train_1_3.jpg",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "{\"answer\":\"B\",...}"}
  ],
  "target": {
    "answer": "B",
    "citations": [{"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}]
  }
}
```

### Acceptance Criteria

- [ ] `sft_train.jsonl` is built only from train split samples.
- [ ] `sft_val.jsonl` is built only from validation split samples.
- [ ] No train/validation `image_id` overlap is introduced.
- [ ] Every target assistant message parses into `Prediction`.
- [ ] Every target citation resolves to LawDB content.
- [ ] Collator masks padding and non-target tokens correctly.
- [ ] Unit tests run without downloading a real VLM processor.

### Tests

- Unit test SFT record creation for Multiple choice.
- Unit test SFT record creation for Yes/No.
- Unit test target JSON parses with `parse_prediction`.
- Unit test split leakage guard by `image_id`.
- Unit test fake processor/collator output shapes and label masking.

### Verification

```bash
python -m src.data_utils --mode build-sft
python -m pytest tests/test_sft_data.py tests/test_collator.py tests/test_schemas.py -q
python - <<'PY'
from pathlib import Path
for path in ["data/processed/sft_train.jsonl", "data/processed/sft_val.jsonl"]:
    rows = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    print(path, len(rows))
PY
```

## W3-02 - Add QLoRA Trainer And GPU Smoke Run

Labels: `week-3`, `P0`, `training`, `needs-gpu`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M3` / `M2`

Branch: `feat/w3-02-qlora-trainer`

PR title: `feat(training): add reproducible QLoRA trainer`

Depends on: `W3-01`

### Description

Implement a small QLoRA training path for the selected open-source VLM. The
goal is not a large final model; the goal is a reproducible PEFT smoke run that
can be compared fairly with the base VLM later.

### Project Files To Change

- `src/train_qlora.py`
- `configs/qlora.yaml`
- `requirements.txt` only if a missing direct dependency is truly required
- `tests/test_train_config.py` (new)
- Optional generated locally: `checkpoints/qlora_adapter/`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: reuse only
  the data-access idea, not the model serving setup.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn the benchmark input and
  answer normalization shape.
- `../LexiSignVQA-main/pyproject.toml`: learn dependency pinning style.
- Use official Transformers/PEFT examples for the actual QLoRA mechanics; the
  two local projects do not implement QLoRA.

### Implementation Notes

- Load the base model from `configs/qlora.yaml`.
- Use 4-bit loading only on Linux CUDA environments where `bitsandbytes` is
  available.
- Freeze the vision encoder unless the team has enough compute to justify
  otherwise.
- Log trainable parameter count, total parameter count, device, dtype, seed,
  batch size, gradient accumulation, max samples, and estimated VRAM.
- The smoke run should support `max_samples: 20` and one tiny overfit pass.
- Save adapter files outside Git under `checkpoints/qlora_adapter/`.
- Save a small metadata JSON with base model, dataset hash, split hash, LoRA
  hyperparameters, and commit hash when available.
- CPU CI should validate config and argument parsing only; it must not load the
  real model.

### Expected Output

- `python -m src.train_qlora --config configs/qlora.yaml --dry-run` validates
  config and prints the training plan.
- A GPU command can run a 20-sample QLoRA smoke training job.
- Adapter output is stored locally and ignored by Git.

### Acceptance Criteria

- [ ] Config validation catches missing SFT files and invalid LoRA settings.
- [ ] Dry-run works on CPU without loading model weights.
- [ ] GPU smoke run completes on a small sample when CUDA is available.
- [ ] Trainable parameter count is logged.
- [ ] Checkpoint resume path is supported or explicitly rejected with a helpful
  error.
- [ ] Checkpoint metadata is saved.
- [ ] No model weights, adapters, or training caches are committed.

### Tests

- Unit test QLoRA config loading.
- Unit test dry-run output with fake filesystem paths.
- Unit test invalid LoRA rank/alpha fails clearly.
- Unit test checkpoint metadata writer.
- Manual GPU smoke run evidence attached to the PR when available.

### Verification

```bash
python -m pytest tests/test_train_config.py -q
python -m src.train_qlora --config configs/qlora.yaml --dry-run

# GPU-only smoke, run only on a CUDA machine:
python -m src.train_qlora --config configs/qlora.yaml --max-samples 20
```

## W3-03 - Connect Real VLM Backend For Non-Mock Benchmarks

Labels: `week-3`, `P0`, `vlm`, `evaluation`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M3` / `M2`

Branch: `feat/w3-03-real-vlm-backend`

PR title: `feat(vlm): connect real model backend for benchmarks`

Depends on: `W1-06`, `W2-05`

### Description

Week 2 proved the pipeline with deterministic mock predictions. Week 3 needs a
real VLM backend so B2/B3/B4 can be evaluated as model runs rather than smoke
tests. This issue connects a configurable backend while keeping tests mockable.

### Project Files To Change

- `src/vlm.py`
- `src/pipeline.py`
- `configs/config.yaml`
- `.env.example`
- `tests/test_vlm_backend.py` (new)
- `tests/test_pipeline.py` if benchmark runtime behavior changes

### Reference Files To Study

- `../LexiSignVQA-main/src/deps/llm_client.py`: learn the client wrapper
  boundary and API-key handling.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn image loading and
  generation orchestration.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`: learn
  multimodal message construction and response handling.

### Implementation Notes

- Keep `VLMClient` injectable so tests can use a fake client.
- Support at least one real backend mode, for example local Transformers or an
  OpenAI-compatible endpoint.
- Keep credentials in environment variables and `.env.example`; never commit
  secrets.
- Preserve `temperature: 0.0` for benchmark runs.
- Save raw model response in `Prediction.raw_response` for debugging, but do
  not store hidden chain-of-thought.
- Real runs must write `experiment.mock: false` in metrics artifacts.
- Failure to call the model should produce an invalid/error record, not a fake
  default answer.

### Expected Output

- `LegalQAVLM` can be configured with a real backend or fake test client.
- The benchmark command can run with `experiment.mock: false` when credentials
  and compute are available.
- Metrics artifacts clearly separate mock and real model runs.

### Acceptance Criteria

- [ ] Real backend configuration is documented.
- [ ] Fake client tests still pass without network, GPU, or model downloads.
- [ ] Missing credentials produce a helpful error.
- [ ] Real runs mark `mock=false`.
- [ ] Model errors are captured as invalid/error samples.
- [ ] Existing parser validation remains enforced.

### Tests

- Unit test fake client success.
- Unit test fake client invalid JSON.
- Unit test missing credential/config error.
- Unit test metrics artifact marks mock vs non-mock correctly.
- Manual one-sample real VLM smoke run if compute/API is available.

### Verification

```bash
python -m pytest tests/test_vlm_backend.py tests/test_vlm_output.py tests/test_pipeline.py -q
python -m src.pipeline --mode benchmark --config configs/experiments/w3_b2_text_rag_real.yaml --limit 1
python -m src.evaluate --predictions data/outputs/experiments/w3_b2_text_rag_real.jsonl
```

## W3-04 - Structured Legal Reasoning Prompt Variant

Labels: `week-3`, `P1`, `vlm`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M3` / `M2`

Branch: `feat/w3-04-structured-legal-reasoning`

PR title: `feat(vlm): add structured legal reasoning prompt variant`

Depends on: `W2-04`, `W3-03`

### Description

Add a controlled legal-reasoning prompt variant that improves explanation
quality without storing unrestricted chain-of-thought. The output must remain
machine-parseable and compatible with `Prediction`.

### Project Files To Change

- `src/prompts.py`
- `src/vlm.py`
- `configs/experiments/` (new W3 prompt configs)
- `tests/test_citation_output.py` (new)
- `tests/test_few_shot_prompt.py` if prompt variants share helpers

### Reference Files To Study

- `../LexiSignVQA-main/src/prompts/sign_filter_prompt.py`: learn staged,
  constrained prompt wording.
- `../LexiSignVQA-main/src/prompts/answer_prompt.py`: learn final-answer
  constraints.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn final answer extraction
  and normalization.

### Implementation Notes

- Add a prompt variant such as `structured_legal_rag`.
- Ask the model to structure the short explanation as:
  `Observation`, `Legal basis`, and `Conclusion`.
- Keep the JSON fields exactly compatible with `Prediction`: `answer`,
  `citations`, `explanation`, `confidence`, `abstained`.
- The `explanation` may contain the three short labeled parts, but must not
  expose hidden chain-of-thought.
- Citations must still be restricted to retrieved `Evidence`.
- The prompt should explicitly say when to abstain if evidence is insufficient.

### Expected Output

Example valid explanation:

```text
Observation: The image/question refers to a time-limited prohibition sign.
Legal basis: Article 22 describes additional panels and time validity.
Conclusion: Option B matches the allowed interpretation outside the listed time.
```

### Acceptance Criteria

- [ ] New prompt variant is selectable from config.
- [ ] Parser still requires the same JSON fields.
- [ ] Explanation is concise and does not include hidden chain-of-thought.
- [ ] Hallucinated citations outside retrieved evidence are rejected.
- [ ] Multiple choice and Yes/No examples are covered.
- [ ] Tests do not require real model calls.

### Tests

- Unit test prompt includes Observation/Legal basis/Conclusion instructions.
- Unit test valid structured explanation parses into `Prediction`.
- Unit test citation outside evidence fails.
- Unit test abstention response remains valid.
- Unit test prompt variant config normalization.

### Verification

```bash
python -m pytest tests/test_citation_output.py tests/test_vlm_output.py tests/test_schemas.py -q
python -m src.vlm --mode build-prompt --variant structured_legal_rag --sample-id train_1
```

## W3-05 - Freeze Retrieval Config And Analyze Hard Cases

Labels: `week-3`, `P1`, `retrieval`, `evaluation`, `documentation`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M2` / `M1`

Branch: `experiment/w3-05-freeze-retrieval`

PR title: `experiment: freeze retrieval config and analyze hard cases`

Depends on: `W2-05`

### Description

Choose the retrieval configuration that will be used for week-3 and week-4
model comparisons. Analyze retrieval failures so the report can explain why
the system succeeds or fails.

### Project Files To Change

- `src/retrieval.py` only if final config support needs a small change
- `configs/experiments/retrieval_final.yaml` (new)
- `docs/experiments.md`
- `docs/error-analysis.md` (new or updated)
- `tests/test_retrieval.py` if final config behavior changes

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  learn top-k and modality comparison patterns.
- `../LexiSignVQA-main/README.md`: learn how retrieval results and ablations
  are summarized.
- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: learn how to inspect
  sample-level retrieval failures.

### Implementation Notes

- Choose final top-k, text/image weights, and evidence fusion settings using
  validation results only.
- Analyze at least 15 retrieval failures or hard cases.
- Classify errors into categories such as visual ambiguity, weak question text,
  missing law context, similar article confusion, annotation mismatch, and
  output-format issue.
- Keep OCR and cropped-sign detection as documented stretch work unless there
  is measured evidence that it improves the locked validation split.
- Record representative sample IDs and retrieved/gold citations.

### Expected Output

- `configs/experiments/retrieval_final.yaml` documents the chosen retrieval
  configuration.
- `docs/error-analysis.md` includes at least 15 hard cases with categories.
- `docs/experiments.md` explains why the final retrieval config was chosen.

### Acceptance Criteria

- [ ] Final retrieval config is explicit and versioned.
- [ ] The chosen config uses the locked validation split.
- [ ] At least 15 hard cases are analyzed.
- [ ] Error categories are consistent and useful for the final report.
- [ ] The doc distinguishes retrieval failures from VLM reasoning failures.
- [ ] No new detector/OCR requirement is introduced without evidence.

### Tests

- Unit test final config can be loaded.
- Unit test final config produces deterministic retrieval with fake stores.
- Manual review of 15 hard cases.
- Manual check that docs do not report mock QA accuracy as real model quality.

### Verification

```bash
python -m pytest tests/test_retrieval.py tests/test_experiment_config.py -q
python -m src.pipeline --mode benchmark --config configs/experiments/retrieval_final.yaml --limit 10
python -m src.evaluate --config configs/experiments/retrieval_final.yaml --predictions data/outputs/experiments/retrieval_final.jsonl
```

## W3-06 - Run Controlled Real Baselines And QLoRA Diagnostic Smoke

Labels: `week-3`, `P0`, `evaluation`, `training`, `vlm`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M4` / `M3`

Branch: `experiment/w3-06-real-baselines-qlora-diagnostic`

PR title: `experiment: evaluate real baselines and QLoRA diagnostic smoke`

Depends on: `W3-02`, `W3-03`, `W3-04`, `W3-05`

### Description

Run controlled week-3 model checks without overstating the QLoRA result. The
main goal is to replace week-2 mock metrics with real/base VLM baselines when
a backend is available, then document QLoRA as a diagnostic PEFT smoke
experiment. The current 80-sample adapter is useful evidence that training
works, but it is not strong enough to become the official submission model.

### Project Files To Change

- `configs/experiments/` (new W3 configs)
- `src/pipeline.py`
- `src/evaluate.py`
- optional `src/adapter_infer.py` if adapter smoke inference needs a separate entrypoint
- `scripts/evaluate.sh`
- `docs/experiments.md`
- `tests/test_experiment_config.py`
- `tests/test_evaluate.py` if artifact fields are extended

### Reference Files To Study

- `../LexiSignVQA-main/src/eval/sub_task_1.py`: retrieval P/R/F2.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: QA accuracy and breakdowns.
- `../LexiSignVQA-main/README.md`: experiment table style.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`:
  retrieved-example few-shot setup.

### Implementation Notes

- P0 real baseline configs:
  - `W3_B2_text_rag_real`: base VLM + direct LawDB retrieval.
  - `W3_B5_structured_real`: base VLM + frozen/fused retrieval + structured
    legal reasoning prompt.
- P0 artifact fields:
  - model/backend name;
  - prompt variant;
  - retrieval config;
  - seed and split hash;
  - `max_new_tokens`;
  - parse success count;
  - invalid JSON count;
  - truncated output count;
  - latency;
  - retrieval metrics and QA metrics.
- QLoRA is diagnostic unless adapter inference is fully integrated into the
  benchmark pipeline. The GPU evidence to document is:
  - 20-sample smoke run succeeded;
  - 80-sample QLoRA run succeeded;
  - 300-sample run failed with CUDA OOM on RTX 3090 24GB;
  - validation smoke with `max_new_tokens=160` had many truncated outputs;
  - validation smoke with `max_new_tokens=320` reached `1/3` exact match.
- Report QLoRA checkpoint metadata, not leaderboard-quality validation claims.
- Real/base VLM runs must be marked `mock=false`.
- Failures must not silently become `A` or `Đúng`.
- No checkpoint, adapter, model weight, or bulky GPU output may be committed.

### Expected Output

- Real/base VLM prediction JSONL and metrics JSON if the backend is available.
- QLoRA diagnostic smoke notes with adapter metadata and validation-smoke
  limitations.
- `docs/experiments.md` clearly separates:
  - week-2 mock smoke metrics;
  - real/base VLM metrics;
  - structured prompt metrics;
  - QLoRA diagnostic smoke evidence.

### Acceptance Criteria

- [ ] All W3 configs use the same locked validation split unless explicitly
  marked non-comparable.
- [ ] Real/base runs clearly mark `mock=false`.
- [ ] Metrics artifacts include backend/model, prompt variant, retrieval
  strategy, `max_new_tokens`, parse success, invalid JSON, truncated output,
  latency, retrieval metrics, and QA metrics.
- [ ] QLoRA diagnostic smoke is not reported as final validation accuracy.
- [ ] The 80-sample checkpoint metadata is linked or summarized.
- [ ] The 300-sample CUDA OOM limitation is documented.
- [ ] The validation smoke result is documented, including `1/3` exact match
  with `max_new_tokens=320`.
- [ ] No checkpoint/model artifacts are committed.

### Tests

- Unit test W3 experiment config loading.
- Unit test artifact includes model/backend, `max_new_tokens`, parse status,
  and adapter metadata when present.
- Unit test split mismatch is rejected.
- Unit test invalid/truncated predictions are counted separately and counted
  as incorrect.
- Manual review of metrics table before reporting numbers.

### Verification

```bash
python -m pytest tests/test_experiment_config.py tests/test_evaluate.py tests/test_pipeline.py -q
python -m src.pipeline --mode benchmark --config configs/experiments/w3_b2_text_rag_real.yaml --limit 5
python -m src.evaluate --config configs/experiments/w3_b2_text_rag_real.yaml --predictions data/outputs/experiments/w3_b2_text_rag_real.jsonl

# Adapter diagnostic only, if adapter inference is wired:
python -m src.adapter_infer --adapter checkpoints/qlora_adapter --split val --limit 3 --max-new-tokens 320
```

## W3-07 - Week 3 Integration, Report, And QLoRA Checkpoint Card

Labels: `week-3`, `P0`, `documentation`, `evaluation`

Milestone: `W3 - QLoRA and controlled experiments`

Owner/reviewer: `M4` / `M3`

Branch: `release/w3-qlora-diagnostic-report`

PR title: `release: integrate week 3 report and QLoRA checkpoint card`

Depends on: `W3-01`, `W3-02`, `W3-03`, `W3-04`, `W3-05`, `W3-06`

### Description

Integrate the week-3 training and experiment work into one reportable release.
The release must be honest about the GPU result: QLoRA training is real and
reproducible, but the current 80-sample adapter is diagnostic and not
submission-ready. The main week-4 product direction remains retrieval,
structured prompting, evaluation, demo, and submission/export.

### Project Files To Change

- `Makefile`
- `README.md`
- `docs/report.md`
- `docs/experiments.md`
- `docs/checkpoint-card.md` (new, or `docs/model-card.md`)
- `configs/qlora.yaml`
- Release notes if the team keeps them

### Reference Files To Study

- `../LexiSignVQA-main/README.md`: concise experiment reporting and pipeline
  explanation.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/README.md`: low-cost baseline
  framing.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: QA metric breakdown wording.

### Implementation Notes

- Separate mock smoke metrics, real base VLM metrics, and QLoRA metrics.
- The checkpoint card should include:
  - base model: `Qwen/Qwen2.5-VL-3B-Instruct`;
  - adapter path: `checkpoints/qlora_adapter`;
  - effective train count: `80`;
  - GPU: RTX 3090 24GB;
  - dtype: `bfloat16`;
  - LoRA rank/alpha/dropout;
  - trainable parameter count;
  - dataset hash and split hash;
  - training commands;
  - validation smoke commands;
  - known limitations.
- Known limitations must include:
  - 300-sample run failed with CUDA OOM;
  - current adapter is weak on validation smoke;
  - low `max_new_tokens` causes JSON truncation;
  - current adapter should not be used as the submission-quality model.
- Store checkpoints outside Git. Include checksum/access instructions only if
  a checkpoint exists.
- Update the four-month continuation plan based on actual week-3 findings.

### Expected Output

- Week-3 report section with commands, metrics, blockers, errors, and member
  contributions.
- QLoRA checkpoint card with GPU environment and limitations.
- README/Makefile commands for W3 training/evaluation.
- Week-4 plan that prioritizes final system quality over bigger training.

### Acceptance Criteria

- [ ] Report includes exact commands and environment.
- [ ] Report includes retrieval P/R/F2 and QA Accuracy for real runs that were
  actually executed.
- [ ] Mock results are not presented as final model quality.
- [ ] QLoRA checkpoint metadata is included.
- [ ] 300-sample OOM and validation-smoke weakness are documented.
- [ ] The report explicitly recommends not using the current adapter as the
  final submission model.
- [ ] Large artifacts and model weights remain outside Git.
- [ ] Next-week plan prioritizes real baseline validation, error analysis,
  submission/export, demo polish, and final defense assets.

### Tests

- Run the project verification command.
- Manual check that docs reference only existing artifacts.
- Manual check that every reported number can be traced to a metrics JSON.
- Manual check that checkpoint claims can be traced to `adapter_metadata.json`
  or trainer logs.
- Manual fresh-clone command review for README/Makefile instructions.

### Verification

```bash
make verify
python -m src.train_qlora --config configs/qlora.yaml --dry-run
python -m src.evaluate --help
git diff --check
```

## Suggested Parallel Work Order

```text
W3-01 -> W3-02 --------\
W3-03 -> W3-04 ---------> W3-06 -> W3-07
W3-05 ------------------/
```

- `M1` owns SFT data and collator: W3-01.
- `M3` owns QLoRA training and VLM behavior: W3-02, W3-03, W3-04.
- `M2` owns retrieval freeze and hard-case analysis: W3-05.
- `M4` owns experiment matrix, integration, and weekly report: W3-06, W3-07.

W3-01 and W3-03 can start in parallel. W3-02 should not run real training until
W3-01 is merged. W3-06 should not publish QLoRA numbers until W3-02 provides a
real adapter and metadata. W3-07 should not merge while reported metrics are
missing source artifacts.

## Week 3 Definition Of Done

- SFT train/validation records are built without split leakage.
- Collator and training config are testable without loading the real model.
- A real VLM backend can run at least one non-mock benchmark sample, or a clear
  blocker is documented.
- QLoRA has a dry-run and, if GPU is available, a 20-sample smoke checkpoint.
- Structured legal reasoning prompt remains citation-grounded and
  machine-parseable.
- Final retrieval configuration and at least 15 hard cases are documented.
- W3 experiment table separates mock, real base VLM, oracle-evidence, and
  QLoRA results.
