# Week 4 Issues - Final Product, Validation, Demo, And Defense Package

This issue pack continues from the week-1, week-2, and week-3 work:

```text
validated LawDB + grouped split
  -> frozen retrieval config
  -> structured prompting and real/model smoke checks
  -> QLoRA diagnostic adapter evidence
  -> final validation, submission/export, demo, report, and release rehearsal
```

Week 4 should not chase bigger QLoRA training first. The main course product
should be a retrieval-grounded legal QA system with structured outputs,
citations, evaluation artifacts, and a reliable demo. The QLoRA adapter should
be reported as an experimental extension unless adapter inference becomes
stable and its validation metrics are clearly better.

Replace `M1`-`M4` with real member names before creating GitHub issues.

## Entry Gate From Week 3

Before merging final week-4 PRs, the team should confirm:

- The frozen retrieval config exists and is documented.
- Week-2 mock metrics are not reported as final VLM quality.
- Week-3 QLoRA metadata exists, but the adapter is treated as diagnostic.
- The 300-sample QLoRA OOM and validation-smoke weakness are documented.
- The real/base VLM backend path is either working or explicitly marked
  unavailable with a fallback plan.
- Large artifacts under `checkpoints/`, `data/outputs/`, and Qdrant storage are
  not committed.

If real model inference is unavailable, the final demo/report must still work
in retrieval-only or cached-prediction mode, with the limitation stated
clearly.

## W4-01 - Adapter Batch Inference Pipeline

Labels: `week-4`, `P1`, `training`, `vlm`, `evaluation`, `needs-gpu`

Milestone: `W4 - Demo and report`

Owner/reviewer: `M3` / `M2`

Branch: `feat/w4-01-adapter-batch-inference`

PR title: `feat(vlm): add QLoRA adapter batch inference diagnostics`

Depends on: `W3-02`, `W3-07`

### Description

Add a diagnostic batch inference path for the local QLoRA adapter. This is not
the main final product path. It exists to evaluate whether the 80-sample
adapter can produce valid JSON predictions over the locked validation split and
to measure parse/truncation failures honestly.

### Project Files To Change

- `src/adapter_infer.py` (new) or `src/pipeline.py` if the team prefers one
  runner
- `src/vlm.py` if adapter output parsing needs a small reusable helper
- `configs/experiments/` (adapter diagnostic configs)
- `scripts/evaluate.sh`
- `tests/test_adapter_infer.py` (new)
- `tests/test_vlm_output.py` if parser diagnostics are extended

### Reference Files To Study

- `src/train_qlora.py`: loading metadata, checkpoint paths, and model config.
- `src/vlm.py`: JSON parsing and citation validation.
- `src/pipeline.py`: benchmark row/artifact shape.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: batch answer-generation
  orchestration and error handling.

### Implementation Notes

- Load adapter from `checkpoints/qlora_adapter` only when the path exists
  locally; do not make CI depend on it.
- Support batch inference over `val_split` or `sft_val`.
- Use configurable `max_new_tokens`; default should be at least `320`.
- Save JSONL rows with sample ID, target answer, raw output, parsed answer,
  exact match, parse status, truncation flag, latency, and adapter metadata.
- Count invalid JSON, missing citations, hallucinated citations, truncated
  outputs, and exact-match accuracy separately.
- Unit tests should use fake adapter outputs and must not load the real model.
- Do not commit checkpoint files or generated predictions.

### Expected Output

- A command can run adapter diagnostic inference for a small validation subset.
- Output JSONL is compatible with the evaluator or has a documented converter.
- The report can cite parse success and exact-match results without claiming
  submission-quality performance.

### Acceptance Criteria

- [ ] Adapter inference command supports `--split`, `--limit`,
  `--max-new-tokens`, `--adapter`, and `--output`.
- [ ] Parser diagnostics distinguish valid JSON, invalid JSON, truncation, and
  unsupported citation.
- [ ] Results include adapter metadata path/hash or a summary copied from
  `adapter_metadata.json`.
- [ ] `max_new_tokens=320` is documented as the safer smoke setting.
- [ ] Tests run without GPU/model/checkpoint.
- [ ] No checkpoint or generated adapter output is committed.

### Tests

- Unit test fake adapter output parses to a valid row.
- Unit test truncated JSON is counted separately.
- Unit test invalid citation is rejected or marked invalid.
- Unit test missing adapter path gives a helpful error.
- Manual GPU smoke run on 3-5 validation samples if GPU is available.

### Verification

```bash
python -m pytest tests/test_adapter_infer.py tests/test_vlm_output.py -q

# GPU/local-adapter diagnostic only:
python -m src.adapter_infer \
  --adapter checkpoints/qlora_adapter \
  --split val \
  --limit 5 \
  --max-new-tokens 320 \
  --output data/outputs/experiments/w4_adapter_diag.jsonl
```

## W4-02 - Full Validation And Error Analysis

Labels: `week-4`, `P0`, `evaluation`, `retrieval`, `vlm`

Milestone: `W4 - Demo and report`

Owner/reviewer: `M4` / `M3`

Branch: `experiment/w4-02-full-validation-error-analysis`

PR title: `experiment: run final validation and error analysis`

Depends on: `W3-05`, `W3-06`, `W4-01` optional

### Description

Run the final comparable validation experiments and analyze failures. The main
comparison should prioritize retrieval-grounded structured prompting. Include
QLoRA only as a diagnostic row if adapter inference is wired and clearly
labeled.

### Project Files To Change

- `configs/experiments/` (final validation configs)
- `src/evaluate.py` if additional diagnostic fields are needed
- `src/pipeline.py` if final model configs need wiring
- `docs/experiments.md`
- `docs/error-analysis.md`
- `tests/test_experiment_config.py`
- `tests/test_evaluate.py`

### Reference Files To Study

- `../LexiSignVQA-main/src/eval/sub_task_1.py`: retrieval P/R/F2.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: QA accuracy and breakdown.
- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: manual inspection of
  retrieval failures.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  retrieval ablation patterns.

### Implementation Notes

- Use the locked validation split and split hash.
- Suggested final rows:
  - frozen retrieval only/evidence inspection;
  - real/base VLM + text RAG if backend is available;
  - real/base VLM + structured legal RAG;
  - QLoRA adapter diagnostic, clearly marked non-final if available.
- Store retrieval metrics, QA accuracy, invalid count, parse failure count,
  truncated output count, and latency.
- Classify at least 30 failures across retrieval, visual ambiguity, legal
  context, reasoning, output-format, annotation/data, and adapter-truncation
  categories.
- Every reported number must trace to a JSON artifact.

### Expected Output

- Final metrics artifacts under `data/outputs/experiments/`.
- `docs/experiments.md` includes a final validation table.
- `docs/error-analysis.md` includes at least 30 categorized cases.
- The report can state clearly which system variant is the main product.

### Acceptance Criteria

- [ ] Final validation configs use the same locked split.
- [ ] Metrics table separates mock, real base VLM, structured prompting, and
  QLoRA diagnostic rows.
- [ ] Invalid JSON and truncation are counted separately from wrong answers.
- [ ] At least 30 error cases are categorized.
- [ ] QLoRA diagnostic results are not presented as final submission quality.
- [ ] Every metric in docs links to or names the source artifact.

### Tests

- Unit test final experiment config loading.
- Unit test split mismatch rejection.
- Unit test evaluator handles parse/truncation diagnostics.
- Manual review of 30 error-analysis entries.

### Verification

```bash
python -m pytest tests/test_experiment_config.py tests/test_evaluate.py -q
python -m src.pipeline --mode benchmark --config configs/experiments/w4_structured_rag.yaml
python -m src.evaluate \
  --config configs/experiments/w4_structured_rag.yaml \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl
```

## W4-03 - Submission Converter And Format Validator

Labels: `week-4`, `P0`, `evaluation`

Milestone: `W4 - Demo and report`

Owner/reviewer: `M1` / `M4`

Branch: `feat/w4-03-submission-converter`

PR title: `feat(eval): add submission converter and validator`

Depends on: `W4-02`

### Description

Convert internal prediction artifacts into the expected benchmark submission
format and validate that every required sample has a legal answer. This issue
is about packaging and safety, not improving model quality.

### Project Files To Change

- `src/submission.py` (new) or `src/evaluate.py` if the team prefers one module
- `scripts/evaluate.sh`
- `tests/test_submission.py` (new)
- `README.md`
- optional `docs/experiments.md`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: result saving
  shape and sample IDs.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: final answer extraction and
  train/public/private set handling.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: valid answer normalization.

### Implementation Notes

- Support conversion from `PipelineResult` JSONL to submission JSON.
- Validate required sample IDs against public/private test input when
  available.
- Validate answer format:
  - Multiple choice: `A/B/C/D`;
  - Yes/No: `Đúng/Sai`.
- Missing or invalid predictions should fail validation unless an explicit
  `--allow-missing` dry-run flag is used.
- Do not silently replace invalid answers with `A` or `Đúng`.
- Include citation output only if the target task format requires it; otherwise
  keep citations in the internal artifact.

### Expected Output

- A dry-run command validates prediction coverage and answer formats.
- A conversion command writes a small submission JSON file.
- Tests prove invalid/missing predictions are caught before submission.

### Acceptance Criteria

- [ ] Converter accepts internal JSONL predictions.
- [ ] Output preserves original sample IDs.
- [ ] Missing sample IDs are reported.
- [ ] Invalid answers are rejected.
- [ ] Public/private test paths are configurable.
- [ ] No raw private-test predictions are committed.

### Tests

- Unit test valid multiple-choice conversion.
- Unit test valid Yes/No conversion.
- Unit test missing sample ID fails.
- Unit test invalid answer fails.
- Unit test citations are not lost from internal artifact even if omitted from
  submission format.

### Verification

```bash
python -m pytest tests/test_submission.py -q
python -m src.submission \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl \
  --output data/outputs/submissions/w4_dry_run_submission.json \
  --dry-run
```

## W4-04 - Final Streamlit Demo

Labels: `week-4`, `P0`, `demo`, `vlm`

Milestone: `W4 - Demo and report`

Owner/reviewer: `M2` / `M1`

Branch: `feat/w4-04-final-streamlit-demo`

PR title: `feat(demo): polish final evidence-grounded Streamlit demo`

Depends on: `W4-02`, `W4-03` optional

### Description

Polish the Streamlit app into a final demonstration for defense. The demo
should work even when no live VLM backend is available by supporting
retrieval-only mode and cached prediction display.

### Project Files To Change

- `app/streamlit_app.py`
- `src/pipeline.py`
- `src/submission.py` or prediction loading helper if needed
- `README.md`
- `scripts/demo.sh`
- `tests/test_demo_contract.py`

### Reference Files To Study

- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: showing image, retrieved
  evidence, and article details.
- `../LexiSignVQA-main/src/ui/inspect_lawdb.py`: LawDB inspection.
- Current `app/streamlit_app.py`: existing evidence inspector.

### Implementation Notes

- Demo modes:
  - retrieval-only;
  - cached prediction display;
  - live VLM call if backend is configured.
- Show image, question, choices, retrieved evidence, citations, scores,
  answer, explanation, latency, and disclaimer.
- Add clear labels for mock/cached/live outputs.
- Avoid exposing secrets, local absolute paths unnecessarily, or hidden
  chain-of-thought.
- Include a small curated list of success/failure examples for presentation.
- Cache expensive resources with Streamlit caching.

### Expected Output

`streamlit run app/streamlit_app.py` opens a demo that can be used in the final
presentation without live GPU dependency.

### Acceptance Criteria

- [ ] Demo starts without VLM credentials.
- [ ] Retrieval-only mode works.
- [ ] Cached prediction mode works from a JSONL artifact.
- [ ] Live mode is available only when configured.
- [ ] Citations and legal evidence are visible.
- [ ] Disclaimer states that answers are educational/research assistance, not
  official legal advice.
- [ ] A screenshot or short recorded demo is produced for defense backup.

### Tests

- Unit test demo response contract.
- Unit test cached prediction loading.
- Unit test missing model credentials stays in retrieval-only mode.
- Manual run of Streamlit demo.

### Verification

```bash
python -m pytest tests/test_demo_contract.py -q
streamlit run app/streamlit_app.py
```

## W4-05 - Final Report, Slides, And Reproducibility Pack

Labels: `week-4`, `P0`, `documentation`

Milestone: `W4 - Demo and report`

Owner/reviewer: `M3` / `M2`

Branch: `docs/w4-05-final-report-slides`

PR title: `docs: complete final report slides and reproducibility pack`

Depends on: `W4-02`, `W4-03`, `W4-04`

### Description

Finish the defense-ready documentation. The report must tell a truthful story:
retrieval-grounded structured prompting is the main system, while QLoRA is an
experimental extension with real GPU evidence and clear limitations.

### Project Files To Change

- `docs/report.md`
- `docs/experiments.md`
- `docs/error-analysis.md`
- `docs/checkpoint-card.md`
- `docs/model-card.md` if separate from checkpoint card
- `README.md`
- optional `docs/assets/` for figures/screenshots
- final slides source if stored in repo

### Reference Files To Study

- `../LexiSignVQA-main/README.md`: concise pipeline and experiment reporting.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/README.md`: low-cost retrieval
  framing.
- The team's week-1, week-2, and week-3 reports/artifacts.

### Implementation Notes

- Include problem statement, dataset, legal corpus, grouped split, architecture,
  retrieval, prompting, evaluation metrics, experiments, QLoRA diagnostic,
  demo, limitations, ethics, member contributions, and four-month continuation
  plan.
- All reported numbers must come from repository artifacts or attached GPU
  logs.
- Do not claim that QLoRA improves final validation unless the artifact proves
  it.
- Include exact commands for preprocessing, indexing, validation, submission
  conversion, and demo.
- Include limitations:
  - no official legal advice;
  - possible visual misinterpretation;
  - retrieval errors;
  - JSON truncation/format failures;
  - current QLoRA adapter not submission-ready.

### Expected Output

- Final report is ready to submit or paste into the course template.
- Slides have a coherent story and match the implemented scope.
- Reproducibility commands are clear enough for another member to rerun.

### Acceptance Criteria

- [ ] Report covers all required course sections.
- [ ] Experiment table separates mock, real, and diagnostic rows.
- [ ] QLoRA checkpoint card is truthful and complete.
- [ ] Every metric has a source artifact.
- [ ] Demo screenshots or links are included.
- [ ] Four-month continuation plan is realistic.

### Tests

- Manual review against source artifacts.
- Manual review for overclaims.
- Run all documented commands in dry-run/smoke mode where feasible.

### Verification

```bash
python -m src.evaluate --help
python -m src.submission --help
make verify
git diff --check
```

## W4-06 - Final Release Cleanup And Defense Rehearsal

Labels: `week-4`, `P0`, `documentation`, `demo`, `evaluation`

Milestone: `W4 - Demo and report`

Owner/reviewer: `M4` / `M3`

Branch: `release/v0.1.0-course`

PR title: `release: publish final four-week capstone prototype`

Depends on: `W4-01`, `W4-02`, `W4-03`, `W4-04`, `W4-05`

### Description

Prepare the final release candidate and rehearse the defense flow. This issue
should catch packaging mistakes, broken commands, missing ignored artifacts,
and documentation drift before submission.

### Project Files To Change

- `README.md`
- `Makefile`
- `.env.example`
- `scripts/check_data.sh`
- `scripts/demo.sh`
- release notes or tag notes
- only fix blocking bugs elsewhere

### Reference Files To Study

- Project issue acceptance criteria from weeks 1-4.
- `.gitignore`: confirm artifacts remain ignored.
- `docs/report.md`: final commands and limitations.

### Implementation Notes

- Run a fresh-clone or clean-environment rehearsal if possible.
- Verify raw data, checkpoints, Qdrant storage, embeddings, and bulky outputs
  are not staged.
- Rehearse:
  - setup;
  - data check;
  - preprocess;
  - index;
  - one benchmark or cached prediction display;
  - submission dry-run;
  - Streamlit demo.
- Prepare backup screenshots/recorded demo in case live services fail.
- Tag the final course release only after checks pass.

### Expected Output

- Final PR/release tag with clean docs and working commands.
- Defense rehearsal notes and backup demo material.
- Clear statement of known limitations and next steps.

### Acceptance Criteria

- [ ] `make verify` passes in the supported local environment.
- [ ] README commands are current.
- [ ] `.env.example` documents required optional credentials without secrets.
- [ ] No ignored artifact is staged.
- [ ] Demo rehearsal succeeds or a backup recorded demo exists.
- [ ] Final release notes summarize implemented scope and limitations.

### Tests

- Run local verification commands.
- Manual `git status --ignored --short` review.
- Manual demo rehearsal.
- Manual report/slides consistency check.

### Verification

```bash
make verify
git status --ignored --short
python -m src.submission --help
streamlit run app/streamlit_app.py
git diff --check
```

## Suggested Parallel Work Order

```text
W4-01 --------\
W4-02 ---------> W4-05 -> W4-06
W4-03 ----\    /
W4-04 -----\--/
```

- `M3` owns adapter diagnostics and final technical narrative: W4-01, W4-05.
- `M4` owns validation, release, and command audit: W4-02, W4-06.
- `M1` owns submission converter and validation format safety: W4-03.
- `M2` owns final demo polish and evidence display: W4-04.

W4-01 should not block the final product if adapter inference remains weak.
W4-02 and W4-03 are the core release gates. W4-04 should support cached output
so the demo does not depend on live GPU availability.

## Week 4 Definition Of Done

- Final validation metrics are produced from the locked split or the blocker is
  documented clearly.
- Submission converter has a dry-run validator.
- Streamlit demo works in retrieval-only and cached-prediction modes.
- Final report/slides explain architecture, metrics, limitations, and QLoRA
  diagnostic evidence truthfully.
- Checkpoints, embeddings, Qdrant storage, raw data, and bulky outputs remain
  outside Git.
- The team has rehearsed the defense flow and prepared a backup demo.
