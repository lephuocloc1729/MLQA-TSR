# Four-Week Execution Plan

This plan covers the course delivery from 2026-06-22 to 2026-07-19. If the
team starts later, keep the same day-of-week cadence and shift all dates.

## 1. Delivery Scope

The four-week deliverable is a reproducible research prototype for static
Vietnamese traffic images and the two VLSP 2025 legal documents. It must:

- flatten and validate LawDB;
- use a split grouped by `image_id`;
- evaluate article retrieval with macro Precision, Recall, and F2;
- evaluate question answering with Accuracy and per-question-type breakdowns;
- compare zero-shot, retrieval-based few-shot, structured legal reasoning,
  and QLoRA;
- return a structured answer with valid `law_id` and `article_id` citations;
- provide a CLI benchmark and a Streamlit demonstration.

Traffic-sign detection, OCR, advanced image retrieval, and production
monitoring are stretch goals unless all required acceptance gates are already
green. Qdrant-backed text retrieval is part of the core Tier A pipeline.

## 2. Team Roles and File Ownership

Replace M1-M4 with the real names before creating GitHub issues.

| Member | Primary role | Owned files | Default reviewer |
|---|---|---|---|
| M1 | Data and validation lead | `src/data_utils.py`, `src/utils.py`, `src/collator.py`, data tests | M4 |
| M2 | Retrieval and vision lead | `src/vision.py`, `src/retrieval.py`, retrieval tests | M1 |
| M3 | VLM and training lead | `src/prompts.py`, `src/vlm.py`, `src/train_qlora.py`, model tests | M2 |
| M4 | Evaluation, app, and integration lead | `src/evaluate.py`, `src/pipeline.py`, `app/`, CI, release docs | M3 |

Shared files are `configs/*.yaml`, `requirements.txt`, `Makefile`, and
`README.md`. Only the issue explicitly assigned a shared file may modify it.
Other PRs should request the change in their issue instead of editing the
shared file opportunistically.

No member merges their own PR. M4 is release owner, but M3 reviews M4's
integration and release PRs.

## 3. Reference Map

The reference projects are siblings of this repository in the local
workspace. Reuse ideas and interfaces, not whole files without understanding
their assumptions and licenses.

### Low-Cost Project

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: LawDB and
  train/test dataset access.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/clean.py`: extraction of
  table and image markers.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/text_embedding.py`: text
  embedding wrapper.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/image_feature_extraction.py`:
  whole-image feature extraction and image resizing.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/object_detection.py`: object
  crop experiment; do not make it mandatory in week 1 or 2.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/db/qdrant.py`: named vector
  design; week 1 uses a simple Qdrant text collection while CI keeps vector
  store tests mocked.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/index_qdrant.ipynb`:
  indexing training examples and payload design.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/naive_vector_search.ipynb`:
  sequential text, full-image, and object retrieval.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`: top-3
  retrieved examples as multimodal few-shot messages and structured output.

### LexiSignVQA

- `../LexiSignVQA-main/src/core/lawdb/preprocess_lawdb_infos.py`: LawDB table,
  image marker, and article preprocessing.
- `../LexiSignVQA-main/src/core/lawdb/extract_lawdb_sign_images.py`: optional
  sign extraction from clean LawDB images.
- `../LexiSignVQA-main/src/core/lawdb/ingest_lawdb_signs.py`: sign payloads
  linked to law and article identifiers.
- `../LexiSignVQA-main/src/core/sub_task_1.py`: task 1 orchestration.
- `../LexiSignVQA-main/src/core/extract_signs.py`, `filter_signs.py`, and
  `query_signs.py`: optional month-two sign pipeline.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: answer generation and response
  normalization.
- `../LexiSignVQA-main/src/prompts/answer_prompt.py`: output constraints.
- `../LexiSignVQA-main/src/eval/sub_task_1.py` and `sub_task_2.py`: official
  metric implementation and breakdowns.
- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: showing images, retrieved
  articles, and legal evidence in Streamlit.

## 4. GitHub Setup

Create four milestones:

| Milestone | Due date | Exit gate |
|---|---|---|
| `W1 - Trusted baseline` | 2026-06-28 | Clean setup, correct data, locked split, B0/B1 metrics |
| `W2 - Retrieval and few-shot` | 2026-07-05 | Task 1 retrieval and B3 run end to end |
| `W3 - QLoRA and ablations` | 2026-07-12 | Training smoke test and experiment matrix complete |
| `W4 - Demo and report` | 2026-07-19 | CI, demo, report, error analysis, and release tag |

Create these labels: `week-1`, `week-2`, `week-3`, `week-4`, `data`,
`retrieval`, `vlm`, `training`, `evaluation`, `demo`, `documentation`,
`blocked`, `stretch`, `needs-gpu`, and priorities `P0`, `P1`, `P2`.

Issue keys such as `W1-01` are stable planning IDs. After creating an issue,
put its GitHub number beside the key in this document.

## 5. Week 1: Trusted Baseline

Detailed GitHub-ready issue bodies for this week are in
[`docs/week1-issues.md`](week1-issues.md). Use that file as the source of truth
when creating the actual week-1 issues.

### W1-01 - Reproducible environment and CI

- **Owner/reviewer:** M4/M3; **priority:** P0; **days:** Mon-Tue.
- **Change:** `requirements.txt`, `Makefile`, `.env.example`, `README.md`, new
  `.github/workflows/ci.yml`.
- **Reference:** `LexiSignVQA-main/pyproject.toml` and `Makefile`.
- **Acceptance:** versions are pinned; setup instructions work in a fresh
  environment; CI runs lightweight unit tests without downloading models;
  secrets and raw data remain ignored.
- **PR:** `chore/w1-01-reproducible-setup`, title
  `chore: make development setup reproducible`.

### W1-02 - Flatten and clean LawDB

- **Owner/reviewer:** M1/M4; **priority:** P0; **days:** Mon-Wed.
- **Change:** `src/data_utils.py`, `src/utils.py`, new
  `tests/test_data_utils.py`; regenerate but do not commit large raw payloads.
- **Reference:** low-cost `src/data/dataset.py`, `src/data/clean.py`; LexiSign
  `src/core/lawdb/preprocess_lawdb_infos.py`.
- **Acceptance:** output has exactly 402 article records; every record contains
  `law_id`, `law_title`, `article_id`, `title`, `content`, `images`, and
  `tables`; no record embeds the complete source document under `raw`.
- **PR:** `feat/w1-02-lawdb-parser`, title
  `feat(data): flatten LawDB into article records`.

### W1-03 - Normalize data and create leakage-safe splits

- **Owner/reviewer:** M1/M4; **priority:** P0; **depends on:** W1-02;
  **days:** Thu-Fri.
- **Change:** `src/data_utils.py`, `configs/config.yaml`, new
  `tests/test_data_split.py` and `data/processed/split_manifest.json`.
- **Reference:** low-cost `src/data/dataset.py`; LexiSign
  `src/core/sub_task_2.py` for NFC answer normalization.
- **Acceptance:** train/validation groups are disjoint by `image_id`; all 530
  samples occur exactly once; split is deterministic; `Đúng` variants and the
  exceptional numeric label are handled by an explicit audited mapping.
- **PR:** `feat/w1-03-grouped-split`, title
  `feat(data): add grouped train validation split`.

### W1-04 - Qdrant text retrieval for legal articles

- **Owner/reviewer:** M2/M1; **priority:** P0; **depends on:** W1-02;
  **days:** Wed-Fri.
- **Change:** `src/retrieval.py`, `configs/config.yaml`, `scripts/index.sh`,
  new `tests/test_retrieval.py`.
- **Reference:** LexiSign `src/deps/qdrant.py`; low-cost `src/db/qdrant.py`
  and `src/nn/text_embedding.py`.
- **Acceptance:** index LawDB articles into one Qdrant text collection; query
  from question and choices; return top-k unique `Evidence` with rank, score,
  and citation IDs; CI tests use fake embeddings/vector store.
- **PR:** `feat/w1-04-qdrant-text-retrieval`, title
  `feat(retrieval): index LawDB articles with Qdrant text search`.

### W1-05 - Evaluation metrics and result artifact contract

- **Owner/reviewer:** M4/M3; **priority:** P0; **depends on:** W1-03;
  **days:** Tue-Thu.
- **Change:** `src/evaluate.py`, `docs/experiments.md`, new
  `tests/test_evaluate.py`.
- **Reference:** LexiSign `src/eval/sub_task_1.py` and
  `src/eval/sub_task_2.py`.
- **Acceptance:** tested macro Precision/Recall/F2 for retrieval, overall and
  per-type Accuracy for QA, invalid prediction count, and JSON artifacts with
  config, seed, split path/hash, latency, metrics, and failed IDs.
- **PR:** `feat/w1-05-evaluation-metrics`, title
  `feat(eval): implement retrieval and QA metrics`.

### W1-06 - Structured prompting and VLM output parser

- **Owner/reviewer:** M3/M2; **priority:** P0; **depends on:** W1-02, W1-04;
  **days:** Thu-Sat.
- **Change:** `src/prompts.py`, `src/vlm.py`, `configs/config.yaml`, new
  `tests/test_vlm_output.py`.
- **Reference:** LexiSign `src/prompts/answer_prompt.py` and
  `src/core/sub_task_2.py`; low-cost `notebooks/vlm_answer.ipynb`.
- **Acceptance:** prompt includes image, question, choices, and retrieved legal
  evidence; parser converts JSON responses into `Prediction`; hallucinated
  citations outside retrieved `Evidence` are rejected without model weights in
  tests.
- **PR:** `feat/w1-06-vlm-parser`, title
  `feat(vlm): add structured legal QA prompt and parser`.

### W1-07 - Text-RAG pipeline smoke run and weekly report

- **Owner/reviewer:** M4/M3; **priority:** P0; **depends on:** W1-01..W1-06;
  **day:** Sun.
- **Change:** `src/pipeline.py`, `scripts/preprocess.sh`, `scripts/index.sh`,
  `scripts/evaluate.sh`, `Makefile`, `README.md`, `docs/report.md`,
  `docs/experiments.md`, optional placeholder in `app/streamlit_app.py`.
- **Reference:** LexiSign `src/core/sub_task_1.py`,
  `src/core/sub_task_2.py`, and `src/ui/inspect_subtask1.py`.
- **Acceptance:** a five-sample smoke run produces `PipelineResult`-compatible
  records; `make eval` evaluates the smoke output; weekly report records
  completed issues, dataset audit, current metrics, blockers, and next steps.
- **PR:** `release/w1-trusted-baseline`, title
  `release: integrate week 1 trusted baseline`.

## 6. Week 2: Retrieval and Few-Shot

### W2-01 - Text and whole-image feature extraction

- **Owner/reviewer:** M2/M1; **priority:** P0; **days:** Mon-Tue.
- **Change:** `src/vision.py`, `src/retrieval.py`, `configs/config.yaml`, new
  `tests/test_vision.py`.
- **Reference:** low-cost `src/nn/text_embedding.py` and
  `src/nn/image_feature_extraction.py`.
- **Acceptance:** batched CPU/GPU-aware embedding interface; normalized vectors;
  cache includes model name and data hash; tests use synthetic vectors/images.
- **PR:** `feat/w2-01-embeddings`, title
  `feat(retrieval): extract cached text and image features`.

### W2-02 - Multimodal example index and top-k search

- **Owner/reviewer:** M2/M1; **priority:** P0; **depends on:** W2-01;
  **days:** Wed-Thu.
- **Change:** `src/retrieval.py`, new `tests/test_retrieval.py`; reuse the
  week-1 Qdrant text adapter and mock the vector store in CI so Docker is not
  mandatory for tests.
- **Reference:** low-cost `src/db/qdrant.py`, `notebooks/index_qdrant.ipynb`,
  and `notebooks/naive_vector_search.ipynb`.
- **Acceptance:** text-only, image-only, and weighted multimodal search; query
  never retrieves examples from the validation group; deterministic top-k;
  payload preserves sample ID and relevant articles.
- **PR:** `feat/w2-02-example-index`, title
  `feat(retrieval): add multimodal example search`.

### W2-03 - Aggregate citations and look up legal evidence

- **Owner/reviewer:** M1/M4; **priority:** P0; **depends on:** W1-02, W2-02;
  **days:** Wed-Fri.
- **Change:** `src/data_utils.py`, `src/retrieval.py`, new
  `tests/test_law_lookup.py`.
- **Reference:** LexiSign `src/core/query_signs.py` and
  `src/core/lawdb/ingest_lawdb_signs.py`.
- **Acceptance:** rank-weighted article aggregation; deduplicated valid IDs;
  each citation resolves to title and content; no unconditional default such as
  article 22; unknown IDs are reported rather than silently replaced.
- **PR:** `feat/w2-03-citation-lookup`, title
  `feat(retrieval): resolve retrieved examples to legal evidence`.

### W2-04 - Retrieval-based top-3 few-shot B3

- **Owner/reviewer:** M3/M2; **priority:** P0; **depends on:** W2-02;
  **days:** Thu-Fri.
- **Change:** `src/prompts.py`, `src/vlm.py`, new
  `tests/test_few_shot_prompt.py`.
- **Reference:** low-cost `notebooks/vlm_answer.ipynb`.
- **Acceptance:** three retrieved examples include image, question, choices,
  and answer; validation answers are never exposed; token/image budget is
  bounded; result follows the same schema as B1.
- **PR:** `feat/w2-04-few-shot`, title
  `feat(vlm): add retrieved multimodal few-shot prompting`.

### W2-05 - Retrieval and prompt ablation

- **Owner/reviewer:** M4/M3; **priority:** P1; **depends on:** W2-02..W2-04;
  **days:** Fri-Sat.
- **Change:** `src/evaluate.py`, `docs/experiments.md`, new experiment configs
  under `configs/experiments/`.
- **Reference:** LexiSign `README.md` experiment table and evaluation scripts;
  low-cost paper/notebook top-k settings.
- **Acceptance:** compare text, image, combined modalities and top-k 1/2/3;
  compare B1 with B3 on the same split; store metrics, latency, and failed IDs.
- **PR:** `experiment/w2-05-retrieval-ablation`, title
  `experiment: compare multimodal retrieval and few-shot settings`.

### W2-06 - Week 2 integration and report

- **Owner/reviewer:** M4/M3; **priority:** P0; **depends on:** W2-01..W2-05;
  **day:** Sun.
- **Change:** `src/pipeline.py`, `Makefile`, `README.md`, `docs/report.md`.
- **Acceptance:** one command builds the index and evaluates B3; weekly report
  includes Task 1 P/R/F2, B1/B3 Accuracy, ablation, errors, and contributions;
  tag release `w2-retrieval`.
- **PR:** `release/w2-retrieval`, title
  `release: integrate retrieval and few-shot pipeline`.

## 7. Week 3: QLoRA and Controlled Experiments

### W3-01 - Build SFT examples and multimodal collator

- **Owner/reviewer:** M1/M4; **priority:** P0; **days:** Mon-Tue.
- **Change:** `src/data_utils.py`, `src/collator.py`, new
  `tests/test_collator.py` and generated `data/processed/sft_*.jsonl`.
- **Reference:** low-cost `src/data/dataset.py`; current base model processor
  documentation rather than copying a reference implementation blindly.
- **Acceptance:** only training groups create SFT records; validation remains
  locked; messages contain image, question, choices, legal context, and target
  answer; collator masks padding and non-target tokens correctly.
- **PR:** `feat/w3-01-sft-data`, title
  `feat(training): build leakage-safe multimodal SFT data`.

### W3-02 - QLoRA trainer and smoke test

- **Owner/reviewer:** M3/M2; **priority:** P0; **needs:** GPU; **depends on:**
  W3-01; **days:** Tue-Thu.
- **Change:** `src/train_qlora.py`, `configs/qlora.yaml`, new
  `tests/test_train_config.py`.
- **Reference:** the current base model and PEFT official examples; use the two
  local projects only for input/output format because neither trains QLoRA.
- **Acceptance:** 20-sample overfit/smoke run completes; trainable parameter
  count and VRAM are logged; checkpoint resumes; vision encoder is frozen;
  adapter and processor metadata are saved.
- **PR:** `feat/w3-02-qlora`, title
  `feat(training): add reproducible QLoRA training`.

### W3-03 - Structured legal reasoning and citation schema B4

- **Owner/reviewer:** M3/M2; **priority:** P1; **days:** Mon-Wed.
- **Change:** `src/prompts.py`, `src/vlm.py`, new
  `tests/test_citation_output.py`.
- **Reference:** LexiSign `src/prompts/sign_filter_prompt.py` for structured
  stages and `src/core/sub_task_2.py` for answer extraction.
- **Acceptance:** output fields are `observation`, `legal_basis`, `conclusion`,
  `answer`, and `citations`; final answer remains machine-parseable; every
  citation is validated against LawDB; concise rationale replaces unrestricted
  chain-of-thought storage.
- **PR:** `feat/w3-03-legal-reasoning`, title
  `feat(vlm): add structured legal reasoning and citations`.

### W3-04 - Freeze retrieval configuration and analyze hard cases

- **Owner/reviewer:** M2/M1; **priority:** P1; **depends on:** W2-05;
  **days:** Mon-Wed.
- **Change:** `src/retrieval.py`, `configs/experiments/retrieval_final.yaml`,
  `docs/experiments.md`.
- **Reference:** low-cost sequential prefetch in
  `notebooks/naive_vector_search.ipynb`; LexiSign sign retrieval ablations in
  `README.md`.
- **Acceptance:** final weights/top-k chosen using validation only; analyze at
  least 15 retrieval failures; object detection remains a documented stretch
  issue unless it demonstrably improves the locked split.
- **PR:** `experiment/w3-04-freeze-retrieval`, title
  `experiment: freeze retrieval configuration from ablations`.

### W3-05 - Run B2-B6 experiment matrix

- **Owner/reviewer:** M4/M3; **priority:** P0; **depends on:** W3-02..W3-04;
  **days:** Thu-Sat.
- **Change:** `src/evaluate.py`, `src/pipeline.py`, `scripts/evaluate.sh`,
  `docs/experiments.md`.
- **Reference:** both reference projects' evaluation scripts and experiment
  tables.
- **Acceptance:** same locked validation set for zero-shot, oracle-law,
  few-shot, structured reasoning, QLoRA+retrieved-law, and QLoRA+oracle-law;
  failures do not silently become answer A/Đúng; all artifacts are versioned.
- **PR:** `experiment/w3-05-model-matrix`, title
  `experiment: evaluate prompting retrieval and QLoRA variants`.

### W3-06 - Week 3 integration and report

- **Owner/reviewer:** M4/M3; **priority:** P0; **depends on:** W3-01..W3-05;
  **day:** Sun.
- **Change:** `Makefile`, `README.md`, `docs/report.md`, model/checkpoint card.
- **Acceptance:** exact training and evaluation commands documented; checkpoint
  is stored outside Git with a checksum and access instructions; weekly report
  records compute, Accuracy, retrieval ceiling, risks, and contributions; tag
  release `w3-qlora`.
- **PR:** `release/w3-qlora`, title
  `release: integrate QLoRA and controlled experiments`.

## 8. Week 4: Product, Quality, and Defense Package

### W4-01 - End-to-end assistant pipeline

- **Owner/reviewer:** M4/M3; **priority:** P0; **days:** Mon-Tue.
- **Change:** `src/pipeline.py`, `configs/config.yaml`, `scripts/demo.sh`, new
  `tests/test_pipeline.py`.
- **Reference:** LexiSign `src/core/sub_task_1.py` and `sub_task_2.py`.
- **Acceptance:** image+question input returns answer, evidence, citations,
  scores, and latency; components are injectable for tests; helpful errors for
  missing model/index/data; no broad exception that returns a fake answer.
- **PR:** `feat/w4-01-assistant`, title
  `feat(pipeline): integrate end-to-end legal assistant`.

### W4-02 - Streamlit demonstration

- **Owner/reviewer:** M2/M1; **priority:** P0; **depends on:** W4-01;
  **days:** Tue-Thu.
- **Change:** `app/streamlit_app.py`, `app/__init__.py`, optional UI helpers.
- **Reference:** LexiSign `src/ui/inspect_subtask1.py`.
- **Acceptance:** upload image, enter question/choices, select approved model
  configuration, view answer and cited legal text; show latency and disclaimer;
  cache resources; do not display secrets or unrestricted internal reasoning.
- **PR:** `feat/w4-02-streamlit`, title
  `feat(demo): add evidence-grounded Streamlit interface`.

### W4-03 - End-to-end tests and release checks

- **Owner/reviewer:** M1/M4; **priority:** P0; **days:** Tue-Thu.
- **Change:** `tests/`, `scripts/check_data.sh`, `.github/workflows/ci.yml`.
- **Reference:** project schemas and acceptance criteria in this plan.
- **Acceptance:** unit tests cover normalization, split leakage, metrics,
  retrieval exclusion, output parsing, citations, and mocked pipeline; data
  check uses actual configured paths; CI passes from a clean checkout.
- **PR:** `test/w4-03-release-suite`, title
  `test: add end-to-end release checks`.

### W4-04 - Error analysis and responsible-use evaluation

- **Owner/reviewer:** M2/M1; **priority:** P1; **days:** Thu-Fri.
- **Change:** new `docs/error-analysis.md`, `docs/experiments.md`.
- **Reference:** failure analyses in both reference project papers and LexiSign
  inspection UI.
- **Acceptance:** manually classify at least 30 errors into visual, retrieval,
  legal-context, reasoning, output-format, and annotation/data errors; include
  representative IDs and proposed fixes; test citation validity and refusal for
  unsupported inputs.
- **PR:** `docs/w4-04-error-analysis`, title
  `docs: analyze model failures and responsible use`.

### W4-05 - Final report and defense assets

- **Owner/reviewer:** M3/M2; **priority:** P0; **days:** Wed-Sat.
- **Change:** `docs/report.md`, new `docs/model-card.md`, final experiment
  figures under `docs/assets/`.
- **Reference:** methodology and experiment organization from both reference
  READMEs and papers; all reported numbers must come from this repository's
  artifacts.
- **Acceptance:** problem, dataset, leakage-safe methodology, architecture,
  experiment matrix, ablations, errors, limitations, ethics, individual work,
  and four-month continuation plan; no claim copied from a reference project as
  the team's own result.
- **PR:** `docs/w4-05-final-report`, title
  `docs: complete report model card and defense assets`.

### W4-06 - Final release and demonstration rehearsal

- **Owner/reviewer:** M4/M3; **priority:** P0; **depends on:** W4-01..W4-05;
  **day:** Sun.
- **Change:** `README.md`, `Makefile`, `.env.example`, release notes; only fix
  blocking integration defects elsewhere.
- **Acceptance:** fresh-clone rehearsal; `make test`, `make benchmark`, and
  `make demo` documented and passing in their supported environments; backup
  recorded demo; final metrics attached; tag `v0.1.0-course`.
- **PR:** `release/v0.1.0-course`, title
  `release: publish four-week capstone prototype`.

## 9. PR Order and Parallel Work

```text
W1-01
  -> W1-02 -> W1-03
          \-> W1-04
W1-03 --------> W1-05
W1-04 --------> W1-06
W1-01..W1-06 -> W1-07

W2-01 -> W2-02 -> W2-03 -----> W2-06
                 \-> W2-04 -> W2-05 -/

W3-01 -> W3-02 --------\
W3-03 -------------------> W3-05 -> W3-06
W3-04 ------------------/

W4-01 -> W4-02 ----\
       -> W4-03 -----\
W4-04 ---------------> W4-06
W4-05 --------------/
```

M1 starts W1-02 and then W1-03. M2 can prepare the retrieval interface while
waiting for W1-02, then owns W1-04. M3 starts W1-06 with fake evidence and
connects it to real retrieval after W1-04. M4 starts W1-01 and W1-05, then
owns W1-07 integration. M3 must not run final baseline experiments before the
split and metric PRs merge.

## 10. Branch and Pull Request Rules

- Branch format: `<type>/<issue-key>-<short-name>`, for example
  `feat/w2-02-example-index`.
- Open a draft PR within one working day so interface conflicts are visible.
- Rebase or merge latest `main` before requesting final review.
- Prefer one issue per PR; split a PR above roughly 500 changed source lines.
- Use `Closes #<github-issue-number>` in the PR description.
- Attach command output, metric artifact, or screenshot appropriate to the task.
- Never commit `.env`, raw VLSP data, model weights, caches, or Qdrant storage.
- Generated metrics may be committed only when small and linked to the exact
  config/split hash; large embeddings and checkpoints use external storage.
- A behavior-changing PR needs tests. A GPU-only path also needs CPU-safe
  configuration/parser tests in CI.

Required checks before merge:

```bash
make ci-test
python -m pip check
git diff --check
```

Add broader checks such as `python -m src.data_utils --mode validate` and
`python -m src.evaluate --help` after the corresponding week-1 issues implement
those commands.

## 11. Weekly Reporting Ceremony

Hold the report every Sunday before merging the weekly release PR. Store the
report in `docs/report.md` and attach it to the milestone.

Each report must include:

1. Completed issue and PR links per member.
2. Commands and environment used.
3. Dataset/split or model version.
4. Metrics compared with the prior week.
5. Five representative successes and failures.
6. Risks, blockers, owner, and resolution date.
7. Next week's committed issues.

Missing work is not marked complete to make the report look better. Move it to
the next milestone with a written impact and remove a lower-priority task if
needed.

## 12. Scope-Cut Order

If the schedule slips, cut in this order:

1. Qdrant tuning, persistence, and deployment extras; keep the simple text
   retrieval path working.
2. Object detection and cropped-sign retrieval.
3. Extra embedding models beyond one text and one image model.
4. Advanced UI styling.
5. QLoRA hyperparameter sweep; keep one smoke run and one controlled run.

Never cut the grouped split, metrics, zero-shot baseline, retrieval baseline,
tests, citation validation, weekly evidence, or final error analysis.
