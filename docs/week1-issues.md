# Week 1 Issues - Trusted Text-RAG Baseline

This issue pack follows the agreed Tier A scope:

```text
traffic image + question + choices
  -> text retrieval from question/choices
  -> top-k legal articles from Qdrant
  -> VLM receives image + question + legal evidence
  -> ANSWER + CITATION + EXPLANATION
  -> evaluate + Streamlit-ready output contract
```

Week 1 should not implement sign crop, OCR, image retrieval, or QLoRA yet.
Those belong to later upgrades. The goal of week 1 is to make the data,
retrieval, evaluation, and model-output contract trustworthy enough that the
team can build on it safely.

Replace `M1`-`M4` with real member names before creating GitHub issues.

## W1-01 - Reproducible Setup, CI, And Scope Alignment

Labels: `week-1`, `P0`, `documentation`, `evaluation`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M4` / `M3`

Branch: `chore/w1-01-reproducible-setup`

PR title: `chore: make week 1 setup reproducible`

### Description

Make the repository setup reliable for all four members and prevent the team
from drifting away from the agreed Tier A scope. This issue is mostly
infrastructure and documentation. Do not download model weights in CI.

### Project Files To Change

- `requirements.txt`
- `README.md`
- `Makefile`
- `.env.example`
- `.github/workflows/ci.yml` (new)
- `docs/4-week-execution-plan.md`
- `docs/week1-issues.md`

### Reference Files To Study

- `../LexiSignVQA-main/pyproject.toml`: learn how dependencies are grouped and
  kept explicit.
- `../LexiSignVQA-main/Makefile`: learn how common commands are exposed.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/scripts/pip_setup.sh`: learn what
  setup assumptions the low-cost project used.

### Implementation Notes

- Keep Qdrant text retrieval in the core scope.
- Keep sign crop, OCR, advanced image retrieval, and production monitoring as
  stretch work.
- CI should run only lightweight tests that do not require raw VLSP data,
  external API keys, GPU, or model downloads.
- Because the local macOS environment may segfault on plain `pytest` due to
  `readline`, document the workaround if it appears on a member's machine.

### Expected Output

- A fresh team member can follow `README.md` and install the environment.
- `make test` or the documented fallback test command runs schema/unit tests.
- CI exists and is intentionally lightweight.
- Documentation clearly says that week 1 builds a trusted text-RAG baseline,
  not the full four-month graduation system.

### Acceptance Criteria

- [ ] `requirements.txt` remains pinned and installable on macOS/CPU.
- [ ] `.env.example` lists only non-secret placeholder variables.
- [ ] `README.md` includes setup, data placement, and week 1 command summary.
- [ ] CI runs tests without downloading VLM/embedding model weights.
- [ ] Planning docs no longer describe Qdrant text retrieval as a stretch goal.
- [ ] Raw data, `.env`, model weights, Qdrant storage, and caches are ignored.

### Verification

```bash
python -m pip check
python -m pip install --dry-run --no-index -r requirements.txt
make test
git diff --check
```

If plain `pytest` segfaults locally because of `readline`, verify the schema
tests with:

```bash
python - <<'PY'
import sys, types
sys.modules["readline"] = types.ModuleType("readline")
import pytest
raise SystemExit(pytest.main(["-q", "tests/test_schemas.py"], plugins=[]))
PY
```

## W1-02 - Flatten LawDB Into Article Records

Labels: `week-1`, `P0`, `data`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M1` / `M4`

Branch: `feat/w1-02-lawdb-parser`

PR title: `feat(data): flatten LawDB into article records`

### Description

Fix `src/data_utils.py` so it parses the real LawDB structure instead of
assuming the top-level documents are article records. The output will become
the legal corpus used by Qdrant retrieval and citation validation.

### Project Files To Change

- `src/data_utils.py`
- `src/utils.py` if small helper functions are needed
- `data/processed/law_articles.jsonl` generated locally
- `tests/test_data_utils.py` (new)
- `scripts/preprocess.sh`

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: copy the
  idea of `LawCorpus.walk_through()` and `get_by(law_id, article_id)`.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/clean.py`: reuse the
  approach of extracting `<<TABLE:.../TABLE>>` and `<<IMAGE:.../IMAGE>>`
  markers.
- `../LexiSignVQA-main/src/core/lawdb/preprocess_lawdb_infos.py`: learn how it
  preserves image markers and table content for later sign/legal retrieval.

### Implementation Notes

- Input LawDB path comes from `configs/config.yaml`.
- Current dataset has 2 legal documents and 402 total articles:
  `QCVN 41:2024/BGTVT` has 313 articles, `36/2024/QH15` has 89 articles.
- Each output line should be one article, not one law document.
- Prefer stable article UID format: `"{law_id}#{article_id}"`.
- Do not store the entire original document under a `raw` field in every row.

### Expected Output

`data/processed/law_articles.jsonl` contains exactly 402 JSONL rows. Each row
has this minimum shape:

```json
{
  "uid": "QCVN 41:2024/BGTVT#22",
  "law_id": "QCVN 41:2024/BGTVT",
  "law_title": "...",
  "article_id": "22",
  "title": "...",
  "content": "...",
  "images": [],
  "tables": []
}
```

### Acceptance Criteria

- [ ] `python -m src.data_utils --mode preprocess` generates 402 article rows.
- [ ] Every row has non-empty `uid`, `law_id`, `law_title`, `article_id`,
  `title`, and `content`.
- [ ] `images` and `tables` are lists even when empty.
- [ ] `get_law_article(law_id, article_id)` or equivalent lookup returns one
  article in O(1) or near O(1) from the processed corpus.
- [ ] Unknown article IDs raise/report a clear error instead of silently
  defaulting to article 22 or another common article.

### Tests

- Unit test `clean_html`/marker parsing with synthetic article text containing
  one table and one image.
- Unit test processed article count using the real local LawDB when available.
- Unit test lookup for `QCVN 41:2024/BGTVT#22`.
- Unit test duplicate UID detection.

### Verification

```bash
python -m src.data_utils --mode preprocess
python -m pytest tests/test_data_utils.py -q
python - <<'PY'
from pathlib import Path
import json
rows = [json.loads(line) for line in Path("data/processed/law_articles.jsonl").read_text(encoding="utf-8").splitlines()]
print(len(rows))
print(rows[0].keys())
assert len(rows) == 402
PY
```

## W1-03 - Validate VLSP Samples And Create Grouped Split

Labels: `week-1`, `P0`, `data`, `evaluation`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M1` / `M4`

Branch: `feat/w1-03-grouped-split`

PR title: `feat(data): add validated grouped train validation split`

### Description

Create a deterministic train/validation split that avoids image leakage. This
is the foundation for all later comparisons: zero-shot, retrieval, few-shot,
and QLoRA.

### Project Files To Change

- `src/data_utils.py`
- `src/schemas.py` only if the existing contract truly needs an extension
- `configs/config.yaml`
- `data/processed/train_split.jsonl` generated locally
- `data/processed/val_split.jsonl` generated locally
- `data/processed/split_manifest.json` generated locally
- `tests/test_data_split.py` (new)

### Reference Files To Study

- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/data/dataset.py`: learn how
  train samples are loaded and image paths are attached.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn the NFC answer
  normalization and the legacy integer `40 -> A` handling.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: learn how answer and question
  type distributions are reported.

### Implementation Notes

- Use `Query` and `Citation` from `src/schemas.py` to validate samples.
- Current train set has 530 samples and 304 unique images.
- Split by `image_id`, not by row index.
- Use project seed from `configs/config.yaml`, currently `42`.
- Store manifest fields such as seed, split ratio, created_at, train_count,
  val_count, train_image_count, val_image_count, and config hash if available.

### Expected Output

- `data/processed/train_split.jsonl`
- `data/processed/val_split.jsonl`
- `data/processed/split_manifest.json`

The two split files together contain all 530 training samples exactly once,
with no overlapping `image_id`.

### Acceptance Criteria

- [ ] All 530 samples validate with `Query`.
- [ ] Train and validation `image_id` sets are disjoint.
- [ ] No sample ID appears in both splits.
- [ ] The split is deterministic for the same seed.
- [ ] Multiple-choice samples have choices `A/B/C/D`.
- [ ] Yes/No answers are normalized to `Đúng` or `Sai`.
- [ ] The legacy numeric answer `40` is mapped explicitly and counted in the
  manifest/audit log.

### Tests

- Unit test deterministic split for a synthetic dataset.
- Unit test no image leakage.
- Unit test answer normalization for `Đúng`, `Đúng`, `Sai`, and `40`.
- Unit test every `relevant_articles` item resolves to a LawDB article after
  W1-02 is merged.

### Verification

```bash
python -m src.data_utils --mode split
python -m src.data_utils --mode validate
python -m pytest tests/test_data_split.py tests/test_schemas.py -q
```

## W1-04 - Build Qdrant Text Retrieval For Legal Articles

Labels: `week-1`, `P0`, `retrieval`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M2` / `M1`

Branch: `feat/w1-04-qdrant-text-retrieval`

PR title: `feat(retrieval): index LawDB articles with Qdrant text search`

Depends on: `W1-02`

### Description

Implement the Tier A retrieval component: embed the question plus choices,
query Qdrant, and return top-k legal article evidence. Keep this issue
text-only. Do not add image retrieval, sign crop, or OCR here.

### Project Files To Change

- `src/retrieval.py`
- `configs/config.yaml`
- `scripts/index.sh`
- `scripts/evaluate.sh` only if retrieval evaluation command needs wiring
- `tests/test_retrieval.py` (new)

### Reference Files To Study

- `../LexiSignVQA-main/src/deps/qdrant.py`: learn the small
  `QdrantVectorStore` wrapper shape: create collection, add embeddings, query.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/db/qdrant.py`: learn named
  vector configuration, but use only one text vector in week 1.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/src/nn/text_embedding.py`: learn
  the idea of wrapping the embedding model behind one class.

### Implementation Notes

- Use `sentence-transformers` or another configured text embedding model from
  `configs/config.yaml`.
- For tests, inject a fake embedder and fake vector store so CI does not
  download models or require Docker.
- Payload should include at least `uid`, `law_id`, `law_title`, `article_id`,
  `title`, `content`, and optionally `source_path`.
- Return retrieved items as `Evidence` from `src/schemas.py`.
- Query text should concatenate question and choices. Example:
  `question + "\nA. ...\nB. ...\nC. ...\nD. ..."` for multiple-choice samples.
- Keep `top_k` configurable. Start with `top_k: 5`.

### Expected Output

- `python -m src.retrieval --mode index` creates or recreates the Qdrant
  collection from `data/processed/law_articles.jsonl`.
- `python -m src.retrieval --mode retrieve --sample-id train_1` prints top-k
  evidence with scores and citations.
- Retrieval results are serializable as `Evidence` and can be passed directly
  into `PipelineResult`.

### Acceptance Criteria

- [ ] The Qdrant collection uses cosine distance and one text vector.
- [ ] Indexing fails with a helpful message if LawDB has not been preprocessed.
- [ ] Retrieval returns at most `top_k` unique article UIDs.
- [ ] Retrieved evidence includes score, rank, and `retrieval_method="text"`.
- [ ] No retrieval code depends on validation/gold answer fields.
- [ ] Unit tests pass with fake embeddings and do not need Docker.

### Tests

- Unit test query text construction for Multiple choice and Yes/No.
- Unit test deduplication and rank ordering.
- Unit test Qdrant payload conversion into `Evidence`.
- Unit test missing processed LawDB file gives a clear error.

### Verification

```bash
make qdrant-up
python -m src.data_utils --mode preprocess
python -m src.retrieval --mode index
python -m src.retrieval --mode retrieve --sample-id train_1 --top-k 5
python -m pytest tests/test_retrieval.py -q
```

## W1-05 - Implement Evaluation Metrics And Result Artifact Contract

Labels: `week-1`, `P0`, `evaluation`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M4` / `M3`

Branch: `feat/w1-05-evaluation-metrics`

PR title: `feat(eval): implement retrieval and QA metrics`

Depends on: `W1-03`

### Description

Implement official-style metrics for retrieval and QA so every experiment is
comparable. This issue should also define the JSON artifact format for
experiment outputs.

### Project Files To Change

- `src/evaluate.py`
- `docs/experiments.md`
- `tests/test_evaluate.py` (new)
- `scripts/evaluate.sh`

### Reference Files To Study

- `../LexiSignVQA-main/src/eval/sub_task_1.py`: reuse the per-sample
  Precision/Recall/F2 idea for `relevant_articles` vs predicted articles.
- `../LexiSignVQA-main/src/eval/sub_task_2.py`: reuse accuracy, question-type
  breakdown, answer distribution, and NFC normalization.

### Implementation Notes

- Retrieval metric input should be lists of citations, not raw text.
- QA metric input should compare normalized `answer` vs `prediction.answer`.
- Count invalid predictions separately instead of converting them to `A` or
  `Đúng`.
- Artifact should store config name, seed, split path/hash, timestamp, latency
  summary, metrics, and failed/invalid sample IDs.

### Expected Output

`python -m src.evaluate --predictions data/outputs/dev_predictions.jsonl`
prints and saves metrics like:

```json
{
  "retrieval": {
    "precision": 0.0,
    "recall": 0.0,
    "f2": 0.0
  },
  "qa": {
    "accuracy": 0.0,
    "by_question_type": {}
  },
  "invalid_predictions": []
}
```

### Acceptance Criteria

- [ ] Retrieval Precision/Recall/F2 are macro-averaged over samples.
- [ ] Empty predictions and empty gold citations are handled explicitly.
- [ ] QA accuracy is reported overall and per question type.
- [ ] Invalid/malformed predictions are counted and listed by sample ID.
- [ ] Evaluation can run on a tiny synthetic file in CI.
- [ ] `docs/experiments.md` documents B0, text-RAG, and later ablation naming.

### Tests

- Unit test perfect retrieval returns P/R/F2 = 1.
- Unit test partial retrieval matches the F2 formula.
- Unit test empty predicted citations returns zero score without crashing.
- Unit test answer normalization handles decomposed Vietnamese accents.
- Unit test invalid prediction is counted, not silently corrected.

### Verification

```bash
python -m src.evaluate --help
python -m pytest tests/test_evaluate.py -q
python -m src.evaluate --predictions tests/fixtures/tiny_predictions.jsonl
```

## W1-06 - Add Structured Prompting And VLM Output Parser

Labels: `week-1`, `P0`, `vlm`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M3` / `M2`

Branch: `feat/w1-06-vlm-parser`

PR title: `feat(vlm): add structured legal QA prompt and parser`

Depends on: `W1-02`, `W1-04`

### Description

Create the prompt and parser for the legal QA answer format. This issue does
not need to achieve high model accuracy yet. Its job is to guarantee that model
responses can be converted into `Prediction` and validated against retrieved
`Evidence`.

### Project Files To Change

- `src/prompts.py`
- `src/vlm.py`
- `configs/config.yaml`
- `tests/test_vlm_output.py` (new)

### Reference Files To Study

- `../LexiSignVQA-main/src/prompts/answer_prompt.py`: learn the idea of forcing
  a final answer tag, but replace it with stricter JSON output.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn image loading,
  question-type branching, answer normalization, and final-answer extraction.
- `../a-low-cost-approach-to-MLQA-TSR-main-2/notebooks/vlm_answer.ipynb`:
  learn how retrieved examples/evidence are placed into a VLM message, but do
  not copy its expensive few-shot setup into week 1.

### Implementation Notes

- Prompt must include image, question, choices if present, and top-k legal
  evidence from retrieval.
- Require model output in JSON with exactly these top-level fields:
  `answer`, `citations`, `explanation`, `confidence`, `abstained`.
- For benchmark questions, `answer` must be `A/B/C/D` or `Đúng/Sai`.
- Citations must refer only to retrieved `Evidence`.
- Do not store unrestricted chain-of-thought. Store a short explanation only.
- If the model cannot answer from evidence, return `abstained=true` and a
  short reason.

### Expected Output

Example parsed prediction:

```json
{
  "answer": "B",
  "citations": [
    {
      "law_id": "QCVN 41:2024/BGTVT",
      "article_id": "22"
    }
  ],
  "explanation": "Biển báo có khung thời gian cấm; ngoài khung giờ đó phương tiện được phép lưu thông.",
  "confidence": 0.72,
  "abstained": false
}
```

### Acceptance Criteria

- [ ] Prompt builder supports Multiple choice, Yes/No, and free-form demo mode.
- [ ] Parser accepts valid JSON even when wrapped in markdown code fences.
- [ ] Parser rejects unknown answers for the question type.
- [ ] Parser rejects citations outside retrieved evidence through
  `PipelineResult` or equivalent validation.
- [ ] Offline tests do not require model weights.
- [ ] Runtime wrapper can be initialized from config but is mockable in tests.

### Tests

- Unit test prompt contains image placeholder, question, choices, and evidence.
- Unit test valid JSON parses to `Prediction`.
- Unit test markdown-fenced JSON parses correctly.
- Unit test hallucinated citation fails validation.
- Unit test invalid answer `E` fails for multiple-choice questions.

### Verification

```bash
python -m pytest tests/test_vlm_output.py tests/test_schemas.py -q
python -m src.vlm --help
```

## W1-07 - Integrate Text-RAG Pipeline Smoke Run And Weekly Report

Labels: `week-1`, `P0`, `demo`, `evaluation`, `documentation`

Milestone: `W1 - Trusted baseline`

Owner/reviewer: `M4` / `M3`

Branch: `release/w1-trusted-baseline`

PR title: `release: integrate week 1 trusted baseline`

Depends on: `W1-01`, `W1-02`, `W1-03`, `W1-04`, `W1-05`, `W1-06`

### Description

Connect the week 1 components into a minimal end-to-end smoke pipeline:
sample -> text retrieval -> prompt/model interface -> validated prediction ->
evaluation artifact. This is the release issue for the first weekly report.

### Project Files To Change

- `src/pipeline.py`
- `scripts/preprocess.sh`
- `scripts/index.sh`
- `scripts/evaluate.sh`
- `Makefile`
- `README.md`
- `docs/report.md`
- `docs/experiments.md`
- `app/streamlit_app.py` only for a placeholder that explains demo status

### Reference Files To Study

- `../LexiSignVQA-main/src/core/sub_task_1.py`: learn the orchestration style
  of retrieval stages over train/public/private sets.
- `../LexiSignVQA-main/src/core/sub_task_2.py`: learn the answer-generation
  orchestration style and path handling.
- `../LexiSignVQA-main/src/ui/inspect_subtask1.py`: learn how to present
  images and retrieved articles later; do not build the full UI this week.

### Implementation Notes

- Use dependency injection or simple class/function parameters so tests can
  run with fake retriever and fake VLM.
- Add a `--limit` argument for smoke runs, for example 5 samples.
- Save outputs to `data/outputs/` with a clear run name.
- Do not make the pipeline return a fake default answer when a component
  fails. Surface the error and count it as invalid.
- Streamlit in week 1 may be a status page or skeleton; the full demo comes
  after the pipeline is stable.

### Expected Output

One command can run a tiny smoke pass and produce:

- retrieval results
- parsed predictions
- metrics artifact
- week 1 report summary

Example command:

```bash
python -m src.pipeline --mode benchmark --split val --limit 5 --output data/outputs/w1_smoke.jsonl
```

### Acceptance Criteria

- [ ] `make preprocess` produces LawDB article JSONL.
- [ ] `make index` creates the Qdrant text index.
- [ ] A 5-sample smoke run produces `PipelineResult`-compatible records.
- [ ] `make eval` can evaluate the smoke output.
- [ ] Weekly report records completed issues, commands, dataset audit,
  current metrics, known blockers, and next-week plan.
- [ ] No raw VLSP data, model weights, caches, or Qdrant local storage are
  committed.

### Tests

- Unit test pipeline with fake retriever and fake VLM.
- Unit test missing Qdrant/index/model config yields a helpful error.
- Unit test output JSONL round-trips through `PipelineResult`.
- Manual smoke test on 5 validation samples.

### Verification

```bash
make preprocess
make qdrant-up
make index
python -m src.pipeline --mode benchmark --split val --limit 5 --output data/outputs/w1_smoke.jsonl
python -m src.evaluate --predictions data/outputs/w1_smoke.jsonl
python -m pytest tests/test_pipeline.py -q
```

## Suggested Parallel Work Order

```text
W1-01
  -> W1-02 -> W1-03
          \-> W1-04
W1-03 --------> W1-05
W1-04 --------> W1-06
W1-01..W1-06 -> W1-07
```

- `M1` starts W1-02, then W1-03.
- `M2` starts reading Qdrant/reference retrieval code while waiting for W1-02,
  then owns W1-04.
- `M3` starts W1-06 prompt/parser with fake evidence, then connects it to
  real retrieval after W1-04.
- `M4` starts W1-01 and W1-05, then owns W1-07 integration.

## Week 1 Definition Of Done

- LawDB is flattened correctly into 402 article rows.
- The 530 training samples are validated and split without image leakage.
- Qdrant text retrieval can return top-k legal evidence from a question.
- VLM output has a strict machine-readable answer/citation/explanation schema.
- Evaluation reports retrieval and QA metrics, including invalid predictions.
- A 5-sample smoke pipeline can run end to end with mocked or real VLM
  depending on available compute.
