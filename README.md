# Traffic Legal VLM

Multimodal Legal Question Answering for Traffic Sign Regulations.

## Project Goal

This project builds a retrieval-augmented Vision-Language Model system for answering legal questions based on traffic images and Vietnamese traffic regulations.

The four-week course prototype focuses on a trusted Tier A baseline:

```text
traffic image + question + choices
  -> text retrieval from question/choices
  -> top-k legal articles from Qdrant
  -> VLM receives image + question + legal evidence
  -> ANSWER + CITATION + EXPLANATION
  -> evaluation + Streamlit-ready output
```

Traffic-sign cropping, OCR, advanced image retrieval, production monitoring,
and broader QLoRA experiments are extensions after the baseline is stable.

## Main Components

- LawDB preprocessing
- Qdrant-backed text legal retrieval
- Optional image-based retrieval
- VLM prompting
- Optional QLoRA fine-tuning
- Benchmark evaluation
- Streamlit demo

## Requirements

- Python 3.11
- Docker with Docker Compose for local Qdrant experiments
- VLSP/LawDB raw data placed locally under `data/raw/`

Raw datasets, generated embeddings, Qdrant storage, model weights, and local
caches must stay out of Git.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
make setup
cp .env.example .env
```

The pinned Python dependencies are in `requirements.txt`. `bitsandbytes` is
restricted to Linux x86_64 so macOS/CPU setup remains installable for app,
retrieval, and tests.

## Data Placement

Place the raw VLSP files like this:

```text
data/raw/
  law_db/
    vlsp2025_law_new.json
    images.fld/
  train_data/
    vlsp_2025_train.json
    train_images/
  public_test/
    vlsp_2025_public_test_task1.json
    vlsp_2025_public_test_task2.json
    public_test_images/
```

Check local data paths with:

```bash
make check-data
```

## Week 1 Commands

W1-01 only guarantees setup, documentation, and lightweight tests. Later week-1
issues will make preprocessing, indexing, evaluation, and the smoke pipeline
fully functional.

```bash
make help
make setup
make ci-test
make test
make verify
```

For local Qdrant experiments:

```bash
make qdrant-up
make qdrant-down
```

## Final Streamlit Demo

Week 4 includes a final evidence-grounded Streamlit demo for defense. It can
run without a live GPU/API by using retrieval-only mode or cached prediction
display. Live VLM mode is optional and appears only when the configured backend
credentials are available.

Prepare local data and indexes:

```bash
make qdrant-up
python -m src.data_utils --mode preprocess
python -m src.data_utils --mode split
python -m src.retrieval --mode index
```

If you want to inspect fused retrieval or few-shot examples, also build the
training-example index:

```bash
python -m src.retrieval --mode index-examples --split train
```

Start the demo:

```bash
bash scripts/demo.sh
```

The app supports:

- retrieval-only evidence inspection;
- cached prediction display from JSONL artifacts under `data/outputs/`;
- optional live VLM answering when backend credentials are configured;
- curated success/failure samples for presentation;
- visible citations, evidence scores, answer, explanation, latency, and a
  research/legal disclaimer.

For cached prediction mode, point the sidebar to a JSONL artifact such as
`data/outputs/experiments/w4_structured_rag.jsonl`. If no live VLM backend is
configured, the demo stays in retrieval-only mode and shows a clear message.

## Week 2 Retrieval And Few-Shot Release

The week-2 release flow compares the locked validation split across the Text
RAG baseline, fused retrieval, and retrieved few-shot prompting. The current
configs use `experiment.mock: true`, so the numbers are pipeline smoke metrics,
not final VLM accuracy.

Build the required processed data and indexes:

```bash
make preprocess
make qdrant-up
make index
make index-examples
```

You can also build both week-2 indexes with one wrapper:

```bash
make index-week2
```

Run the smoke benchmarks with a five-sample limit:

```bash
make benchmark-b2
make benchmark-b3
make benchmark-b4
```

Or run the same three configs in sequence:

```bash
make benchmark-week2-smoke
```

The prediction JSONL and metrics artifacts are written under
`data/outputs/experiments/`. That directory is ignored by Git because it may
contain bulky generated outputs.

## Real VLM Backend

Week 3 non-mock benchmarks use an OpenAI-compatible chat-completions endpoint.
Credentials stay in `.env` or shell environment variables:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="replace-with-your-key"
export OPENAI_COMPATIBLE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
```

Run one real Text-RAG sample when the endpoint is available:

```bash
make qdrant-up
make preprocess
make index
python -m src.pipeline --mode benchmark \
  --config configs/experiments/w3_b2_text_rag_real.yaml \
  --limit 1
python -m src.evaluate \
  --predictions data/outputs/experiments/w3_b2_text_rag_real.jsonl
```

If credentials or the endpoint are missing, the pipeline fails with a clear
configuration error before contacting a model. If the model call itself fails,
the benchmark writes an invalid/error sample instead of inventing a fallback
answer.

For the week-3 controlled smoke matrix, build both indexes and run:

```bash
make qdrant-up
make preprocess
make index
make index-examples
SMOKE_LIMIT=5 make benchmark-w3-real
```

Generated prediction and metrics files are written to
`data/outputs/experiments/`, which is intentionally ignored by Git.

## QLoRA Diagnostic Checkpoint

Week 3 includes a reproducible QLoRA training path and a local diagnostic
adapter under `checkpoints/qlora_adapter/`. Checkpoints and adapter weights are
not committed.

Validate the training plan on CPU/macOS without loading model weights:

```bash
make qlora-dry-run
```

Run a small GPU-only smoke training job:

```bash
make qlora-smoke20
```

Run a local adapter diagnostic over a few validation samples:

```bash
SMOKE_LIMIT=5 make adapter-diagnostic
```

The current 80-sample adapter is diagnostic, not submission-ready. Its
checkpoint details and limitations are documented in
[`docs/checkpoint-card.md`](docs/checkpoint-card.md). The week-3 finding is to
prioritize real baseline validation, frozen retrieval, structured prompting,
submission/export, and demo polish before investing in larger QLoRA runs.

## Submission Converter

Week 4 includes a task-2 submission converter. It packages internal
`PipelineResult` JSONL predictions into the benchmark-style JSON where each
sample keeps its original `id` and receives an `answer`. By default citations
stay in the internal artifact and are not copied into the submission file.

Dry-run validation checks answer format and, when a public/private template is
available, required sample coverage:

```bash
python -m src.submission \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl \
  --set-name public_test \
  --dry-run
```

Write a submission JSON only after validation is clean:

```bash
python -m src.submission \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl \
  --set-name public_test \
  --output data/outputs/submissions/w4_public_submission.json
```

For local smoke checks on incomplete prediction files, use
`--allow-missing --dry-run`. Do not use missing-answer output as a real
submission. Generated submissions under `data/outputs/` are ignored by Git.

## Final Report And Defense Pack

Week 4 final documentation is organized for defense and reproducibility:

- [`docs/report.md`](docs/report.md): final course report with scope, dataset,
  architecture, experiments, limitations, ethics, member contributions, and
  four-month continuation plan.
- [`docs/experiments.md`](docs/experiments.md): experiment contract, metric
  source rules, final validation commands, and submission export gate.
- [`docs/error-analysis.md`](docs/error-analysis.md): 30 categorized retrieval
  and model hard cases.
- [`docs/checkpoint-card.md`](docs/checkpoint-card.md): QLoRA diagnostic
  checkpoint card and limitations.
- [`docs/model-card.md`](docs/model-card.md): system model card for the main
  retrieval-grounded prototype.
- [`docs/final-slides.md`](docs/final-slides.md): slide source for the final
  presentation.
- [`docs/assets/`](docs/assets/): screenshot/asset notes for the demo.

Before reporting any number, verify that it comes from a metrics JSON artifact
or a documented local GPU smoke log. Mock rows must stay labeled as smoke
checks, and QLoRA must stay labeled as diagnostic unless a comparable
locked-validation artifact proves otherwise.

Final smoke verification:

```bash
python -m src.evaluate --help
python -m src.submission --help
make verify
git diff --check
```

If plain `pytest` crashes on macOS because of a local `readline` issue, use
`make test` or `make ci-test`. Both commands inject a lightweight `readline`
shim before importing pytest.

## CI

GitHub Actions runs a lightweight CI workflow that:

- validates the issue template YAML;
- installs only schema/test dependencies;
- runs `tests/test_schemas.py`;
- checks whitespace with `git diff --check`.

The CI workflow intentionally does not download VLM weights, embedding models,
raw datasets, or GPU-only packages.

## Project Planning

The team execution schedule, issue breakdown, file ownership, reference map,
and pull request order are documented in
[`docs/4-week-execution-plan.md`](docs/4-week-execution-plan.md).

GitHub-ready week-1 issue bodies are documented in
[`docs/week1-issues.md`](docs/week1-issues.md).
