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

## Streamlit Evidence Inspector

Week 2 includes a lightweight retrieval inspection demo. It is meant for
debugging and weekly reporting, not as the final polished app.

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
python -m streamlit run app/streamlit_app.py
```

The app supports selecting a validation sample, viewing the image/question,
retrieving top-k legal evidence, copying citation IDs, and optionally showing a
mock prediction panel. It does not require model/API credentials for retrieval
inspection. If no VLM backend is configured, it stays in retrieval-only mode and
shows a clear message.

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
