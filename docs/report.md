# Week 2 Integration Report

## Scope

Week 2 integrates the retrieval and few-shot path needed for the four-week
prototype:

```text
preprocess LawDB
  -> build LawDB text index
  -> build train-example index
  -> run B2/B3/B4 benchmark configs
  -> evaluate retrieval and QA
  -> update weekly report
```

This release candidate keeps the agreed Tier A direction stable while adding
week-2 comparison hooks. The work follows the low-cost reference project for
retrieval and few-shot experiment structure, and LexiSignVQA for concise
LawDB/Qdrant orchestration, citation-grounded evaluation, and result review.

## Implemented

- Reproducible Makefile targets for LawDB preprocessing, LawDB indexing,
  train-example indexing, and week-2 benchmark smoke runs.
- Qdrant LawDB text index over 402 processed legal articles.
- Qdrant train-example index over 421 train split examples.
- Benchmark configs for B2 Text-RAG, B3 fused retrieval, and B4 retrieved
  few-shot prompting.
- Metrics artifacts with retrieval Precision/Recall/F2, QA accuracy, invalid
  prediction count, latency summary, config name, seed, and split hash.
- Pipeline integration tests with fake retriever and fake VLM so CI does not
  need Docker, GPU, model downloads, or raw VLSP data.

## Week 2 Smoke Metrics

The following table is from a five-sample validation smoke run on
2026-06-27. All rows use `experiment.mock: true`; therefore QA accuracy here
is not final VLM quality. It only proves that retrieval, prompting, parsing,
artifact writing, and metric accounting are connected.

| Config | Retrieval | Prompt | Mock? | Samples | Retrieval P/R/F2 | QA Acc. | Invalid | Mean latency ms |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: |
| B2_text_rag | text | text_rag | yes | 5 | 0.0400 / 0.1000 / 0.0769 | 0.2000 | 0 | 3083.9 |
| B3_fused_rag | fusion | text_rag | yes | 5 | 0.2000 / 0.5333 / 0.3764 | 0.2000 | 0 | 4938.5 |
| B4_few_shot_rag | fusion | few_shot_rag | yes | 5 | 0.2000 / 0.5333 / 0.3764 | 0.2000 | 0 | 4069.5 |

Shared run metadata:

- Seed: `42`
- Split hash: `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448`
- Train split size: 421
- Validation split size: 109
- Metrics artifacts: `data/outputs/experiments/*_metrics.json`

## Commands Used

```bash
make preprocess
make qdrant-up
make index
python -m src.retrieval --mode index-examples --split train
make benchmark-week2-smoke
```

Equivalent individual benchmark commands:

```bash
make benchmark-b2
make benchmark-b3
make benchmark-b4
```

## Blockers And Notes

- Real VLM inference is not connected in this release smoke. Any row marked
  `mock: true` must not be reported as final model accuracy.
- B3 and B4 require the train-example index. The first run may download or load
  embedding model weights locally, which makes latency higher than steady-state
  retrieval.
- The SigLIP tokenizer warning about token IDs appeared during local image
  embedding initialization. It did not stop indexing or benchmarking.
- Local shell sessions can accidentally use Anaconda instead of `.venv`; use
  `PATH="$PWD/.venv/bin:$PATH"` or activate `.venv` before verification.

## Member Contributions

- M1: LawDB parsing, sample validation, grouped split checks, citation sanity.
- M2: Qdrant retrieval, text/image embeddings, example index, fusion retrieval.
- M3: structured legal QA prompt, output parser, few-shot prompt review.
- M4: evaluation artifacts, integration Makefile targets, weekly report, smoke
  benchmark verification.

## Next Steps

- Connect a real VLM backend and rerun B2/B3/B4 on the same validation split.
- Use the Streamlit evidence inspector to review failed retrieval cases.
- Run a larger validation benchmark after smoke checks are stable.
- Decide whether OCR, sign crop, or QLoRA are worth adding only after the real
  VLM comparison is available.
