# Week 3 Integration Report

## Scope

Week 3 integrated the training and controlled-experiment path around the
retrieval-grounded legal QA prototype:

```text
validated LawDB + grouped split
  -> frozen retrieval config
  -> structured prompting and real-backend artifact contract
  -> QLoRA diagnostic training
  -> week-4 product and reporting direction
```

The release follows the low-cost reference project for retrieval and few-shot
ablation framing, and LexiSignVQA for concise reporting of retrieval
Precision/Recall/F2, QA accuracy, question-type breakdowns, and
citation-grounded evaluation.

## Implemented

- Leakage-safe SFT JSONL records and a processor-aware collator.
- Reproducible QLoRA trainer with dry-run validation, metadata writing, LoRA
  parameter accounting, and checkpoint safety under ignored `checkpoints/`.
- OpenAI-compatible real VLM backend wiring with `mock=false` configs and
  explicit error rows instead of fake fallback answers.
- Structured legal reasoning prompt variant with short `Observation`,
  `Legal basis`, and `Conclusion` explanation style.
- Frozen retrieval config `retrieval-final-v1` using fused legal evidence.
- Metrics artifact extension for backend/model, `max_new_tokens`, parse
  success, invalid JSON, truncation count, latency, retrieval metrics, and QA
  metrics.
- QLoRA checkpoint card in `docs/checkpoint-card.md`.

## Environment And Commands

Local validation environment:

- Python: 3.11 in `.venv`
- Qdrant: Docker Compose service for retrieval benchmarks
- QLoRA diagnostic GPU: RTX 3090 24GB
- QLoRA dtype: `bfloat16`
- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`

Core verification:

```bash
make verify
python -m src.train_qlora --config configs/qlora.yaml --dry-run
python -m src.evaluate --help
git diff --check
```

W3 real-baseline command, when an OpenAI-compatible endpoint is available:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export OPENAI_COMPATIBLE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

make qdrant-up
make preprocess
make index
make index-examples
SMOKE_LIMIT=5 make benchmark-w3-real
```

QLoRA diagnostic commands:

```bash
python -m src.data_utils --mode build-sft
python -m src.train_qlora --config configs/qlora.yaml --dry-run
python -m src.train_qlora --config configs/qlora.yaml --max-samples 20
python -m src.train_qlora --config configs/qlora.yaml --max-samples 80
```

## Metrics

### Mock Smoke Metrics

These rows are from local JSON artifacts under `data/outputs/experiments/`.
They use `experiment.mock: true`, so QA accuracy is not model quality. The rows
only prove retrieval, artifact writing, and evaluator accounting.

| Config | Mock? | Samples | Retrieval P/R/F2 | QA Acc. | Invalid | Mean latency ms | Source |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `w2_b2_text_rag` | yes | 5 | 0.0400 / 0.1000 / 0.0769 | 0.2000 | 0 | 3083.9 | `w2_b2_text_rag_metrics.json` |
| `w2_b3_fused_rag` | yes | 5 | 0.2000 / 0.5333 / 0.3764 | 0.2000 | 0 | 4938.5 | `w2_b3_fused_rag_metrics.json` |
| `w2_b4_few_shot_rag` | yes | 5 | 0.2000 / 0.5333 / 0.3764 | 0.2000 | 0 | 4069.5 | `w2_b4_few_shot_rag_metrics.json` |
| `retrieval_final` | yes | 5 | 0.2000 / 0.5333 / 0.3764 | 0.2000 | 0 | 4716.1 | `retrieval_final_metrics.json` |

Shared split metadata:

- Seed: `42`
- Train split size: `421`
- Validation split size: `109`
- Split hash:
  `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448`

### Real Base VLM Metrics

No real/base VLM metrics are reported in this release because the available
local artifacts do not include `w3_b2_text_rag_real_metrics.json` or
`w3_b5_structured_real_metrics.json`. The non-mock benchmark path is wired, but
live execution requires `OPENAI_COMPATIBLE_API_KEY` and
`OPENAI_COMPATIBLE_BASE_URL`.

This is the correct reporting state: do not copy week-2 mock QA accuracy into
the real-run table.

### QLoRA Diagnostic Evidence

Checkpoint metadata is summarized in `docs/checkpoint-card.md` and traced to
`checkpoints/qlora_adapter/adapter_metadata.json`.

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter path | `checkpoints/qlora_adapter` |
| Effective train count | `80` |
| Train/val count | `421 / 109` |
| Dataset hash | `939d6ebce062970c2e0f16a73579174084bf2f3b05d1c90b82dad053e4b67df7` |
| Split hash | `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448` |
| LoRA rank/alpha/dropout | `8 / 16 / 0.05` |
| Trainable parameters | `18,576,384` (`0.905%`) |
| Metadata SHA-256 | `3c75e149babddd56722c5b7e7f36175452b7e205fb877d1e88364102ee989db5` |

Observed diagnostic outcomes:

- 20-sample GPU smoke run succeeded.
- 80-sample QLoRA run succeeded and wrote adapter metadata.
- 300-sample run failed with CUDA OOM on RTX 3090 24GB.
- Validation smoke with `max_new_tokens=160` had many truncated JSON outputs.
- Validation smoke with `max_new_tokens=320` reached `1/3` exact match.

The 80-sample adapter is not submission-ready and should not be reported as
final validation accuracy.

## Blockers And Errors

- Real VLM baselines require a reachable OpenAI-compatible endpoint and
  credentials. Without them, the command fails early with a configuration
  error instead of producing fake answers.
- Qdrant must be running before retrieval and real benchmark commands.
- The 300-sample QLoRA run exceeded RTX 3090 24GB memory.
- Adapter validation outputs can truncate when `max_new_tokens` is too low.
- Current adapter inference is diagnostic only and is not integrated into the
  main benchmark runner.

## Member Contributions

- M1: leakage-safe split/SFT data validation, citation sanity, data audit.
- M2: frozen retrieval config, Qdrant retrieval, example/fusion retrieval
  review, hard-case analysis support.
- M3: structured legal reasoning prompt, real VLM backend wiring, QLoRA trainer
  and checkpoint metadata.
- M4: evaluation artifact contract, W3 experiment docs, report integration,
  checkpoint card, release verification.

## Week 4 Direction

Week 4 should prioritize final system quality over larger training:

1. Run real/base VLM smoke and, if possible, locked-validation benchmarks.
2. Extend error analysis with final retrieval and model failures.
3. Add submission/export converter and format validator.
4. Polish Streamlit demo in retrieval-only, cached-prediction, and optional
   live-VLM modes.
5. Prepare final report, slides, and defense assets.
6. Treat QLoRA as an experimental extension unless adapter inference becomes
   stable and clearly improves the locked validation metrics.
