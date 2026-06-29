# Final Course Report

## Project

**English name:** Multimodal Legal Question Answering for Traffic Sign
Regulations

**Vietnamese name:** Trả lời câu hỏi pháp lý đa phương thức cho quy định biển
báo giao thông

The course prototype builds a retrieval-grounded legal QA system for Vietnamese
traffic-sign regulations. Given a traffic image, a user question, and optional
multiple-choice answers, the system retrieves relevant legal articles and asks a
Vision-Language Model (VLM) to return a structured answer with citation and a
short explanation.

This final report keeps the evidence story honest:

- the main course product is retrieval-grounded structured prompting plus
  evaluation, submission conversion, and a Streamlit demo;
- week-2 mock metrics are engineering smoke checks, not model quality;
- real/base VLM runs require an OpenAI-compatible endpoint or hosted model;
- QLoRA is a diagnostic extension with real GPU evidence, but the current
  80-sample adapter is not submission-ready.

## Problem Statement

Traffic regulations are difficult for non-experts because the correct answer
often depends on both visual context and legal text. A standard text-only legal
search engine cannot directly answer a question such as whether a driver may
park, turn, or enter a road segment shown in an uploaded image. The project
therefore combines:

- visual input from the traffic scene;
- question and answer-choice text;
- retrieved legal evidence from LawDB;
- a VLM constrained to answer in a citation-aware JSON format.

The system is intended for research and education assistance only. It is not an
official legal-advice system.

## Dataset And Legal Corpus

The local data pipeline prepares two resources:

| Resource | Output | Current local count | Purpose |
| --- | --- | ---: | --- |
| LawDB legal corpus | `data/processed/law_articles.jsonl` | 402 article rows | Qdrant legal evidence retrieval and citation validation |
| VLSP train split | `data/processed/train_split.jsonl` | 421 samples | train examples, SFT records, few-shot examples |
| VLSP validation split | `data/processed/val_split.jsonl` | 109 samples | locked validation and error analysis |

The grouped split is leakage-safe by `image_id`: training and validation images
do not overlap. The split uses seed `42` and split hash
`3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448` from
`data/processed/split_manifest.json`.

## Reference Projects

The implementation was designed around two local reference projects:

- `a-low-cost-approach-to-MLQA-TSR-main-2`: low-cost retrieval, vector-search
  framing, and ablation style.
- `LexiSignVQA-main`: LawDB preprocessing style, citation-aware evaluation,
  question-type normalization, and compact experiment reporting.

The course project adapts those ideas into a smaller, reproducible four-week
prototype instead of copying their full pipeline.

## System Architecture

```text
traffic image + question + choices
  -> validated Query schema
  -> LawDB article retrieval from Qdrant
  -> optional train-example retrieval/fusion
  -> structured legal QA prompt
  -> VLM or mock client
  -> Prediction JSON: answer, citations, explanation, confidence, abstained
  -> PipelineResult JSONL
  -> evaluation, submission conversion, Streamlit demo
```

The final product path is `w4_structured_rag`: frozen fused retrieval plus the
structured legal-reasoning prompt. When a real VLM backend is unavailable, the
same app and evaluator can still run in retrieval-only or cached-prediction
mode.

## Implemented Components

- LawDB flattening into article-level records with images/tables preserved as
  structured lists.
- Deterministic grouped train/validation split with answer normalization and
  citation validation.
- Qdrant legal article retrieval and train-example retrieval.
- Cached text and whole-image embedding adapters with fake-injection tests.
- Prompt variants: zero-shot, text RAG, few-shot RAG, and structured legal RAG.
- Strict JSON output parser for `Prediction`.
- Evaluation metrics for retrieval Precision/Recall/F2 and QA accuracy.
- Real VLM backend wrapper for OpenAI-compatible endpoints.
- Diagnostic QLoRA SFT data, collator, trainer, and adapter inference runner.
- Submission converter and format validator.
- Final Streamlit demo with retrieval-only, cached prediction, mock, and live
  modes.

## Evaluation Metrics

Retrieval is evaluated by macro-averaged per-sample Precision, Recall, and F2
over citation UIDs in the format `law_id#article_id`.

QA is evaluated by exact-match accuracy after NFC normalization. Invalid
answers, invalid JSON, truncated output, missing citations, and unsupported
citations are tracked separately and counted as incorrect. Predictions are not
silently converted into `A` or `Đúng`.

The full artifact contract is documented in `docs/experiments.md`.

## Experiment Results

### Available Smoke Metrics

The following metrics are from local JSON artifacts under
`data/outputs/experiments/`. They are useful engineering checks, but they use
`experiment.mock: true`, so QA accuracy is not real VLM quality.

| Config | Mock? | Samples | Retrieval P/R/F2 | QA Acc. | Invalid | Mean latency ms | Source artifact |
| --- | --- | ---: | --- | ---: | ---: | ---: | --- |
| `w2_b2_text_rag` | yes | 5 | `0.0400 / 0.1000 / 0.0769` | `0.2000` | 0 | 3083.9 | `data/outputs/experiments/w2_b2_text_rag_metrics.json` |
| `w2_b3_fused_rag` | yes | 5 | `0.2000 / 0.5333 / 0.3764` | `0.2000` | 0 | 4938.5 | `data/outputs/experiments/w2_b3_fused_rag_metrics.json` |
| `w2_b4_few_shot_rag` | yes | 5 | `0.2000 / 0.5333 / 0.3764` | `0.2000` | 0 | 4069.5 | `data/outputs/experiments/w2_b4_few_shot_rag_metrics.json` |
| `retrieval_final` | yes | 5 | `0.2000 / 0.5333 / 0.3764` | `0.2000` | 0 | 4716.1 | `data/outputs/experiments/retrieval_final_metrics.json` |

### Final Validation Status

| Row | Role | Artifact status | Reported status |
| --- | --- | --- | --- |
| Retrieval-only evidence | Inspect frozen retrieval quality | `retrieval_final_metrics.json` exists as a 5-sample smoke equivalent | Report retrieval smoke only |
| Base VLM + text RAG | Real direct-retrieval baseline | `w4_text_rag_real_metrics.json` not present locally | Pending real backend |
| Base VLM + structured legal RAG | Main product candidate | `w4_structured_rag_metrics.json` not present locally | Pending real backend |
| QLoRA adapter diagnostic | PEFT diagnostic row | GPU smoke JSONL exists locally under ignored `data/outputs/gpu_smoke/` | Diagnostic only |

No real/base VLM validation accuracy is claimed in this report because the local
repo state does not contain real-backend metrics JSON. The non-mock pipeline is
wired, but live execution requires `OPENAI_COMPATIBLE_BASE_URL` and
`OPENAI_COMPATIBLE_API_KEY`.

## QLoRA Diagnostic

The QLoRA path is real and reproducible, but it is not the main final product.
Checkpoint details are summarized in `docs/checkpoint-card.md`.

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter path | `checkpoints/qlora_adapter` |
| Effective train count | 80 |
| GPU | RTX 3090 24GB |
| dtype | `bfloat16` |
| LoRA rank/alpha/dropout | `8 / 16 / 0.05` |
| Trainable parameters | `18,576,384` (`0.905%`) |
| Dataset hash | `939d6ebce062970c2e0f16a73579174084bf2f3b05d1c90b82dad053e4b67df7` |
| Split hash | `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448` |

Diagnostic observations:

- 20-sample GPU smoke training succeeded.
- 80-sample QLoRA training succeeded and wrote adapter metadata.
- 300-sample training failed with CUDA OOM on RTX 3090 24GB.
- 5-sample adapter validation smoke with low token budget produced truncated
  JSON-like outputs.
- 3-sample adapter validation smoke with `max_new_tokens=320` reached `1/3`
  exact match.

These are integration diagnostics, not leaderboard-quality validation metrics.

## Demo

The final Streamlit demo supports:

- retrieval-only evidence inspection;
- cached prediction display from JSONL artifacts;
- mock smoke prediction for offline presentations;
- live VLM calls only when a backend is configured.

The demo shows the image, question, choices, retrieved evidence, citation IDs,
scores, answer, explanation, latency, output mode, and a clear research/legal
disclaimer.

Demo asset notes and screenshot instructions are stored under `docs/assets/`.
The current backup screenshot is
`docs/assets/final-demo-retrieval-only.png`.

## Reproducibility Pack

Use the project virtual environment for trusted validation:

```bash
python -m venv .venv
source .venv/bin/activate
make setup
```

Prepare data and indexes:

```bash
make qdrant-up
make preprocess
python -m src.data_utils --mode split
make index
make index-examples
```

Run final structured RAG when a real backend is available:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export OPENAI_COMPATIBLE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"

python -m src.pipeline --mode benchmark \
  --config configs/experiments/w4_structured_rag.yaml

python -m src.evaluate \
  --config configs/experiments/w4_structured_rag.yaml \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl
```

Validate and convert a submission after metrics review:

```bash
python -m src.submission \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl \
  --set-name public_test \
  --dry-run

python -m src.submission \
  --predictions data/outputs/experiments/w4_structured_rag.jsonl \
  --set-name public_test \
  --output data/outputs/submissions/w4_public_submission.json
```

Run the demo:

```bash
bash scripts/demo.sh
```

Run project verification:

```bash
python -m src.evaluate --help
python -m src.submission --help
make verify
git diff --check
```

## Limitations And Ethics

- The system is not official legal advice.
- The VLM can misread traffic signs, supplementary panels, lane context, speed
  limits, or vehicle classes.
- Retrieval can miss the correct article or confuse visually/legal-similar
  sign IDs.
- JSON generation can fail, truncate, or omit required citations.
- The QLoRA adapter is not submission-ready and should not be used as the final
  model.
- Public demos should avoid exposing secrets, raw private-test predictions, or
  hidden chain-of-thought.

## Member Contributions

- M1: data preprocessing, grouped split, SFT data checks, citation sanity, and
  submission format validation.
- M2: Qdrant retrieval, example retrieval/fusion, final demo review, and
  retrieval hard-case analysis.
- M3: structured prompts, VLM backend wrapper, QLoRA trainer, adapter
  diagnostics, and final slide narrative.
- M4: evaluation artifact contract, experiment tables, final report
  integration, documentation review, and reproducibility verification.

## Four-Month Continuation Plan

1. Run real/base VLM validation on the full locked split and archive metrics
   artifacts.
2. Use the error-analysis table to target the highest-impact retrieval
   failures.
3. Add OCR or sign-crop detection only after an ablation proves it improves the
   locked validation split.
4. Improve prompt robustness for JSON validity, citations, abstention, and
   concise explanations.
5. Train a larger QLoRA adapter only after the real baseline is stable and GPU
   memory planning is solved.
6. Polish the final web demo, export submission files safely, and prepare
   defense slides with metric-source traceability.
