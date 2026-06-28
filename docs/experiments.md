# Experiment Tracking

Every experiment should write a JSONL prediction file and then run
`python -m src.evaluate --predictions <path>`. The saved metrics artifact is
the comparison contract for weekly reports and later ablations.

## Prediction JSONL Contract

Preferred rows use the project `PipelineResult` shape:

```json
{
  "query": {
    "id": "val_1",
    "image_id": "train_1_3",
    "question_type": "Multiple choice",
    "answer": "B",
    "relevant_articles": [{"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}]
  },
  "prediction": {
    "answer": "B",
    "citations": [{"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}],
    "explanation": "..."
  },
  "timings_ms": {"retrieval": 12.5, "generation": 130.0}
}
```

The evaluator also accepts legacy flat fields from the reference projects:
`answer`, `predict`, `relevant_articles`, and `predicted_articles`.

## Metrics

Retrieval is evaluated with macro-averaged per-sample Precision, Recall, and
F2 over citation UIDs in the format `law_id#article_id`. Empty predicted
citations are scored as zero when gold citations exist.

QA is evaluated with exact-match accuracy after NFC normalization. Gold labels
may use known VLSP legacy forms such as `40 -> A`, but predictions are not
repaired into valid labels. Invalid predictions are listed separately and count
as incorrect.

The metrics artifact stores:

- config name and seed
- experiment name, label, retrieval strategy, prompt variant, and `mock`
- model backend/name, `max_new_tokens`, and image-input flag
- retrieval config snapshot, including frozen config version when available
- prediction path and file hash
- split manifest path, split hash, and split counts when available
- timestamp
- latency summary
- parse success count, invalid JSON count, and truncated output count
- retrieval and QA metrics
- failed or invalid sample IDs
- adapter diagnostic metadata when an adapter run is explicitly configured

## Week 2 Ablation Matrix

Week 2 uses the locked validation split from
`data/processed/split_manifest.json`. Do not compare runs that use different
`val_split_path` or `split_hash` unless the config explicitly marks the run as
non-comparable.

All initial W2 configs are smoke-run ready with `experiment.mock: true`. This
means the pipeline uses a deterministic mock predictor to test retrieval,
prompt construction, artifact writing, and metric accounting. These runs are
valid engineering checks, but they are not final VLM accuracy.

| Config | Label | Retrieval | Prompt | Output |
| --- | --- | --- | --- | --- |
| `configs/experiments/w2_b1_zero_shot.yaml` | `B1_zero_shot` | none | `zero_shot` | `data/outputs/experiments/w2_b1_zero_shot.jsonl` |
| `configs/experiments/w2_b2_text_rag.yaml` | `B2_text_rag` | direct LawDB text top-5 | `text_rag` | `data/outputs/experiments/w2_b2_text_rag.jsonl` |
| `configs/experiments/w2_b3_fused_rag.yaml` | `B3_fused_rag` | direct LawDB + example citation fusion | `text_rag` | `data/outputs/experiments/w2_b3_fused_rag.jsonl` |
| `configs/experiments/w2_b4_few_shot_rag.yaml` | `B4_few_shot_rag` | fused evidence + top-3 examples | `few_shot_rag` | `data/outputs/experiments/w2_b4_few_shot_rag.jsonl` |

### Commands

Run one small smoke benchmark:

```bash
python -m src.pipeline --mode benchmark \
  --config configs/experiments/w2_b2_text_rag.yaml \
  --limit 5

python -m src.evaluate \
  --config configs/experiments/w2_b2_text_rag.yaml \
  --predictions data/outputs/experiments/w2_b2_text_rag.jsonl
```

Or use the helper:

```bash
scripts/evaluate.sh run-experiment configs/experiments/w2_b2_text_rag.yaml 5
```

For the week-2 release candidate, use the Makefile wrappers:

```bash
make preprocess
make qdrant-up
make index
make index-examples
make benchmark-b2
make benchmark-b3
make benchmark-b4
```

`make index-week2` can replace `make index` plus `make index-examples`.
`make benchmark-week2-smoke` runs B2, B3, and B4 in sequence with
`SMOKE_LIMIT ?= 5`.

Run the full W2 matrix without a limit only after Qdrant indexes are ready and
the model backend has enough compute budget:

```bash
for cfg in configs/experiments/w2_b*.yaml; do
  scripts/evaluate.sh run-experiment "$cfg"
done
```

Before running `B3_fused_rag` or `B4_few_shot_rag`, build the example index:

```bash
python -m src.retrieval --mode index-examples --split train
```

The metrics files are saved beside the predictions:

- `data/outputs/experiments/w2_b1_zero_shot_metrics.json`
- `data/outputs/experiments/w2_b2_text_rag_metrics.json`
- `data/outputs/experiments/w2_b3_fused_rag_metrics.json`
- `data/outputs/experiments/w2_b4_few_shot_rag_metrics.json`

### Week 2 Smoke Ablation

These values come from a five-sample smoke run on the locked validation split.
For mock runs, keep `Mock? = yes` and do not present QA accuracy as model
quality.

| Run | Mock? | Split hash | Top-k | Retrieval F2 | QA Accuracy | Invalid | Mean latency ms | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| B1_zero_shot | yes | `3ffba07c...` | 0 | not applicable | not run | not run | not run | no legal retrieval in W2 release smoke |
| B2_text_rag | yes | `3ffba07c...` | 5 | 0.0769 | 0.2000 | 0 | 3083.9 | week-1 direct LawDB retrieval |
| B3_fused_rag | yes | `3ffba07c...` | 5 | 0.3764 | 0.2000 | 0 | 4938.5 | direct retrieval plus example citation votes |
| B4_few_shot_rag | yes | `3ffba07c...` | 5 | 0.3764 | 0.2000 | 0 | 4069.5 | fused evidence plus top-3 retrieved examples |

The smoke run used seed `42`, `train_count=421`, `val_count=109`, and split
hash `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448`.
The QA predictor was deterministic mock logic, so the QA accuracy column only
confirms evaluator and artifact wiring.

## Week 3 Retrieval Freeze

The frozen retrieval config for week-3 and week-4 comparisons is:

- `configs/experiments/retrieval_final.yaml`
- version: `retrieval-final-v1`
- locked split: `data/processed/val_split.jsonl`
- retrieval strategy: direct LawDB text retrieval plus train-example citation
  fusion
- top-k legal evidence: `5`
- retrieved training examples: `3`
- example mode: `fusion`
- text/image example weights: `0.7 / 0.3`
- direct/example citation fusion weights: `1.0 / 1.0`

This config is intentionally close to `B3_fused_rag`. On the available
five-sample W2 smoke run, fused retrieval improved retrieval F2 from `0.0769`
for direct text-only retrieval to `0.3764` for fused retrieval on the same
locked split and seed. `B4_few_shot_rag` uses the same evidence stack, so it
does not change the retrieval freeze decision.

| Frozen config | Basis | Retrieval F2 observed | Decision |
| --- | --- | ---: | --- |
| `retrieval_final-v1` | `B3_fused_rag` smoke artifact | `0.3764` | Use for W3/W4 model comparisons until a larger locked-validation run proves a better setting. |

The QA values in the W2 table are mock smoke values. They must not be reported
as real model quality. Use them only to confirm that prediction artifacts,
latency accounting, and evaluation wiring work.

### Freeze Commands

Build or refresh the required indexes:

```bash
make qdrant-up
make preprocess
make index
make index-examples
```

Run the frozen retrieval smoke benchmark and evaluation:

```bash
python -m src.pipeline --mode benchmark \
  --config configs/experiments/retrieval_final.yaml \
  --limit 10

python -m src.evaluate \
  --config configs/experiments/retrieval_final.yaml \
  --predictions data/outputs/experiments/retrieval_final.jsonl
```

For a full validation pass, remove `--limit 10` only after the LawDB and
train-example Qdrant collections are indexed.

### Error Analysis

Use `docs/error-analysis.md` as the case-level review log. A case is a
retrieval failure when the gold citation is missing from top-k evidence. A
case is a VLM reasoning failure only when the required evidence is present but
the answer or explanation is wrong.

OCR, cropped-sign detection, and detector-driven sign retrieval remain
documented stretch work. Do not add them as required components unless a
locked-validation ablation shows measured improvement over
`retrieval_final-v1`.

## Week 3 Controlled Runs

Week 3 replaces week-2 mock smoke checks with real/base VLM baselines when an
OpenAI-compatible backend or hosted local model is available. Every real run
must use the locked validation split and must write `mock=false` in both the
prediction JSONL and metrics JSON.

| Config | Label | Retrieval | Prompt | Mock? | Output |
| --- | --- | --- | --- | --- | --- |
| `configs/experiments/w3_b2_text_rag_real.yaml` | `W3_B2_text_rag_real` | direct LawDB text top-5 | `text_rag` | no | `data/outputs/experiments/w3_b2_text_rag_real.jsonl` |
| `configs/experiments/w3_b5_structured_real.yaml` | `W3_B5_structured_real` | frozen fused retrieval top-5 | `structured_legal_rag` | no | `data/outputs/experiments/w3_b5_structured_real.jsonl` |

Before running these configs, set backend credentials in the environment:

```bash
export OPENAI_COMPATIBLE_BASE_URL="http://localhost:8000/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export OPENAI_COMPATIBLE_MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
```

Then run a controlled smoke pass:

```bash
make qdrant-up
make preprocess
make index
make index-examples
scripts/evaluate.sh run-w3-real 5
```

The W3 metrics table should include retrieval P/R/F2, QA accuracy,
`parse.parse_success_count`, `parse.invalid_json_count`,
`parse.truncated_output_count`, latency, backend/model, prompt variant, and
`max_new_tokens`. Do not copy week-2 mock QA values into the real-run table.

### QLoRA Diagnostic Smoke

The local QLoRA adapter is diagnostic evidence that PEFT training can run; it
is not the official submission model.

Checkpoint metadata summary from `checkpoints/qlora_adapter/adapter_metadata.json`
(ignored by Git, summarized here only):

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter path | `checkpoints/qlora_adapter` |
| Effective train count | `80` |
| Train/val counts | `421 / 109` |
| Split hash | `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448` |
| LoRA rank/alpha | `8 / 16` |
| Trainable parameters | `18,576,384` (`0.905%`) |
| Device/dtype | `cuda`, `bfloat16` |

Diagnostic evidence:

- 20-sample GPU smoke run succeeded.
- 80-sample QLoRA run succeeded and wrote adapter metadata.
- 300-sample QLoRA run failed with CUDA OOM on RTX 3090 24GB.
- Validation smoke with `max_new_tokens=160` produced many truncated outputs.
- Validation smoke with `max_new_tokens=320` reached `1/3` exact match.

Report these as training/integration diagnostics only. Do not report the
80-sample adapter as leaderboard-quality validation accuracy unless adapter
inference is fully integrated into the benchmark pipeline and evaluated on the
locked split with the same artifact contract.

### Week 3 Release Status

| Category | Artifact status | Reportable metric status |
| --- | --- | --- |
| Week-2 Text-RAG/Fusion/Few-shot smoke | Metrics JSON exists under `data/outputs/experiments/` | Report as `mock=true` engineering smoke only |
| `retrieval_final-v1` | `retrieval_final_metrics.json` exists locally | Report retrieval P/R/F2 as mock retrieval smoke, not VLM quality |
| Real base VLM `W3_B2_text_rag_real` | Config exists; metrics artifact not present in repo state | Do not report real QA accuracy until `w3_b2_text_rag_real_metrics.json` exists |
| Real structured VLM `W3_B5_structured_real` | Config exists; metrics artifact not present in repo state | Do not report structured-prompt QA accuracy until metrics JSON exists |
| QLoRA adapter | `adapter_metadata.json` exists locally under ignored `checkpoints/` | Report as diagnostic training evidence only |

The release report is `docs/report.md`. The checkpoint details are in
`docs/checkpoint-card.md`.

## Week 4 Adapter Diagnostic

`configs/experiments/w4_adapter_diag.yaml` defines the local QLoRA adapter
diagnostic run. It defaults to `checkpoints/qlora_adapter`, the locked
validation split, and `max_new_tokens=320` because lower token limits produced
truncated JSON during week-3 smoke checks.

```bash
python -m src.adapter_infer \
  --adapter checkpoints/qlora_adapter \
  --split val \
  --limit 5 \
  --max-new-tokens 320 \
  --output data/outputs/experiments/w4_adapter_diag.jsonl

python -m src.evaluate \
  --config configs/experiments/w4_adapter_diag.yaml \
  --predictions data/outputs/experiments/w4_adapter_diag.jsonl
```

The JSONL rows include adapter metadata hash, raw response, parsed answer,
exact match, parse status, truncation flag, unsupported-citation flag, latency,
target answer, and evidence. Treat this row as diagnostic unless it is evaluated
on the locked split and clearly improves the main base-VLM system.

## Week 4 Final Validation

Week 4 uses one locked validation split for every comparable row:
`data/processed/val_split.jsonl`, split hash
`3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448`.
Every reported number must come from a metrics JSON artifact; rows without an
artifact stay marked `pending`.

The main product variant is `w4_structured_rag`: base/real VLM, frozen fused
retrieval, and the structured legal-reasoning prompt. The retrieval-only row is
for evidence inspection and retrieval Precision/Recall/F2. The QLoRA row is
diagnostic only unless adapter inference is fully run on the locked split and
outperforms the main base-VLM system.

| Row | Config | Mock? | Role | Artifact source | Retrieval P/R/F2 | QA accuracy | Invalid / parse / truncated | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Retrieval-only evidence | `configs/experiments/w4_retrieval_only.yaml` | yes | frozen retrieval inspection | pending `data/outputs/experiments/w4_retrieval_only_metrics.json`; current equivalent smoke source: `data/outputs/experiments/retrieval_final_metrics.json` | current smoke: `0.2000 / 0.5333 / 0.3764` | current smoke mock only: `0.2000` | current smoke: invalid `0`; parse/truncated unavailable in older artifact | config ready; rerun full validation before final report |
| Base VLM + text RAG | `configs/experiments/w4_text_rag_real.yaml` | no | direct legal retrieval baseline | `data/outputs/experiments/w4_text_rag_real_metrics.json` | pending | pending | pending | requires OpenAI-compatible backend |
| Base VLM + structured legal RAG | `configs/experiments/w4_structured_rag.yaml` | no | main product candidate | `data/outputs/experiments/w4_structured_rag_metrics.json` | pending | pending | pending | requires OpenAI-compatible backend |
| QLoRA adapter diagnostic | `configs/experiments/w4_adapter_diag.yaml` | no | diagnostic PEFT row, non-final | `data/outputs/experiments/w4_adapter_diag_metrics.json` | oracle/evidence-dependent | pending | pending invalid JSON / truncation / unsupported citation counts | optional GPU/local adapter run |

### Final Validation Commands

Refresh the indexes once:

```bash
make qdrant-up
make preprocess
make index
make index-examples
```

Run retrieval-only evidence inspection:

```bash
python -m src.pipeline --mode benchmark \
  --config configs/experiments/w4_retrieval_only.yaml

python -m src.evaluate \
  --config configs/experiments/w4_retrieval_only.yaml \
  --predictions data/outputs/experiments/w4_retrieval_only.jsonl
```

Run the main structured legal RAG row when a real backend is available:

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

Run the adapter diagnostic only on a machine with the local checkpoint:

```bash
python -m src.adapter_infer \
  --adapter checkpoints/qlora_adapter \
  --split val \
  --max-new-tokens 320 \
  --output data/outputs/experiments/w4_adapter_diag.jsonl

python -m src.evaluate \
  --config configs/experiments/w4_adapter_diag.yaml \
  --predictions data/outputs/experiments/w4_adapter_diag.jsonl
```

For the final report, copy metrics only from the JSON files named in the table.
Do not turn pending rows into claimed accuracy. Invalid JSON, truncated output,
unsupported citations, and valid-but-wrong answers should be counted
separately.

## Naming

- `B0`: schema/data sanity baseline. Use tiny or oracle-style predictions to
  prove the evaluator, split, and artifact contract work.
- `B1_zero_shot`: image/question prompt without LawDB evidence.
- `B2_text_rag`: Tier A text retrieval baseline. Retrieve top-k LawDB articles
  from the question and choices, then answer with the VLM using those articles.
- `B3_fused_rag`: direct legal retrieval plus retrieved-example citation votes.
- `B4_few_shot_rag`: fused legal evidence plus top-3 solved retrieved examples.
- `text-rag-k{K}`: same as `B2_text_rag` with a different retrieval `top_k`.
- `qlora-{model}`: later fine-tuning experiment. Always compare against the
  matching base VLM and the same split.

For week 1, report `B0` and the first `text-rag` run if predictions are
available. Keep image retrieval, sign crop, and QLoRA as later ablations.
