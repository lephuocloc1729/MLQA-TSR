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
- prediction path and file hash
- split manifest path, split hash, and split counts when available
- timestamp
- latency summary
- retrieval and QA metrics
- failed or invalid sample IDs

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
