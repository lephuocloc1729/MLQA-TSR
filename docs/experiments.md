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
- prediction path and file hash
- split manifest path, split hash, and split counts when available
- timestamp
- latency summary
- retrieval and QA metrics
- failed or invalid sample IDs

## Naming

- `B0`: schema/data sanity baseline. Use tiny or oracle-style predictions to
  prove the evaluator, split, and artifact contract work.
- `text-rag`: Tier A text retrieval baseline. Retrieve top-k LawDB articles
  from the question and choices, then answer with the VLM using those articles.
- `vlm-zero-shot`: VLM answer without LawDB retrieval. This is useful as a
  comparison, not as the trusted week-1 baseline.
- `text-rag-k{K}`: same as `text-rag` with a different retrieval `top_k`.
- `fusion-rag`: later ablation with text retrieval plus image/sign evidence.
- `qlora-{model}`: later fine-tuning experiment. Always compare against the
  matching base VLM and the same split.

For week 1, report `B0` and the first `text-rag` run if predictions are
available. Keep image retrieval, sign crop, and QLoRA as later ablations.
