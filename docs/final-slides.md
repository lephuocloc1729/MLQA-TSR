# Final Defense Slides Source

This file is the slide source for the week-4 defense deck. It is intentionally
plain Markdown so the team can paste it into PowerPoint, Google Slides, or
Canva without changing the technical story.

## Slide 1 - Title

**Multimodal Legal Question Answering for Traffic Sign Regulations**

Team: M1, M2, M3, M4

Key message: We built a citation-grounded prototype that connects traffic
images, user questions, legal retrieval, structured VLM prompting, evaluation,
and demo/export tooling.

## Slide 2 - Problem

Citizens often need to interpret traffic signs in a concrete visual scene, but
the law is written as dense text. A text-only legal search engine cannot answer
image-grounded questions reliably.

Visual idea: user uploads dashcam image and asks whether a maneuver is allowed.

## Slide 3 - Scope And Honesty

Main product: retrieval-grounded structured prompting.

Experimental extension: QLoRA diagnostic adapter.

Do not claim: QLoRA improves final validation or mock QA equals real model
accuracy.

Source: `docs/report.md`, `docs/checkpoint-card.md`.

## Slide 4 - Data

- LawDB flattened to 402 legal article rows.
- Train split: 421 samples.
- Validation split: 109 samples.
- Split is grouped by `image_id`.
- Split hash:
  `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448`.

Source: `data/processed/law_articles.jsonl`,
`data/processed/split_manifest.json`.

## Slide 5 - Architecture

```text
image + question + choices
  -> Query schema
  -> Qdrant LawDB retrieval
  -> optional example retrieval/fusion
  -> structured VLM prompt
  -> JSON Prediction
  -> evaluation + demo + submission converter
```

Visual idea: one pipeline diagram with citations flowing from Qdrant to answer.

## Slide 6 - Retrieval

Tier A retrieval indexes LawDB articles and returns top-k legal evidence. Week
2/3 added train-example retrieval and citation fusion.

Frozen config: `retrieval-final-v1`.

Observed smoke retrieval F2:

- text-only: `0.0769`;
- fused retrieval: `0.3764`;
- source: `data/outputs/experiments/w2_b2_text_rag_metrics.json` and
  `data/outputs/experiments/w2_b3_fused_rag_metrics.json`.

Speaker note: These are 5-sample mock smoke artifacts, not full validation.

## Slide 7 - Prompting And Output Contract

The VLM must output JSON:

```json
{
  "answer": "B",
  "citations": [{"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}],
  "explanation": "Observation: ... Legal basis: ... Conclusion: ...",
  "confidence": 0.72,
  "abstained": false
}
```

Benefits: parseable answers, citation validation, invalid-output accounting,
and safe demo display without hidden chain-of-thought.

## Slide 8 - Evaluation

Retrieval:

- macro Precision, Recall, F2 over `law_id#article_id`.

QA:

- exact-match answer accuracy;
- per question type;
- invalid JSON, truncation, unsupported citations counted separately.

Source: `src/evaluate.py`, `docs/experiments.md`.

## Slide 9 - Experiment Table

| Row | Status | Retrieval F2 | QA |
| --- | --- | ---: | --- |
| Text RAG smoke | `mock=true`, 5 samples | `0.0769` | smoke only |
| Fused RAG smoke | `mock=true`, 5 samples | `0.3764` | smoke only |
| Structured real RAG | config ready | pending | pending backend |
| QLoRA adapter | diagnostic | not comparable | `1/3` small smoke only |

Speaker note: Every number must point to an artifact; pending rows stay pending.

## Slide 10 - QLoRA Diagnostic

- Base model: `Qwen/Qwen2.5-VL-3B-Instruct`.
- Effective train count: 80.
- GPU: RTX 3090 24GB.
- LoRA rank/alpha/dropout: `8 / 16 / 0.05`.
- Trainable parameters: `18,576,384` (`0.905%`).
- 300-sample run failed with CUDA OOM.
- Current adapter is not the final submission model.

Source: `docs/checkpoint-card.md`,
`checkpoints/qlora_adapter/adapter_metadata.json`.

## Slide 11 - Demo

Demo modes:

- retrieval-only;
- cached prediction;
- mock smoke;
- optional live VLM.

Show: image, question, choices, evidence, citations, answer, explanation,
latency, disclaimer.

Source: `app/streamlit_app.py`, `scripts/demo.sh`, `docs/assets/`.

Backup screenshot: `docs/assets/final-demo-retrieval-only.png`.

## Slide 12 - Error Analysis

30 cases are categorized into:

- retrieval;
- visual ambiguity;
- legal context;
- reasoning;
- output format;
- annotation/data;
- adapter truncation.

Source: `docs/error-analysis.md`.

## Slide 13 - Limitations

- Not official legal advice.
- Possible visual misinterpretation.
- Retrieval can miss or confuse similar articles.
- JSON/citation formatting can fail.
- QLoRA adapter is diagnostic only.
- Real VLM metrics require a configured backend.

## Slide 14 - Four-Month Continuation

1. Run full locked-validation real VLM baselines.
2. Improve retrieval using error analysis.
3. Add OCR/sign crop only after measured ablation.
4. Strengthen JSON/citation robustness.
5. Train larger adapters after baseline is stable.
6. Polish demo and final submission workflow.

## Slide 15 - Closing

We completed a reproducible multimodal legal QA prototype with clear evidence
retrieval, structured answer validation, demo/export tooling, and a realistic
path to the larger graduation project.
