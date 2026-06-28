# Retrieval Error Analysis

This note freezes the week-3 retrieval review format for
`configs/experiments/retrieval_final.yaml`. It separates retrieval quality from
VLM answer quality so the final report does not confuse evidence failure with
reasoning failure.

## Frozen Retrieval Setup

- Config: `configs/experiments/retrieval_final.yaml`
- Version: `retrieval-final-v1`
- Locked split: `data/processed/val_split.jsonl`
- Strategy: direct LawDB text retrieval plus leakage-safe train-example
  citation voting
- Top-k evidence: `5`
- Retrieved examples: top `3`
- Example search mode: `fusion`
- Text/image example weights: `0.7 / 0.3`

OCR, cropped-sign detection, and detector-driven sign retrieval remain stretch
work. They should not become week-3 requirements unless a locked-validation
ablation shows a measurable retrieval gain.

## Error Categories

Use one primary category per case:

- `visual_ambiguity`: the legal target depends on recognizing a sign, panel,
  lane context, or vehicle class from the image.
- `weak_question_text`: the question text alone is too generic for reliable
  text retrieval.
- `missing_law_context`: the query needs a legal hierarchy or rule family that
  is not obvious from the words in the question.
- `similar_article_confusion`: retrieval returns nearby sign/article IDs but
  misses the exact gold citation.
- `annotation_mismatch`: the gold citation set is broader or narrower than the
  evidence a user-facing answer would naturally cite.
- `output_format_issue`: prediction/parser output is malformed or invalid. This
  is not a retrieval failure.

## Hard Cases

Rows marked `measured` use the available five-sample W2 fused smoke artifact
(`data/outputs/experiments/w2_b3_fused_rag.jsonl`). Rows marked `watchlist`
come from the same locked validation split and should be refreshed after
running the full `retrieval_final` benchmark with Qdrant available.
For compactness, bare article IDs such as `B.31` mean
`QCVN 41:2024/BGTVT#B.31`; citations from other laws keep the full `law#article`
form.

| # | Sample | Status | Gold citations | Retrieved citations | Category | Retrieval vs VLM note |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | `train_1` | measured | `QCVN 41:2024/BGTVT#22` | `F.10`, `22`, `41`, `B.7`, `73` | `similar_article_confusion` | Retrieval partially succeeds because gold `22` appears at rank 2, but rank 1 is a panel article. A VLM may over-focus on the highest ranked panel evidence. |
| 2 | `train_104` | measured | `QCVN 41:2024/BGTVT#36`, `E.14` | `36`, `E.14`, `64`, `E.15`, `70` | `weak_question_text` | Fusion fixes the weak text-only query and retrieves both gold citations in top 2. Treat this as a hard success case, not a VLM reasoning result. |
| 3 | `train_116` | measured | `28`, `C.3`, `C.27`, `C.45` | `22`, `B.25`, `E.5`, `C.36`, `C.15` | `visual_ambiguity` | The question asks which sign is not mentioned in the image; retrieval misses all gold danger/sign-definition articles. |
| 4 | `train_127` | measured | `28`, `C.3`, `C.27`, `C.45`, `22`, `B.24` | `22`, `B.2`, `B.37`, `B.38`, `54` | `annotation_mismatch` | Retrieval finds the general prohibition article `22` but misses several gold sign IDs. The broad gold set makes recall difficult. |
| 5 | `train_134` | measured | `22`, `B.30` | `22`, `B.31`, `B.39`, `B.38`, `36/2024/QH15#18` | `similar_article_confusion` | Retrieval finds the general prohibition article but confuses nearby stopping/parking sign IDs. |
| 6 | `train_135` | watchlist | `28`, `C.2` | pending full `retrieval_final` run | `visual_ambiguity` | Yes/No wording depends on recognizing the exact warning sign family from the image. |
| 7 | `train_136` | watchlist | `27`, `11` | pending full `retrieval_final` run | `missing_law_context` | The answer depends on group-level warning-sign rules, not only one sign label. |
| 8 | `train_138` | watchlist | `22`, `B.31` | pending full `retrieval_final` run | `weak_question_text` | “Đây là biển báo gì?” is too short; retrieval must lean heavily on visual/example signals. |
| 9 | `train_139` | watchlist | `22`, `B.3`, `41`, `B.27` | pending full `retrieval_final` run | `missing_law_context` | Time-window and vehicle-class restrictions require combining sign and auxiliary-panel rules. |
| 10 | `train_18` | watchlist | `26` | pending full `retrieval_final` run | `similar_article_confusion` | The wording may retrieve general prohibition content instead of the specific U-turn rule. |
| 11 | `train_187` | watchlist | `22`, `B.3`, `41`, `F.11` | pending full `retrieval_final` run | `visual_ambiguity` | Requires reading a restriction sign and a time/vehicle supplementary panel together. |
| 12 | `train_188` | watchlist | `41`, `F.11` | pending full `retrieval_final` run | `similar_article_confusion` | Rectangular blue/white sign descriptions can be confused with nearby auxiliary-sign articles. |
| 13 | `train_189` | watchlist | `36/2024/QH15#11` | pending full `retrieval_final` run | `missing_law_context` | This is a legal-priority question about traffic-controller orders, not a traffic-sign definition. |
| 14 | `train_190` | watchlist | `41`, `F.10` | pending full `retrieval_final` run | `similar_article_confusion` | Similar auxiliary-panel articles `F.10` and `F.11` are easy to swap. |
| 15 | `train_191` | watchlist | `41`, `F.11` | pending full `retrieval_final` run | `visual_ambiguity` | Yes/No answer requires matching the stated time range against the visual panel. |

## Reporting Rules

- Report retrieval Precision/Recall/F2 from `retrieval_final` separately from
  QA accuracy.
- Mock QA accuracy is only a pipeline smoke signal. Do not describe it as real
  model quality.
- If gold evidence is missing from top-k, classify the case as a retrieval
  failure before blaming the VLM.
- If gold evidence is present but the answer is wrong, classify it as a VLM
  reasoning or prompt-output failure.
- Keep the case table updated with retrieved citations after every full
  validation run.
