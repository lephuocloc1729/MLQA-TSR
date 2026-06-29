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

- `retrieval`: gold citations are missing from top-k, ranked too low, or
  confused with nearby articles/sign IDs.
- `visual_ambiguity`: the legal target depends on recognizing a sign, panel,
  lane context, vehicle class, or distance/speed text from the image.
- `legal_context`: the question requires combining sign rules with traffic-law
  hierarchy, exceptions, or supplementary-panel validity.
- `reasoning`: the evidence is present but the answer or explanation is wrong.
- `output_format`: prediction/parser output is malformed or invalid. This is
  not a retrieval failure.
- `annotation_data`: the gold citation set appears broader, narrower, or
  noisier than the evidence a user-facing answer would naturally cite.
- `adapter_truncation`: adapter output is cut off or cannot produce valid JSON
  within the configured token budget.

Legacy W3 tags such as `weak_question_text`, `missing_law_context`,
`similar_article_confusion`, `annotation_mismatch`, and `output_format_issue`
should be mapped into the W4 categories above when writing the final report.

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
| 1 | `train_1` | measured | `QCVN 41:2024/BGTVT#22` | `F.10`, `22`, `41`, `B.7`, `73` | `retrieval` | Retrieval partially succeeds because gold `22` appears at rank 2, but rank 1 is a panel article. A VLM may over-focus on the highest ranked panel evidence. |
| 2 | `train_104` | measured | `QCVN 41:2024/BGTVT#36`, `E.14` | `36`, `E.14`, `64`, `E.15`, `70` | `retrieval` | Fusion fixes the weak text-only query and retrieves both gold citations in top 2. Treat this as a hard success case, not a VLM reasoning result. |
| 3 | `train_116` | measured | `28`, `C.3`, `C.27`, `C.45` | `22`, `B.25`, `E.5`, `C.36`, `C.15` | `visual_ambiguity` | The question asks which sign is not mentioned in the image; retrieval misses all gold danger/sign-definition articles. |
| 4 | `train_127` | measured | `28`, `C.3`, `C.27`, `C.45`, `22`, `B.24` | `22`, `B.2`, `B.37`, `B.38`, `54` | `annotation_data` | Retrieval finds the general prohibition article `22` but misses several gold sign IDs. The broad gold set makes recall difficult. |
| 5 | `train_134` | measured | `22`, `B.30` | `22`, `B.31`, `B.39`, `B.38`, `36/2024/QH15#18` | `retrieval` | Retrieval finds the general prohibition article but confuses nearby stopping/parking sign IDs. |
| 6 | `train_135` | watchlist | `28`, `C.2` | pending full `retrieval_final` run | `visual_ambiguity` | Yes/No wording depends on recognizing the exact warning sign family from the image. |
| 7 | `train_136` | watchlist | `27`, `11` | pending full `retrieval_final` run | `legal_context` | The answer depends on group-level warning-sign rules, not only one sign label. |
| 8 | `train_138` | watchlist | `22`, `B.31` | pending full `retrieval_final` run | `retrieval` | “Đây là biển báo gì?” is too short; retrieval must lean heavily on visual/example signals. |
| 9 | `train_139` | watchlist | `22`, `B.3`, `41`, `B.27` | pending full `retrieval_final` run | `legal_context` | Time-window and vehicle-class restrictions require combining sign and auxiliary-panel rules. |
| 10 | `train_18` | watchlist | `26` | pending full `retrieval_final` run | `retrieval` | The wording may retrieve general prohibition content instead of the specific U-turn rule. |
| 11 | `train_187` | watchlist | `22`, `B.3`, `41`, `F.11` | pending full `retrieval_final` run | `visual_ambiguity` | Requires reading a restriction sign and a time/vehicle supplementary panel together. |
| 12 | `train_188` | watchlist | `41`, `F.11` | pending full `retrieval_final` run | `retrieval` | Rectangular blue/white sign descriptions can be confused with nearby auxiliary-sign articles. |
| 13 | `train_189` | watchlist | `36/2024/QH15#11` | pending full `retrieval_final` run | `legal_context` | This is a legal-priority question about traffic-controller orders, not a traffic-sign definition. |
| 14 | `train_190` | watchlist | `41`, `F.10` | pending full `retrieval_final` run | `retrieval` | Similar auxiliary-panel articles `F.10` and `F.11` are easy to swap. |
| 15 | `train_191` | watchlist | `41`, `F.11` | pending full `retrieval_final` run | `visual_ambiguity` | Yes/No answer requires matching the stated time range against the visual panel. |

## Week 4 Extended Case List

The following rows bring the review log to 30 categorized cases. They are
selected from the same locked validation split and must be refreshed with
retrieved citations after running `w4_retrieval_only` or `w4_structured_rag`.

| # | Sample | Status | Gold citations | Retrieved citations | W4 category | Retrieval vs VLM note |
| ---: | --- | --- | --- | --- | --- | --- |
| 16 | `train_192` | watchlist | `41`, `F.10`, `F.11` | pending full W4 run | `legal_context` | The answer depends on whether the driver must obey supplementary-panel content, so retrieval must combine panel validity and sign scope. |
| 17 | `train_193` | watchlist | `22`, `B.3` | pending full W4 run | `retrieval` | Short sign-definition question can retrieve general prohibition content while missing the exact `B.3` sign article. |
| 18 | `train_2` | watchlist | `26`, `14` | pending full W4 run | `visual_ambiguity` | Requires identifying the prohibited vehicle classes in the image before the VLM can choose the legal interpretation. |
| 19 | `train_203` | watchlist | `36/2024/QH15#11` | pending full W4 run | `legal_context` | This is about priority among traffic-controller orders, signs, and lights, not the visual sign definition alone. |
| 20 | `train_204` | watchlist | `36/2024/QH15#9`, `22`, `B.15` | pending full W4 run | `legal_context` | The answer mixes the general road-user compliance duty with a specific prohibition sign. |
| 21 | `train_205` | watchlist | `22`, `B.27` | pending full W4 run | `visual_ambiguity` | Speed-limit interpretation depends on reading the number and sign class correctly. |
| 22 | `train_212` | watchlist | `22`, `B.3` | pending full W4 run | `retrieval` | The wording asks which vehicle is banned from turning right; similar prohibition signs can dominate top-k retrieval. |
| 23 | `train_216` | watchlist | `28`, `C.3` | pending full W4 run | `visual_ambiguity` | Warning-sign meaning and required driver action depend on image recognition rather than question text. |
| 24 | `train_218` | watchlist | `22`, `B.27`, `F.5` | pending full W4 run | `legal_context` | Speed limit plus supplementary distance/scope panel requires evidence fusion across sign and panel articles. |
| 25 | `train_221` | watchlist | `28`, `C.4` | pending full W4 run | `visual_ambiguity` | The target warning sign is visually specific; text-only retrieval may drift to nearby warning articles. |
| 26 | `train_23` | watchlist | `26` | pending full W4 run | `retrieval` | U-turn prohibition wording can be confused with general prohibition articles instead of the exact turning rule. |
| 27 | `train_230` | watchlist | `37`, `E.14` | pending full W4 run | `annotation_data` | Direction/distance signs may require preserving textual content in the image; gold evidence may be narrower than the natural explanation. |
| 28 | `train_231` | watchlist | `37`, `E.14` | pending full W4 run | `visual_ambiguity` | The answer depends on reading place names and distances from the road sign image. |
| 29 | `train_232` | watchlist | `22`, `B.27` | pending full W4 run | `reasoning` | If the speed-limit evidence is retrieved, the model must still infer the maximum speed after passing the sign. |
| 30 | `train_239` | watchlist | `22`, `B.30` | pending full W4 run | `retrieval` | No-stopping/no-parking signs are visually and legally close; retrieval must distinguish `B.30` from nearby `B.31`/`B.39` articles. |

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
