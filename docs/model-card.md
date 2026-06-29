# System Model Card

## Model/System Name

Traffic Legal VLM course prototype.

## Intended Use

The system is intended for research, education, and defense demonstration of a
multimodal legal QA pipeline for Vietnamese traffic-sign regulations. It can:

- inspect a traffic image and question;
- retrieve legal evidence from LawDB;
- produce a structured answer with citations and a short explanation;
- export internal predictions to a submission-style JSON format.

It is not official legal advice and should not be used as the only basis for
real-world traffic or legal decisions.

## Main Product Variant

The recommended course-product variant is:

- config: `configs/experiments/w4_structured_rag.yaml`;
- retrieval: frozen fused LawDB evidence from `retrieval-final-v1`;
- prompt: `structured_legal_rag`;
- output: JSON-compatible `Prediction` with `answer`, `citations`,
  `explanation`, `confidence`, and `abstained`.

Real metrics are reportable only after a `mock=false` metrics artifact exists.

## Inputs

- Traffic image path or uploaded image.
- Question text.
- Optional choices for multiple-choice questions.
- Retrieved evidence from Qdrant.
- Optional cached prediction JSONL for demo mode.

## Outputs

- Benchmark answer: `A/B/C/D` or `Đúng/Sai` after normalization in the runtime.
- Citations: LawDB `law_id` and `article_id`.
- Short explanation, not hidden chain-of-thought.
- Confidence and abstention flag.
- Timing and parse diagnostics in benchmark artifacts.

## Data And Evaluation

The current locked split uses:

- train samples: 421;
- validation samples: 109;
- split hash:
  `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448`.

Retrieval metrics are macro Precision/Recall/F2 over citation UIDs. QA metrics
are exact-match accuracy after NFC normalization. Invalid JSON, unsupported
citations, truncated output, and invalid labels are counted separately.

## Known Limitations

- Visual signs, supplementary panels, lane context, speeds, and vehicle classes
  can be misread.
- Retrieval can miss the correct legal article or retrieve a nearby sign ID.
- The model may output malformed JSON or citations outside the evidence set.
- Real-backend metrics are unavailable until a hosted/local VLM endpoint is
  configured and run.
- The QLoRA adapter is diagnostic and not submission-ready.

## Safety And Ethics

- Display the legal/research disclaimer in demos.
- Do not expose API keys, `.env`, hidden reasoning, raw private-test outputs, or
  local absolute paths unnecessarily.
- Do not silently replace invalid answers with defaults.
- Keep model checkpoints and generated private outputs outside Git.

## Four-Month Continuation

The next project phase should prioritize:

1. full locked-validation real VLM evaluation;
2. retrieval error reduction based on `docs/error-analysis.md`;
3. measured OCR/sign-crop ablations only if they improve validation retrieval;
4. robust JSON/citation validation;
5. larger QLoRA training only after baseline metrics are stable.
