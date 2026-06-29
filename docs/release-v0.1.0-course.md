# Release Notes - v0.1.0-course

## Release Goal

`v0.1.0-course` is the final four-week course prototype for Multimodal Legal
Question Answering for Traffic Sign Regulations. It packages the implemented
retrieval-grounded QA pipeline, evaluation contract, Streamlit demo,
submission converter, final report, slides source, and QLoRA diagnostic notes.

## Implemented Scope

- LawDB preprocessing into 402 article-level records.
- Leakage-safe grouped train/validation split by `image_id`.
- Qdrant legal retrieval and train-example retrieval/fusion.
- Structured legal QA prompt variants and strict JSON output parsing.
- Benchmark artifact contract and retrieval/QA metrics.
- OpenAI-compatible real VLM backend wiring.
- QLoRA SFT data, trainer, checkpoint metadata, and adapter diagnostic runner.
- Submission converter and validator.
- Final Streamlit demo with retrieval-only, cached prediction, mock, and live
  modes.
- Final report, slide source, model card, checkpoint card, and error analysis.

## Reported Metrics

Only metrics backed by local JSON artifacts are reportable:

| Artifact | Status |
| --- | --- |
| `data/outputs/experiments/w2_b2_text_rag_metrics.json` | 5-sample mock smoke |
| `data/outputs/experiments/w2_b3_fused_rag_metrics.json` | 5-sample mock smoke |
| `data/outputs/experiments/w2_b4_few_shot_rag_metrics.json` | 5-sample mock smoke |
| `data/outputs/experiments/retrieval_final_metrics.json` | 5-sample mock smoke |

Real/base VLM validation metrics are pending until a `mock=false` metrics JSON
exists. QLoRA is diagnostic only and must not be presented as the final model.

## Defense Rehearsal Checklist

Run in the project virtual environment:

```bash
source .venv/bin/activate
make release-check
make qdrant-up
make preprocess
python -m src.data_utils --mode split
make index
make index-examples
```

Smoke one benchmark path:

```bash
python -m src.pipeline --mode benchmark \
  --config configs/experiments/retrieval_final.yaml \
  --limit 5

python -m src.evaluate \
  --config configs/experiments/retrieval_final.yaml \
  --predictions data/outputs/experiments/retrieval_final.jsonl
```

Rehearse submission dry-run with an available prediction artifact:

```bash
python -m src.submission \
  --predictions data/outputs/experiments/retrieval_final.jsonl \
  --set-name val \
  --allow-missing \
  --dry-run
```

Rehearse the demo:

```bash
bash scripts/demo.sh
```

Use `docs/assets/final-demo-retrieval-only.png` as a backup screenshot if live
services fail during defense.

## Local Rehearsal Result

Recorded on `2026-06-29` in the supported local `.venv` environment:

| Check | Result |
| --- | --- |
| `make release-check` | passed schema CI test, `pip check`, CLI help checks, `check_data.sh --all`, and ignored-artifact review |
| `bash scripts/check_data.sh --all` | passed required, public, and private raw-data path checks |
| `make qdrant-up preprocess index` | Qdrant running; 402 LawDB articles indexed into `traffic_law` |
| `python -m src.pipeline --mode benchmark --config configs/experiments/retrieval_final.yaml --limit 1` | passed mock retrieval benchmark smoke |
| `python -m src.evaluate --config configs/experiments/retrieval_final.yaml --predictions data/outputs/experiments/retrieval_final.jsonl` | passed and wrote ignored metrics artifact |
| `python -m src.submission --predictions data/outputs/experiments/retrieval_final.jsonl --set-name val --allow-missing --dry-run` | passed smoke validation; reports missing IDs because the artifact is intentionally limited |
| `bash scripts/demo.sh --server.headless true --server.port 8610` | Streamlit server started successfully and was stopped after startup check |

The benchmark/evaluation outputs from this rehearsal live under
`data/outputs/` and must remain ignored.

## Release Hygiene

- Do not stage raw data under `data/raw/`.
- Do not stage generated outputs under `data/outputs/`.
- Do not stage Qdrant storage, embeddings, model weights, or checkpoints.
- Keep real credentials only in `.env` or shell environment variables.
- Run `git status --ignored --short` before tagging.

## Known Limitations

- The system is research/education assistance, not official legal advice.
- Real VLM metrics require a configured OpenAI-compatible endpoint.
- Retrieval can confuse visually or legally similar traffic-sign articles.
- VLM output can be malformed or truncated.
- The 80-sample QLoRA adapter is diagnostic and not submission-ready.

## Tagging

Create the release tag only after checks pass and the PR is merged:

```bash
git tag -a v0.1.0-course -m "Final four-week capstone prototype"
git push origin v0.1.0-course
```
