# QLoRA Diagnostic Checkpoint Card

## Status

This checkpoint is a diagnostic PEFT artifact. It proves that the training
pipeline, SFT records, collator, LoRA configuration, and checkpoint metadata
writer work end-to-end. It is not the submission-quality model for the course
prototype.

## Checkpoint Identity

| Field | Value |
| --- | --- |
| Base model | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Adapter path | `checkpoints/qlora_adapter` |
| Metadata path | `checkpoints/qlora_adapter/adapter_metadata.json` |
| Metadata schema | `qlora-adapter-metadata-v1` |
| Metadata SHA-256 | `3c75e149babddd56722c5b7e7f36175452b7e205fb877d1e88364102ee989db5` |
| Adapter weights SHA-256 | `5bb63e5782c7f454161a8d1a1f6b9cca7fe3362aced74f5c41649b2d8826a002` |
| Created at | `2026-06-28T08:09:36.814182Z` |
| Source commit in metadata | `d4c27b9e1a31002e466020deba259ec854ab9a9a` |

`checkpoints/` is ignored by Git. Do not commit adapter weights, optimizer
states, tokenizer files, model weights, or generated GPU outputs.

## Data

| Field | Value |
| --- | --- |
| SFT train file | `data/processed/sft_train.jsonl` |
| SFT validation file | `data/processed/sft_val.jsonl` |
| Train records | `421` |
| Validation records | `109` |
| Effective training records | `80` |
| Dataset hash | `939d6ebce062970c2e0f16a73579174084bf2f3b05d1c90b82dad053e4b67df7` |
| Split hash | `3ffba07cf68cccfdfaf921d34d01903223c96810979cb573c68c67c7b3471448` |

## Training Configuration

| Field | Value |
| --- | --- |
| Method | QLoRA |
| GPU used for diagnostic run | RTX 3090 24GB |
| Device in metadata | `cuda` |
| dtype | `bfloat16` |
| 4-bit loading | `true` |
| Frozen vision encoder | `true` |
| LoRA rank / alpha / dropout | `8 / 16 / 0.05` |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | `0.0001` |
| Epochs | `1.0` |
| Per-device batch size | `1` |
| Gradient accumulation | `8` |
| Seed | `42` |
| Estimated VRAM in metadata | `4.7 GB` |

Parameter summary:

| Parameter count | Value |
| --- | ---: |
| Total parameters | `2,052,600,832` |
| Trainable parameters | `18,576,384` |
| Trainable percent | `0.905%` |
| Frozen vision parameters | `335,351,808` |

## Commands

Build SFT records before training:

```bash
python -m src.data_utils --mode build-sft
```

Validate the QLoRA plan without loading model weights:

```bash
python -m src.train_qlora --config configs/qlora.yaml --dry-run
```

Run the small GPU smoke:

```bash
python -m src.train_qlora --config configs/qlora.yaml --max-samples 20
```

Run the reported 80-sample diagnostic adapter:

```bash
python -m src.train_qlora --config configs/qlora.yaml --max-samples 80
```

The attempted 300-sample run uses the default `training.max_samples: 300` in
`configs/qlora.yaml`, but it failed with CUDA OOM on RTX 3090 24GB.

## Validation Smoke

Adapter inference is diagnostic only unless it is wired into the same
benchmark artifact contract as the base VLM runs.

Run a 3-5 sample diagnostic with the safer `max_new_tokens=320` setting:

```bash
python -m src.adapter_infer \
  --adapter checkpoints/qlora_adapter \
  --split val \
  --limit 5 \
  --max-new-tokens 320 \
  --output data/outputs/experiments/w4_adapter_diag.jsonl
```

The equivalent helper is:

```bash
SMOKE_LIMIT=5 make adapter-diagnostic
```

Recorded validation-smoke observations:

- `max_new_tokens=160`: many outputs were truncated and could not be parsed
  reliably as JSON.
- `max_new_tokens=320`: 3-sample validation smoke reached `1/3` exact match
  in `data/outputs/gpu_smoke/adapter_val_smoke_320.jsonl`.

These observations are not leaderboard-quality validation metrics. They should
be used to guide week-4 adapter diagnostics and parser/truncation accounting.

## Evidence Trail

| Claim | Local source |
| --- | --- |
| 80-sample adapter metadata exists | `checkpoints/qlora_adapter/adapter_metadata.json` |
| 20-sample smoke adapter exists | `checkpoints/qlora_adapter_smoke20/adapter_metadata.json` |
| Low-token validation smoke had truncated raw outputs | `data/outputs/gpu_smoke/adapter_val_smoke.jsonl` |
| Safer 320-token validation smoke reached `1/3` exact match | `data/outputs/gpu_smoke/adapter_val_smoke_320.jsonl` |
| 300-sample run exceeded RTX 3090 24GB memory | trainer/GPU run log; summarize in report, do not report as metric |

The checkpoint and GPU-smoke paths above are intentionally ignored by Git. The
report should summarize them, not commit the generated model files.

## Known Limitations

- The 300-sample QLoRA run failed with CUDA OOM on RTX 3090 24GB.
- The current 80-sample adapter is weak on validation smoke.
- Low `max_new_tokens` can truncate JSON outputs.
- Adapter inference is not yet part of the main benchmark runner.
- The current adapter should not be used as the final submission model.

## Recommended Use

Use this checkpoint as evidence for the report that QLoRA training is
reproducible and that PEFT integration is feasible. For week 4, prioritize the
retrieval-grounded base VLM pipeline, structured prompting, final validation,
submission/export, Streamlit demo polish, and defense assets.

## Local Access And Checksum Commands

Verify the local metadata and adapter files without committing them:

```bash
shasum -a 256 checkpoints/qlora_adapter/adapter_metadata.json
shasum -a 256 checkpoints/qlora_adapter/adapter_model.safetensors
```

If the checkpoint must be shared, store it outside Git, for example in a
private drive or release artifact, and share the metadata SHA-256 with the
team.
