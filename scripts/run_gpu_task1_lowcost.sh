#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

CONFIG="${CONFIG:-configs/experiments/lowcost_retrieval.yaml}"
TASK1_CONFIG="${TASK1_CONFIG:-configs/experiments/lowcost_task1_text_image_object.yaml}"
SET_NAME="${SET_NAME:-private_test}"
FEATURE_DIR="${FEATURE_DIR:-data/outputs/lowcost_features_gpu}"
TRAIN_FEATURES="$FEATURE_DIR/train_features.jsonl"
QUERY_FEATURES="$FEATURE_DIR/${SET_NAME}_features.jsonl"
TASK1_OUTPUT="${TASK1_OUTPUT:-data/outputs/competitions/${SET_NAME}_task1_lowcost_gpu_t10_i5_o3.jsonl}"
TASK2_ARTIFACT="${TASK2_ARTIFACT:-data/outputs/competitions/private_task2_lowcost_answer_only_no_examples_repaired_strict.jsonl}"
CANDIDATE="${CANDIDATE:-task1_lowcost_gpu_t10_i5_o3_task2_answer_only_repaired_$(date -u +%Y%m%dT%H%M%SZ)}"
PACKAGE="${PACKAGE:-1}"
ALLOW_CPU="${ALLOW_CPU:-0}"

if [ "$ALLOW_CPU" != "1" ]; then
  "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Set ALLOW_CPU=1 only for an intentional slow CPU smoke run.")
print("cuda_available=true")
print("cuda_device_count=" + str(torch.cuda.device_count()))
print("cuda_device_name=" + torch.cuda.get_device_name(0))
PY
else
  echo "ALLOW_CPU=1 set; running low-cost feature extraction without requiring CUDA."
fi

if command -v docker >/dev/null 2>&1 && [ -f docker-compose.yml ]; then
  docker compose up -d
fi

echo "Caching low-cost train features into $FEATURE_DIR"
bash scripts/lowcost_features.sh \
  --config "$CONFIG" \
  --set-name train \
  --resume \
  --output-dir "$FEATURE_DIR"

echo "Caching low-cost $SET_NAME features into $FEATURE_DIR"
bash scripts/lowcost_features.sh \
  --config "$CONFIG" \
  --set-name "$SET_NAME" \
  --resume \
  --output-dir "$FEATURE_DIR"

echo "Indexing train examples in Qdrant"
bash scripts/lowcost_index.sh \
  --config "$CONFIG" \
  --mode index \
  --features "$TRAIN_FEATURES"

echo "Running low-cost Task 1 retrieval for $SET_NAME"
bash scripts/evaluate.sh lowcost-task1 \
  --config "$TASK1_CONFIG" \
  --set-name "$SET_NAME" \
  --features "$QUERY_FEATURES" \
  --output "$TASK1_OUTPUT"

if [ "$PACKAGE" = "1" ]; then
  if [ "$SET_NAME" != "private_test" ]; then
    echo "Skipping hybrid packaging because SET_NAME=$SET_NAME is not private_test."
  elif [ ! -f "$TASK2_ARTIFACT" ]; then
    echo "Skipping hybrid packaging because Task 2 artifact is missing: $TASK2_ARTIFACT"
  else
    echo "Packaging hybrid candidate $CANDIDATE"
    bash scripts/evaluate.sh hybrid-submission \
      "$CANDIDATE" \
      "$TASK1_OUTPUT" \
      "$TASK2_ARTIFACT" \
      "$SET_NAME"
  fi
fi

echo "Done."
echo "task1_output=$TASK1_OUTPUT"
echo "features=$FEATURE_DIR"
