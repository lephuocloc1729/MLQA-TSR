#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  python -m src.evaluate --predictions data/outputs/dev_predictions.jsonl
elif [ "$1" = "benchmark" ]; then
  shift
  python -m src.pipeline --mode benchmark "$@"
elif [ "$1" = "run-experiment" ]; then
  shift
  if [ "$#" -lt 1 ]; then
    echo "Usage: scripts/evaluate.sh run-experiment <config> [limit]" >&2
    exit 2
  fi
  config="$1"
  limit="${2:-}"
  if [ -n "$limit" ]; then
    python -m src.pipeline --mode benchmark --config "$config" --limit "$limit"
  else
    python -m src.pipeline --mode benchmark --config "$config"
  fi
  output_path="$(python - "$config" <<'PY'
import sys
from src.pipeline import benchmark_output_path
from src.utils import load_config
print(benchmark_output_path(load_config(sys.argv[1])))
PY
)"
  python -m src.evaluate --config "$config" --predictions "$output_path"
elif [ "$1" = "run-w3-real" ]; then
  shift
  limit="${1:-5}"
  for config in \
    configs/experiments/w3_b2_text_rag_real.yaml \
    configs/experiments/w3_b5_structured_real.yaml
  do
    "$0" run-experiment "$config" "$limit"
  done
else
  python -m src.evaluate "$@"
fi
