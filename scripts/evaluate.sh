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
elif [ "$1" = "adapter-diagnostic" ]; then
  shift
  limit="${1:-5}"
  config="${2:-configs/experiments/w4_adapter_diag.yaml}"
  python -m src.adapter_infer --config "$config" --limit "$limit"
  output_path="$(python - "$config" <<'PY'
import sys
from src.utils import load_config
config = load_config(sys.argv[1])
print(config.get("adapter_diagnostic", {}).get("output_path", "data/outputs/experiments/w4_adapter_diag.jsonl"))
PY
)"
  python -m src.evaluate --config "$config" --predictions "$output_path"
elif [ "$1" = "submission" ]; then
  shift
  python -m src.submission "$@"
elif [ "$1" = "competition-submission" ]; then
  shift
  python -m src.competition_submission "$@"
else
  python -m src.evaluate "$@"
fi
