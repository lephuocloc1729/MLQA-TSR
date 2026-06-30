#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [ "$#" -eq 0 ]; then
  "$PYTHON_BIN" -m src.evaluate --predictions data/outputs/dev_predictions.jsonl
elif [ "$1" = "benchmark" ]; then
  shift
  "$PYTHON_BIN" -m src.pipeline --mode benchmark "$@"
elif [ "$1" = "vlsp-test" ]; then
  shift
  "$PYTHON_BIN" -m src.pipeline --mode vlsp-test "$@"
elif [ "$1" = "run-experiment" ]; then
  shift
  if [ "$#" -lt 1 ]; then
    echo "Usage: scripts/evaluate.sh run-experiment <config> [limit]" >&2
    exit 2
  fi
  config="$1"
  limit="${2:-}"
  if [ -n "$limit" ]; then
    "$PYTHON_BIN" -m src.pipeline --mode benchmark --config "$config" --limit "$limit"
  else
    "$PYTHON_BIN" -m src.pipeline --mode benchmark --config "$config"
  fi
  output_path="$("$PYTHON_BIN" - "$config" <<'PY'
import sys
from src.pipeline import benchmark_output_path
from src.utils import load_config
print(benchmark_output_path(load_config(sys.argv[1])))
PY
)"
  "$PYTHON_BIN" -m src.evaluate --config "$config" --predictions "$output_path"
elif [ "$1" = "run-w3-real" ]; then
  shift
  limit="${1:-5}"
  for config in \
    configs/experiments/w3_b2_text_rag_real.yaml \
    configs/experiments/w3_b5_structured_real.yaml
  do
    "$0" run-experiment "$config" "$limit"
  done
elif [ "$1" = "run-w5-qwen" ]; then
  shift
  limit="${1:-1}"
  config="${2:-configs/experiments/vlsp_task2_qwen25vl_7b.yaml}"
  if [ "$limit" = "full" ]; then
    "$0" run-experiment "$config"
  else
    "$0" run-experiment "$config" "$limit"
  fi
elif [ "$1" = "run-w6-lowcost-task2" ]; then
  shift
  limit="${1:-5}"
  config="${2:-configs/experiments/lowcost_task2_qwen_answer_only.yaml}"
  if [ "$limit" = "full" ]; then
    "$0" run-experiment "$config"
  else
    "$0" run-experiment "$config" "$limit"
  fi
elif [ "$1" = "run-w6-lowcost-task2-matrix" ]; then
  shift
  limit="${1:-5}"
  for config in \
    configs/experiments/lowcost_task2_qwen_answer_only_no_examples.yaml \
    configs/experiments/lowcost_task2_qwen_answer_only.yaml
  do
    "$0" run-w6-lowcost-task2 "$limit" "$config"
  done
elif [ "$1" = "adapter-diagnostic" ]; then
  shift
  limit="${1:-5}"
  config="${2:-configs/experiments/w4_adapter_diag.yaml}"
  "$PYTHON_BIN" -m src.adapter_infer --config "$config" --limit "$limit"
  output_path="$("$PYTHON_BIN" - "$config" <<'PY'
import sys
from src.utils import load_config
config = load_config(sys.argv[1])
print(config.get("adapter_diagnostic", {}).get("output_path", "data/outputs/experiments/w4_adapter_diag.jsonl"))
PY
)"
  "$PYTHON_BIN" -m src.evaluate --config "$config" --predictions "$output_path"
elif [ "$1" = "submission" ]; then
  shift
  "$PYTHON_BIN" -m src.submission "$@"
elif [ "$1" = "competition-submission" ]; then
  shift
  "$PYTHON_BIN" -m src.competition_submission "$@"
elif [ "$1" = "lowcost-task1" ]; then
  shift
  "$PYTHON_BIN" -m src.lowcost_retrieval --mode task1 "$@"
elif [ "$1" = "lowcost-task1-ablate" ]; then
  shift
  "$PYTHON_BIN" -m src.lowcost_retrieval --mode ablate "$@"
else
  "$PYTHON_BIN" -m src.evaluate "$@"
fi
