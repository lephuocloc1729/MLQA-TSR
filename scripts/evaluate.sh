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
elif [ "$1" = "hybrid-submission" ]; then
  shift
  if [ "$#" -lt 3 ]; then
    echo "Usage: scripts/evaluate.sh hybrid-submission <candidate-name> <task1-jsonl> <task2-jsonl> [set-name]" >&2
    exit 2
  fi
  candidate="$1"
  task1_predictions="$2"
  task2_predictions="$3"
  set_name="${4:-private_test}"
  case "$candidate" in
    *[!A-Za-z0-9_.-]*|"")
      echo "candidate-name must contain only letters, numbers, dot, underscore, or dash" >&2
      exit 2
      ;;
  esac

  output_root="data/outputs/submissions"
  output_dir="$output_root/$candidate"
  named_zip="$output_root/submission_${candidate}.zip"
  final_zip="$output_root/submission.zip"
  if [ -e "$output_dir" ]; then
    echo "Refusing to overwrite existing candidate directory: $output_dir" >&2
    exit 2
  fi
  if [ -e "$named_zip" ]; then
    echo "Refusing to overwrite existing named zip: $named_zip" >&2
    exit 2
  fi

  "$PYTHON_BIN" -m src.competition_submission \
    --set-name "$set_name" \
    --task both \
    --task1-predictions "$task1_predictions" \
    --task2-predictions "$task2_predictions" \
    --output-dir "$output_dir"

  "$PYTHON_BIN" -m src.competition_submission \
    --pack "$output_dir" \
    --output "$named_zip"

  if [ -e "$final_zip" ]; then
    timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
    cp "$final_zip" "$output_root/submission_backup_${timestamp}.zip"
  fi
  cp "$named_zip" "$final_zip"

  "$PYTHON_BIN" - "$candidate" "$task1_predictions" "$task2_predictions" "$named_zip" "$final_zip" <<'PY'
import hashlib
import sys
import zipfile
from pathlib import Path

candidate, task1, task2, named_zip, final_zip = sys.argv[1:]
zip_path = Path(named_zip)
digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
with zipfile.ZipFile(zip_path) as archive:
    entries = archive.namelist()
if entries != ["submission_task1.json", "submission_task2.json"]:
    raise SystemExit(f"ERROR: unexpected zip entries: {entries}")
print(f"candidate={candidate}")
print(f"named_zip={named_zip}")
print(f"submission_zip={final_zip}")
print(f"sha256={digest}")
print("entries=" + ",".join(entries))
print()
print("| candidate | task1 artifact | task2 artifact | sha256 | F2 | Accuracy | notes |")
print("|---|---|---|---|---:|---:|---|")
print(
    f"| {candidate} | {task1} | {task2} | {digest} | TBD | TBD | "
    "record Codabench result after upload |"
)
PY
elif [ "$1" = "lowcost-task1" ]; then
  shift
  "$PYTHON_BIN" -m src.lowcost_retrieval --mode task1 "$@"
elif [ "$1" = "lowcost-task1-ablate" ]; then
  shift
  "$PYTHON_BIN" -m src.lowcost_retrieval --mode ablate "$@"
else
  "$PYTHON_BIN" -m src.evaluate "$@"
fi
