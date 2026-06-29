#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

if [ "$#" -eq 0 ]; then
  "$PYTHON_BIN" -m src.retrieval --mode index
elif [ "$1" = "law" ]; then
  shift
  "$PYTHON_BIN" -m src.retrieval --mode index "$@"
elif [ "$1" = "examples" ]; then
  shift
  "$PYTHON_BIN" -m src.retrieval --mode index-examples --split train "$@"
elif [ "$1" = "week2" ]; then
  shift
  "$PYTHON_BIN" -m src.retrieval --mode index "$@"
  "$PYTHON_BIN" -m src.retrieval --mode index-examples --split train
else
  "$PYTHON_BIN" -m src.retrieval "$@"
fi
