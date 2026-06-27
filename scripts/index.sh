#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  python -m src.retrieval --mode index
elif [ "$1" = "law" ]; then
  shift
  python -m src.retrieval --mode index "$@"
elif [ "$1" = "examples" ]; then
  shift
  python -m src.retrieval --mode index-examples --split train "$@"
elif [ "$1" = "week2" ]; then
  shift
  python -m src.retrieval --mode index "$@"
  python -m src.retrieval --mode index-examples --split train
else
  python -m src.retrieval "$@"
fi
