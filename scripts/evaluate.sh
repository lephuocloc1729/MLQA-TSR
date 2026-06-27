#!/bin/bash
set -euo pipefail

if [ "$#" -eq 0 ]; then
  python -m src.evaluate --predictions data/outputs/dev_predictions.jsonl
else
  python -m src.evaluate "$@"
fi
