#!/bin/bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [ -z "${PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
fi

"$PYTHON_BIN" -m src.lowcost_retrieval "$@"
