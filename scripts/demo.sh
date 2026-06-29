#!/bin/bash
set -euo pipefail

python_bin="${PYTHON:-python}"
if [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
fi

"$python_bin" - <<'PY'
import importlib.util
if importlib.util.find_spec("streamlit") is None:
    print("Missing dependency: streamlit. Run `make setup` inside the project virtual environment.", flush=True)
    raise SystemExit(1)
PY

"$python_bin" -m streamlit run app/streamlit_app.py "$@"
