#!/bin/bash
set -euo pipefail

python -m streamlit run app/streamlit_app.py "$@"
