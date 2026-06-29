# Demo Assets

This directory stores lightweight report/slide assets such as screenshots.

Recommended final screenshot:

- `final-demo-retrieval-only.png`: Streamlit app opened in retrieval-only mode.
  This screenshot is included as a lightweight defense backup and shows the
  app title, demo settings sidebar, and legal/research disclaimer.

How to refresh the screenshot locally:

```bash
source .venv/bin/activate
make qdrant-up
make preprocess
python -m src.data_utils --mode split
make index
bash scripts/demo.sh
```

Open the Streamlit URL, select a curated sample, keep the output mode as
retrieval-only or cached prediction, and capture the browser window. Do not
include `.env`, API keys, private-test predictions, or hidden reasoning in the
screenshot.

If a live VLM backend is not available, retrieval-only mode is sufficient for
the defense backup because it demonstrates image/question display, retrieved
legal evidence, citation IDs, scores, latency, and the research disclaimer.
