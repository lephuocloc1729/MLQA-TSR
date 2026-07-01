from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import streamlit as st

from src.pipeline import (
    BenchmarkRuntime,
    DEMO_DISCLAIMER,
    build_freeform_demo_inspection,
    demo_model_status,
    normalize_demo_prediction_mode,
)
from src.utils import load_config


DEFAULT_CONFIG_PATH = "configs/config.yaml"
UPLOAD_DIR = Path("data/outputs/demo_uploads")
ALLOWED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]


@st.cache_data(show_spinner=False)
def cached_config(config_path: str) -> dict[str, Any]:
    return load_config(config_path)


@st.cache_resource(show_spinner=False)
def cached_runtime(config_path: str, use_live_backend: bool) -> BenchmarkRuntime:
    config = cached_config(config_path)
    runtime_config = dict(config)
    if not use_live_backend:
        runtime_config["model"] = {
            **dict(config.get("model", {})),
            "backend": "none",
        }
    return BenchmarkRuntime(runtime_config)


def mode_to_internal(label: str) -> str:
    return {
        "Retrieval-only": "retrieval_only",
        "Live VLM": "live",
        "Mock smoke": "mock",
    }[label]


def persist_uploaded_image(uploaded_file: Any) -> Path:
    payload = uploaded_file.getvalue()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    suffix = Path(uploaded_file.name).suffix.lower() or ".jpg"
    if suffix.lstrip(".") not in ALLOWED_IMAGE_TYPES:
        suffix = ".jpg"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / f"freeform_{digest}{suffix}"
    path.write_bytes(payload)
    return path


def render_question(result: dict[str, Any]) -> None:
    sample = result["sample"]
    st.subheader("Free-Form Question")
    st.write(sample["question"])
    st.caption(
        f"Image: `{sample['image_display_name']}` | Mode: free-form legal QA"
    )


def render_image(local_image_path: str | None, result: dict[str, Any]) -> None:
    st.subheader("Uploaded Image")
    if not local_image_path:
        st.warning("No uploaded image path was provided.")
        return
    image_path = Path(local_image_path)
    if not image_path.exists():
        st.warning("Uploaded image is missing from local demo storage.")
        return
    st.image(str(image_path), caption=result["sample"]["image_display_name"], use_container_width=True)


def render_evidence(result: dict[str, Any]) -> None:
    retrieval = result["retrieval"]
    st.subheader("Retrieved Legal Evidence")
    st.caption(
        f"Strategy: `{retrieval['strategy']}` | Top-k: `{retrieval['top_k']}` | "
        f"Evidence count: `{retrieval['evidence_count']}`"
    )

    citation_ids = retrieval.get("citation_ids", [])
    if citation_ids:
        st.text_area("Citation IDs", "\n".join(citation_ids), height=96)
    else:
        st.warning("No legal evidence was retrieved.")

    for item in retrieval["evidence"]:
        score = item["score"]
        score_text = f"{score:.4f}" if isinstance(score, int | float) else "N/A"
        title = item.get("title") or "(untitled article)"
        header = (
            f"#{item.get('rank')} | {item['uid']} | score={score_text} | "
            f"{item['retrieval_method']}"
        )
        with st.expander(header, expanded=item.get("rank") == 1):
            st.markdown(f"**Title:** {title}")
            st.markdown(f"**Citation:** `{item['law_id']}#{item['article_id']}`")
            st.markdown("**Content**")
            st.write(item["content"])

    diagnostics = retrieval.get("diagnostics") or []
    if diagnostics:
        with st.expander("Retrieval diagnostics", expanded=False):
            st.json(diagnostics)


def render_answer(result: dict[str, Any]) -> None:
    st.subheader("Free-Form Answer")
    model = result.get("model") or {}
    st.caption(
        f"Output mode: `{model.get('label') or model.get('mode', 'unknown')}`"
    )

    prediction = result.get("prediction")
    if not prediction:
        st.info(model.get("reason") or "Answer is not available in this mode.")
        return

    answer = prediction.get("answer")
    if answer:
        st.success(answer)
    else:
        st.warning("No valid free-form answer was produced.")

    explanation = prediction.get("explanation")
    if explanation:
        st.markdown("**Explanation**")
        st.write(explanation)

    citations = prediction.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        for citation in citations:
            st.code(f"{citation['law_id']}#{citation['article_id']}")

    confidence = prediction.get("confidence")
    if confidence is not None:
        st.write(f"Confidence: `{confidence}`")

    if prediction.get("abstained"):
        st.warning("The model abstained because evidence was insufficient.")

    parse = prediction.get("parse") or {}
    if parse:
        with st.expander("Parse status", expanded=False):
            st.json(parse)

    error = prediction.get("error") or {}
    if error:
        st.error(f"{error.get('type', 'Error')}: {error.get('message', '')}")


def render_latency(result: dict[str, Any]) -> None:
    latency = result.get("latency_ms") or {}
    if not latency:
        return
    st.subheader("Latency")
    cols = st.columns(3)
    for col, key in zip(cols, ["retrieval", "generation", "total"], strict=False):
        value = latency.get(key)
        label = key.replace("_", " ").title()
        col.metric(label, f"{value:.1f} ms" if isinstance(value, int | float) else "N/A")
    with st.expander("Latency details", expanded=False):
        st.json(latency)


def main() -> None:
    st.set_page_config(
        page_title="Traffic Legal VLM Free-Form Demo",
        layout="wide",
    )
    st.title("Traffic Legal VLM - Free-Form Legal QA Demo")
    st.caption(
        "Upload a traffic image, ask a natural-language question, inspect retrieved "
        "legal evidence, and optionally call a live VLM for a cited answer."
    )
    st.warning(DEMO_DISCLAIMER)

    with st.sidebar:
        st.header("Demo Settings")
        config_path = st.text_input("Config path", DEFAULT_CONFIG_PATH)
        retrieval_strategy = st.selectbox(
            "Retrieval strategy",
            ["text", "fusion", "none"],
            index=0,
            help="Fusion also uses similar train examples if the example index exists.",
        )
        top_k = st.slider("Top-k legal evidence", 1, 10, 5)
        mode_label = st.selectbox(
            "Answer mode",
            ["Retrieval-only", "Live VLM", "Mock smoke"],
            index=0,
        )
        prediction_mode = normalize_demo_prediction_mode(mode_to_internal(mode_label))

    try:
        config = cached_config(config_path)
    except Exception as exc:
        st.error("Could not load config.")
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    model_status = demo_model_status(config, prediction_mode=prediction_mode)
    if model_status.get("available"):
        st.success(model_status.get("reason"))
    else:
        st.info(model_status.get("reason"))

    uploaded_file = st.file_uploader(
        "Upload traffic image",
        type=ALLOWED_IMAGE_TYPES,
        help="Use a dashcam/street image containing traffic signs or road context.",
    )
    question = st.text_area(
        "Ask a free-form legal question",
        placeholder="Ví dụ: Tôi có được đỗ xe ở vị trí này vào cuối tuần không?",
        height=110,
    )

    run_clicked = st.button("Run Free-Form QA", type="primary")
    st.info(
        "Benchmark multiple-choice answering is intentionally removed from this demo. "
        "For official VLSP scoring, use the CLI submission pipeline instead."
    )

    if not run_clicked:
        st.caption("Upload an image, enter a question, then press **Run Free-Form QA**.")
        return
    if uploaded_file is None:
        st.warning("Please upload an image first.")
        return
    if not question.strip():
        st.warning("Please enter a free-form question.")
        return

    image_path = persist_uploaded_image(uploaded_file)
    use_live_backend = prediction_mode == "live" and model_status.get("available")
    try:
        with st.spinner("Retrieving legal evidence and preparing answer..."):
            result = build_freeform_demo_inspection(
                image_path=image_path,
                question=question.strip(),
                config=config,
                top_k=top_k,
                retrieval_strategy_name=retrieval_strategy,
                prediction_mode=prediction_mode,
                runtime=cached_runtime(
                    config_path,
                    use_live_backend=bool(use_live_backend),
                ),
            )
    except Exception as exc:
        st.error("Free-form demo run failed.")
        st.caption(
            "Check Qdrant status, processed LawDB, selected retrieval strategy, "
            "and live backend settings if Live VLM is selected."
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    left, right = st.columns([1, 1])
    with left:
        render_image(result.get("local_image_path"), result)
        render_question(result)
        render_latency(result)
    with right:
        render_answer(result)
        render_evidence(result)


if __name__ == "__main__":
    main()
