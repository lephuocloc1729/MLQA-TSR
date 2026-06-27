from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.pipeline import (
    BenchmarkRuntime,
    build_demo_inspection,
    load_demo_samples,
)
from src.utils import load_config


DEFAULT_CONFIG_PATH = "configs/config.yaml"


@st.cache_data(show_spinner=False)
def cached_config(config_path: str) -> dict[str, Any]:
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def cached_samples(config_path: str, split: str) -> list[dict[str, Any]]:
    return load_demo_samples(cached_config(config_path), split=split)


@st.cache_resource(show_spinner=False)
def cached_runtime(config_path: str) -> BenchmarkRuntime:
    return BenchmarkRuntime(cached_config(config_path))


def format_sample_option(sample: dict[str, Any]) -> str:
    sample_id = sample.get("id", "(missing id)")
    image_id = sample.get("image_id", "(missing image)")
    question_type = sample.get("question_type", "(unknown)")
    return f"{sample_id} | {image_id} | {question_type}"


def render_question(sample: dict[str, Any]) -> None:
    st.subheader("Question")
    st.write(sample["question"])
    st.caption(f"Sample: `{sample['id']}` | Image: `{sample['image_id']}`")

    choices = sample.get("choices") or {}
    if choices:
        st.markdown("**Choices**")
        for key in sorted(choices):
            st.write(f"**{key}.** {choices[key]}")
    else:
        st.info("This sample has no A/B/C/D choices.")


def render_image(local_image_path: str | None, sample: dict[str, Any]) -> None:
    st.subheader("Image")
    if not local_image_path:
        st.warning("This sample does not include a local image path.")
        return

    image_path = Path(local_image_path)
    if not image_path.exists():
        st.warning(
            "Image file is missing locally. Check data/raw placement before the demo."
        )
        st.caption(f"Expected image file name: `{sample['image_display_name']}`")
        return

    st.image(str(image_path), caption=sample["image_display_name"], use_container_width=True)


def render_evidence(result: dict[str, Any]) -> None:
    retrieval = result["retrieval"]
    st.subheader("Retrieved Legal Evidence")
    st.caption(
        f"Strategy: `{retrieval['strategy']}` | Top-k: `{retrieval['top_k']}` | "
        f"Evidence count: `{retrieval['evidence_count']}`"
    )

    citation_ids = retrieval.get("citation_ids", [])
    if citation_ids:
        st.text_area(
            "Citation IDs for copy/debug",
            "\n".join(citation_ids),
            height=120,
        )
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
            st.markdown(
                f"**Citation:** `{item['law_id']}#{item['article_id']}`"
            )
            st.markdown("**Content**")
            st.write(item["content"])

    diagnostics = retrieval.get("diagnostics") or []
    if diagnostics:
        with st.expander("Retrieval diagnostics", expanded=False):
            st.json(diagnostics)


def render_prediction(result: dict[str, Any]) -> None:
    st.subheader("Model Answer")
    model = result.get("model") or {}
    st.caption(f"Mode: `{model.get('mode', 'unknown')}`")

    prediction = result.get("prediction")
    if not prediction:
        st.info(model.get("reason") or "Prediction is not available in this run.")
        return

    st.success(f"Answer: {prediction['answer']}")
    confidence = prediction.get("confidence")
    if confidence is not None:
        st.write(f"Confidence: `{confidence}`")
    st.write("Explanation:")
    st.write(prediction["explanation"])

    citations = prediction.get("citations") or []
    if citations:
        st.write("Citations:")
        for citation in citations:
            st.code(f"{citation['law_id']}#{citation['article_id']}")
    if prediction.get("abstained"):
        st.warning("The model abstained.")


def main() -> None:
    st.set_page_config(
        page_title="Traffic Legal VLM Evidence Inspector",
        layout="wide",
    )
    st.title("Traffic Legal VLM - Evidence Inspector")
    st.caption(
        "Week-2 debug demo for sample selection, image viewing, legal evidence "
        "inspection, and optional mock answer display."
    )

    with st.sidebar:
        st.header("Demo Settings")
        config_path = st.text_input("Config path", DEFAULT_CONFIG_PATH)
        split = st.selectbox("Split", ["val", "train"], index=0)
        retrieval_strategy = st.selectbox(
            "Retrieval strategy",
            ["text", "fusion", "none"],
            index=0,
            help="Fusion requires the training example index to exist in Qdrant.",
        )
        top_k = st.slider("Top-k legal evidence", 1, 10, 5)
        include_prediction = st.checkbox("Show prediction panel", value=False)
        use_mock_prediction = st.checkbox(
            "Use mock prediction",
            value=False,
            help="Smoke-test only. This is not real VLM accuracy.",
        )

    try:
        samples = cached_samples(config_path, split)
    except Exception as exc:
        st.error(
            "Could not load demo samples. Run preprocessing/split first and check "
            "the config path."
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    if not samples:
        st.warning(f"No samples found in `{split}` split.")
        return

    with st.sidebar:
        selected_sample = st.selectbox(
            "Validation sample",
            samples,
            format_func=format_sample_option,
        )
        run_clicked = st.button("Run Retrieval Inspection", type="primary")

    st.info(
        "This inspection page does not require model/API credentials. If no VLM "
        "backend is configured, it stays in retrieval-only mode."
    )

    if not run_clicked:
        render_question(selected_sample)
        st.caption("Press **Run Retrieval Inspection** to retrieve legal evidence.")
        return

    try:
        with st.spinner("Retrieving legal evidence..."):
            result = build_demo_inspection(
                sample_id=selected_sample["id"],
                config=cached_config(config_path),
                split=split,
                top_k=top_k,
                retrieval_strategy_name=retrieval_strategy,
                include_prediction=include_prediction,
                use_mock_prediction=use_mock_prediction,
                runtime=cached_runtime(config_path),
            )
    except Exception as exc:
        st.error("Retrieval inspection failed.")
        st.caption(
            "Check Qdrant status, processed LawDB, split files, and whether the "
            "selected retrieval strategy has been indexed."
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    left, right = st.columns([1, 1])
    with left:
        render_image(result.get("local_image_path"), result["sample"])
        render_question(result["sample"])
    with right:
        render_evidence(result)
        render_prediction(result)


if __name__ == "__main__":
    main()
