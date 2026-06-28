from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.pipeline import (
    BenchmarkRuntime,
    DEMO_DISCLAIMER,
    build_demo_inspection,
    demo_model_status,
    load_cached_prediction_index,
    load_demo_samples,
    normalize_demo_prediction_mode,
)
from src.utils import load_config


DEFAULT_CONFIG_PATH = "configs/config.yaml"
DEFAULT_CACHED_PREDICTIONS = "data/outputs/experiments/w4_structured_rag.jsonl"
CURATED_CASES = [
    ("train_1", "Success-style: general prohibition article appears in retrieved evidence."),
    ("train_116", "Failure-style: visual ambiguity, danger/warning sign family."),
    ("train_134", "Hard case: no-stopping/no-parking sign confusion."),
    ("train_203", "Legal-context case: traffic-controller order priority."),
    ("train_239", "Hard case: visually similar parking/stopping signs."),
]


@st.cache_data(show_spinner=False)
def cached_config(config_path: str) -> dict[str, Any]:
    return load_config(config_path)


@st.cache_data(show_spinner=False)
def cached_samples(config_path: str, split: str) -> list[dict[str, Any]]:
    return load_demo_samples(cached_config(config_path), split=split)


@st.cache_data(show_spinner=False)
def cached_prediction_index(path: str) -> dict[str, Any]:
    return load_cached_prediction_index(path)


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
            height=100,
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
            st.markdown(f"**Citation:** `{item['law_id']}#{item['article_id']}`")
            st.markdown("**Content**")
            st.write(item["content"])

    diagnostics = retrieval.get("diagnostics") or []
    if diagnostics:
        with st.expander("Retrieval diagnostics", expanded=False):
            st.json(diagnostics)


def render_prediction(result: dict[str, Any]) -> None:
    st.subheader("Answer")
    model = result.get("model") or {}
    st.caption(
        f"Output mode: `{model.get('label') or model.get('mode', 'unknown')}`"
    )

    prediction = result.get("prediction")
    if not prediction:
        st.info(model.get("reason") or "Prediction is not available in this mode.")
        return

    answer = prediction.get("answer")
    if answer:
        st.success(f"Answer: {answer}")
    else:
        st.warning("No valid answer was produced for this sample.")

    confidence = prediction.get("confidence")
    if confidence is not None:
        st.write(f"Confidence: `{confidence}`")

    explanation = prediction.get("explanation")
    if explanation:
        st.markdown("**Explanation**")
        st.write(explanation)

    citations = prediction.get("citations") or []
    if citations:
        st.markdown("**Citations**")
        for citation in citations:
            st.code(f"{citation['law_id']}#{citation['article_id']}")

    parse = prediction.get("parse") or {}
    if parse:
        st.markdown("**Parse status**")
        st.json(parse)

    error = prediction.get("error") or {}
    if error:
        st.error(f"{error.get('type', 'Error')}: {error.get('message', '')}")

    if prediction.get("abstained"):
        st.warning("The model abstained.")


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


def select_curated_sample(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    sample_by_id = {sample.get("id"): sample for sample in samples}
    available = [
        (sample_id, note)
        for sample_id, note in CURATED_CASES
        if sample_id in sample_by_id
    ]
    if not available:
        return None
    label = st.selectbox(
        "Curated presentation case",
        available,
        format_func=lambda item: f"{item[0]} - {item[1]}",
    )
    return sample_by_id[label[0]]


def mode_to_internal(label: str) -> str:
    return {
        "Retrieval-only": "retrieval_only",
        "Cached prediction": "cached",
        "Live VLM": "live",
        "Mock smoke": "mock",
    }[label]


def main() -> None:
    st.set_page_config(
        page_title="Traffic Legal VLM Final Demo",
        layout="wide",
    )
    st.title("Traffic Legal VLM - Final Evidence-Grounded Demo")
    st.caption(
        "Defense-ready demo for retrieval inspection, cached prediction display, "
        "and optional live VLM answering."
    )
    st.warning(DEMO_DISCLAIMER)

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
        mode_label = st.selectbox(
            "Demo mode",
            ["Retrieval-only", "Cached prediction", "Live VLM", "Mock smoke"],
            index=0,
        )
        prediction_mode = normalize_demo_prediction_mode(mode_to_internal(mode_label))
        cached_predictions_path = st.text_input(
            "Cached prediction JSONL",
            DEFAULT_CACHED_PREDICTIONS,
            disabled=prediction_mode != "cached",
        )
        use_curated = st.checkbox("Use curated presentation cases", value=True)

    try:
        config = cached_config(config_path)
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
        curated_sample = select_curated_sample(samples) if use_curated else None
        selected_sample = curated_sample or st.selectbox(
            "Sample",
            samples,
            format_func=format_sample_option,
        )
        model_status = demo_model_status(
            config,
            prediction_mode=prediction_mode,
            cached_predictions_path=cached_predictions_path,
        )
        if model_status.get("available"):
            st.success(model_status.get("reason"))
        else:
            st.info(model_status.get("reason"))

        if prediction_mode == "cached":
            try:
                cache = cached_prediction_index(cached_predictions_path)
                st.caption(f"Cached predictions loaded: `{len(cache)}` rows")
            except Exception as exc:
                st.warning(f"Cached artifact unavailable: {type(exc).__name__}")

        run_clicked = st.button("Run Demo", type="primary")

    st.info(
        "The demo can be presented without a live GPU/API by using retrieval-only "
        "or cached prediction mode."
    )

    if not run_clicked:
        render_question(selected_sample)
        st.caption("Press **Run Demo** to retrieve legal evidence and render output.")
        return

    use_live_backend = prediction_mode == "live" and model_status.get("available")
    try:
        with st.spinner("Running evidence-grounded demo..."):
            result = build_demo_inspection(
                sample_id=selected_sample["id"],
                config=config,
                split=split,
                top_k=top_k,
                retrieval_strategy_name=retrieval_strategy,
                prediction_mode=prediction_mode,
                cached_predictions_path=cached_predictions_path,
                runtime=cached_runtime(config_path, use_live_backend=bool(use_live_backend)),
            )
    except Exception as exc:
        st.error("Demo run failed.")
        st.caption(
            "Check Qdrant status, processed LawDB, split files, cached artifact, "
            "and whether the selected retrieval strategy has been indexed."
        )
        st.caption(f"{type(exc).__name__}: {exc}")
        return

    left, right = st.columns([1, 1])
    with left:
        render_image(result.get("local_image_path"), result["sample"])
        render_question(result["sample"])
        render_latency(result)
    with right:
        render_prediction(result)
        render_evidence(result)


if __name__ == "__main__":
    main()
