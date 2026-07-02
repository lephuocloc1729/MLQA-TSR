from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from src.data_utils import (
    load_processed_law_article_index,
    load_split_samples,
    load_vlsp_test_samples,
)
from src.evaluate import extract_sample_id, read_prediction_jsonl
from src.lowcost_features import LowCostFeatureExtractor, make_lowcost_extractor
from src.lowcost_retrieval import (
    LowCostVectorStore,
    make_lowcost_vector_store,
    retrieve_task1_citation_evidence,
)
from src.prompts import PromptVariant, normalize_prompt_variant
from src.retrieval import (
    ImageEmbedder,
    TextEmbedder,
    TextVectorStore,
    ExampleVectorStore,
    make_embedder,
    make_example_vector_store,
    make_image_embedder_from_config,
    make_vector_store,
    retrieve_evidence,
    retrieve_examples,
    retrieve_fused_evidence,
)
from src.schemas import Citation, Evidence, PipelineResult, Prediction, Query, QuestionType
from src.utils import load_config, read_json, stable_json_hash, utc_now_iso, write_jsonl
from src.vlm import LegalQAVLM


EXPERIMENT_SCHEMA_VERSION = "w2-ablation-v1"
VLSP_TEST_SCHEMA_VERSION = "vlsp-test-v1"
DEFAULT_EXPERIMENT_DIR = "data/outputs/experiments"
DEFAULT_COMPETITION_DIR = "data/outputs/competitions"
DEMO_SCHEMA_VERSION = "demo-inspection-v1"
DEMO_DISCLAIMER = (
    "Demo phục vụ nghiên cứu/giáo dục và hỗ trợ tra cứu. Kết quả không phải "
    "tư vấn pháp lý chính thức; hãy đối chiếu văn bản pháp luật gốc khi sử dụng."
)
DEMO_PREDICTION_MODES = {"retrieval_only", "cached", "live", "mock"}


class BenchmarkRuntime:
    """Lazy dependencies reused across samples in one benchmark run."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.vlm = LegalQAVLM(config)
        self._text_embedder: TextEmbedder | None = None
        self._law_vector_store: TextVectorStore | None = None
        self._example_vector_store: ExampleVectorStore | None = None
        self._image_embedder: ImageEmbedder | None = None
        self._lowcost_feature_extractor: LowCostFeatureExtractor | None = None
        self._lowcost_vector_store: LowCostVectorStore | None = None

    def text_embedder(self) -> TextEmbedder:
        if self._text_embedder is None:
            self._text_embedder = make_embedder(self.config)
        return self._text_embedder

    def law_vector_store(self) -> TextVectorStore:
        if self._law_vector_store is None:
            self._law_vector_store = make_vector_store(self.config)
        return self._law_vector_store

    def example_vector_store(self) -> ExampleVectorStore:
        if self._example_vector_store is None:
            self._example_vector_store = make_example_vector_store(self.config)
        return self._example_vector_store

    def image_embedder(self) -> ImageEmbedder:
        if self._image_embedder is None:
            self._image_embedder = make_image_embedder_from_config(self.config)
        return self._image_embedder

    def lowcost_feature_extractor(self) -> LowCostFeatureExtractor:
        if self._lowcost_feature_extractor is None:
            self._lowcost_feature_extractor = make_lowcost_extractor(self.config)
        return self._lowcost_feature_extractor

    def lowcost_vector_store(self) -> LowCostVectorStore:
        if self._lowcost_vector_store is None:
            self._lowcost_vector_store = make_lowcost_vector_store(self.config)
        return self._lowcost_vector_store


def experiment_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("experiment", {}))


def experiment_name(config: Mapping[str, Any]) -> str:
    experiment = experiment_config(config)
    project = config.get("project", {})
    return str(experiment.get("name") or project.get("name") or "experiment")


def benchmark_output_path(config: Mapping[str, Any]) -> Path:
    experiment = experiment_config(config)
    if experiment.get("output_path"):
        return Path(str(experiment["output_path"]))
    return Path(DEFAULT_EXPERIMENT_DIR) / f"{experiment_name(config)}.jsonl"


def split_path_for_config(config: Mapping[str, Any]) -> str:
    experiment = experiment_config(config)
    split_name = str(experiment.get("split", "val"))
    data_config = config.get("data", {})
    key = "train_split_path" if split_name == "train" else "val_split_path"
    path = data_config.get(key)
    if not path:
        raise KeyError(f"Missing data.{key} for experiment split {split_name!r}")
    return str(path)


def locked_split_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    data_config = config.get("data", {})
    manifest_path = data_config.get("split_manifest_path")
    identity: dict[str, Any] = {
        "split": experiment_config(config).get("split", "val"),
        "split_path": split_path_for_config(config),
        "split_manifest_path": str(manifest_path) if manifest_path else None,
        "split_hash": None,
    }
    if manifest_path and Path(str(manifest_path)).exists():
        manifest = read_json(str(manifest_path))
        identity["split_hash"] = manifest.get("split_hash")
    return identity


def assert_locked_validation_split(configs: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Reject accidental split drift across an ablation matrix."""
    if not configs:
        raise ValueError("At least one experiment config is required")

    baseline = locked_split_identity(configs[0])
    for config in configs[1:]:
        current = locked_split_identity(config)
        if current == baseline:
            continue
        if experiment_config(config).get("allow_split_override"):
            continue
        raise ValueError(
            "Experiment configs use different validation splits: "
            f"{baseline} != {current}. Set experiment.allow_split_override=true "
            "only for an intentional non-comparable run."
        )
    return baseline


def load_benchmark_samples(config: Mapping[str, Any], limit: int | None = None) -> list[dict]:
    split = str(experiment_config(config).get("split", "val"))
    samples = load_split_samples(dict(config), split)
    return samples[:limit] if limit is not None else samples


def load_vlsp_samples(
    config: Mapping[str, Any],
    set_name: str,
    task: str,
    limit: int | None = None,
) -> list[dict]:
    samples = load_vlsp_test_samples(config, set_name=set_name, task=task)
    return samples[:limit] if limit is not None else samples


def demo_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("demo", {}))


def normalize_demo_prediction_mode(
    prediction_mode: str | None = None,
    include_prediction: bool = False,
    use_mock_prediction: bool = False,
) -> str:
    if prediction_mode is None:
        if not include_prediction:
            return "retrieval_only"
        return "mock" if use_mock_prediction else "live"

    mode = str(prediction_mode).strip().lower().replace("-", "_")
    aliases = {
        "retrieval": "retrieval_only",
        "none": "retrieval_only",
        "cached_prediction": "cached",
        "live_vlm": "live",
        "mock_prediction": "mock",
    }
    mode = aliases.get(mode, mode)
    if mode not in DEMO_PREDICTION_MODES:
        raise ValueError(
            f"prediction_mode must be one of {sorted(DEMO_PREDICTION_MODES)}"
        )
    return mode


def load_demo_samples(config: Mapping[str, Any], split: str = "val") -> list[dict[str, Any]]:
    return load_split_samples(dict(config), split)


def load_demo_sample_by_id(
    config: Mapping[str, Any],
    sample_id: str,
    split: str = "val",
) -> dict[str, Any]:
    for sample in load_demo_samples(config, split=split):
        if sample.get("id") == sample_id:
            return sample
    raise KeyError(f"Sample {sample_id!r} not found in {split!r} split")


def retrieval_strategy(config: Mapping[str, Any]) -> str:
    experiment = experiment_config(config)
    strategy = str(experiment.get("retrieval_strategy", "text")).strip().lower()
    allowed = {"none", "text", "fusion", "task1", "hybrid"}
    if strategy not in allowed:
        raise ValueError(f"retrieval_strategy must be one of {sorted(allowed)}")
    return strategy


def prompt_variant(config: Mapping[str, Any]) -> PromptVariant:
    experiment = experiment_config(config)
    prompt_config = config.get("prompt", {})
    return normalize_prompt_variant(
        experiment.get("prompt_variant") or prompt_config.get("variant")
    )


def use_examples_for_prompt(config: Mapping[str, Any]) -> bool:
    experiment = experiment_config(config)
    if "use_examples" in experiment:
        return bool(experiment["use_examples"])
    return prompt_variant(config) in {
        PromptVariant.FEW_SHOT_RAG,
        PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT,
    }


def is_mock_run(config: Mapping[str, Any]) -> bool:
    return bool(experiment_config(config).get("mock", True))


def model_run_metadata(
    config: Mapping[str, Any],
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    """Return lightweight model/backend metadata for benchmark artifacts."""
    model_config = config.get("model", {})
    vlm = getattr(runtime, "vlm", None) if runtime is not None else None
    gpu_host_env = model_config.get("gpu_host_env")
    return {
        "backend": getattr(vlm, "backend", model_config.get("backend", "none")),
        "name": getattr(vlm, "model_name", model_config.get("name")),
        "name_env": model_config.get("name_env"),
        "temperature": getattr(vlm, "temperature", model_config.get("temperature")),
        "max_new_tokens": getattr(vlm, "max_new_tokens", model_config.get("max_new_tokens")),
        "include_image": getattr(vlm, "include_image", model_config.get("include_image")),
        "max_retries": model_config.get("max_retries"),
        "serving": model_config.get("serving"),
        "generation_config": model_config.get("generation_config"),
        "gpu_host": (
            os.environ.get(str(gpu_host_env))
            if gpu_host_env
            else model_config.get("gpu_host")
        ),
        "gpu_host_env": gpu_host_env,
        "dtype": model_config.get("dtype"),
        "quantization": model_config.get("quantization"),
    }


def retrieval_run_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_config = config.get("retrieval", {})
    experiment = experiment_config(config)
    freeze = config.get("retrieval_freeze", {})
    return {
        "strategy": retrieval_strategy(config),
        "freeze_version": freeze.get("version"),
        "top_k": retrieval_config.get("top_k"),
        "example_top_k": retrieval_config.get("example_top_k"),
        "example_retrieval_mode": experiment.get("example_retrieval_mode"),
        "text_weight": retrieval_config.get("text_weight"),
        "image_weight": retrieval_config.get("image_weight"),
        "fusion_direct_weight": retrieval_config.get("fusion_direct_weight"),
        "fusion_example_vote_weight": retrieval_config.get("fusion_example_vote_weight"),
    }


def looks_like_invalid_json_error(error: Exception) -> bool:
    message = str(error).casefold()
    error_type = type(error).__name__.casefold()
    return (
        "json" in message
        or "json" in error_type
        or "does not contain a json object" in message
        or "must contain exactly answer" in message
        or "response json must be an object" in message
    )


def looks_like_truncated_output(error: Exception | None = None, raw_response: str | None = None) -> bool:
    text = " ".join(
        part
        for part in [
            str(error) if error is not None else "",
            raw_response or "",
        ]
        if part
    ).casefold()
    return any(
        marker in text
        for marker in (
            "truncated",
            "unterminated",
            "unexpected end",
            "eof",
            "max_new_tokens",
        )
    )


def parse_success_metadata(prediction: Prediction) -> dict[str, Any]:
    raw_response = prediction.raw_response
    return {
        "status": "success",
        "success": True,
        "invalid_json": False,
        "truncated_output": looks_like_truncated_output(raw_response=raw_response),
        "raw_response_available": bool(raw_response),
    }


def parse_error_metadata(error: Exception) -> dict[str, Any]:
    invalid_json = looks_like_invalid_json_error(error)
    truncated = looks_like_truncated_output(error=error)
    return {
        "status": "invalid_json" if invalid_json else "error",
        "success": False,
        "invalid_json": invalid_json,
        "truncated_output": truncated,
        "raw_response_available": False,
        "error_type": type(error).__name__,
    }


def demo_model_status(
    config: Mapping[str, Any],
    include_prediction: bool = False,
    use_mock_prediction: bool = False,
    prediction_mode: str | None = None,
    cached_predictions_path: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe whether the demo can produce an answer without exposing secrets."""
    environ = environ or os.environ
    mode = normalize_demo_prediction_mode(
        prediction_mode,
        include_prediction=include_prediction,
        use_mock_prediction=use_mock_prediction,
    )
    config_demo = demo_config(config)

    if mode == "retrieval_only":
        return {
            "available": False,
            "mode": "retrieval_only",
            "label": "Retrieval-only",
            "reason": "Prediction disabled; showing retrieval evidence only.",
        }
    if mode == "cached":
        if not cached_predictions_path:
            return {
                "available": False,
                "mode": "cached_prediction",
                "label": "Cached prediction",
                "reason": "No cached prediction artifact was provided.",
            }
        if not Path(cached_predictions_path).exists():
            return {
                "available": False,
                "mode": "cached_prediction",
                "label": "Cached prediction",
                "reason": f"Cached prediction artifact not found: {cached_predictions_path}.",
            }
        return {
            "available": True,
            "mode": "cached_prediction",
            "label": "Cached prediction",
            "reason": "Showing a saved model/pipeline prediction artifact.",
            "artifact_path": str(Path(cached_predictions_path)),
        }
    if mode == "mock":
        return {
            "available": True,
            "mode": "mock_prediction",
            "label": "Mock smoke",
            "reason": "Using deterministic mock prediction for UI smoke testing.",
        }

    if config_demo.get("enable_vlm"):
        api_key_env = str(config_demo.get("vlm_api_key_env", "OPENAI_API_KEY"))
        if not environ.get(api_key_env):
            return {
                "available": False,
                "mode": "retrieval_only",
                "label": "Retrieval-only",
                "reason": f"Missing VLM credential environment variable: {api_key_env}.",
            }

    model_config = config.get("model", {})
    backend = str(model_config.get("backend", "none")).strip().lower()
    if backend in {"", "none", "mock", "fake"}:
        return {
            "available": False,
            "mode": "retrieval_only",
            "label": "Retrieval-only",
            "reason": "No live VLM backend is configured.",
        }

    if backend in {"openai_compatible", "openai-compatible", "openai"}:
        api_key_env = str(model_config.get("api_key_env", "OPENAI_COMPATIBLE_API_KEY"))
        base_url_env = str(
            model_config.get("base_url_env", "OPENAI_COMPATIBLE_BASE_URL")
        )
        api_key = environ.get(api_key_env) or model_config.get("api_key")
        base_url = environ.get(base_url_env) or model_config.get("base_url")
        missing = []
        if not api_key:
            missing.append(api_key_env)
        if not base_url:
            missing.append(base_url_env)
        if missing:
            return {
                "available": False,
                "mode": "retrieval_only",
                "label": "Retrieval-only",
                "reason": "Missing VLM backend environment variable(s): "
                + ", ".join(missing)
                + ".",
            }
        return {
            "available": True,
            "mode": "live_vlm",
            "label": "Live VLM",
            "reason": "Live VLM backend is configured.",
            "backend": backend,
            "model": model_config.get("name") or model_config.get("name_env"),
        }

    return {
        "available": False,
        "mode": "retrieval_only",
        "label": "Retrieval-only",
        "reason": f"Unsupported live VLM backend for demo: {backend}.",
    }


def citation_dict(citation: Citation) -> dict[str, str]:
    return {"law_id": citation.law_id, "article_id": citation.article_id}


def evidence_citations(evidence: list[Evidence]) -> list[dict[str, str]]:
    return [citation_dict(item.to_citation()) for item in evidence]


def evidence_to_demo_item(evidence: Evidence) -> dict[str, Any]:
    return {
        "uid": evidence.uid,
        "law_id": evidence.law_id,
        "article_id": evidence.article_id,
        "title": evidence.title,
        "content": evidence.content,
        "score": evidence.score,
        "rank": evidence.rank,
        "retrieval_method": evidence.retrieval_method.value,
        "citation": citation_dict(evidence.to_citation()),
    }


def query_to_demo_sample(query: Query) -> dict[str, Any]:
    image_path = Path(query.image_path) if query.image_path else None
    return {
        "id": query.id,
        "image_id": query.image_id,
        "image_display_name": image_path.name if image_path else query.image_id,
        "image_exists": bool(image_path and image_path.exists()),
        "question": query.question,
        "question_type": query.question_type.value if query.question_type else None,
        "choices": query.choices,
        "gold_answer_available": query.answer is not None,
        "gold_citation_count": len(query.relevant_articles),
    }


def safe_prediction_citations(citations: Any) -> list[dict[str, str]]:
    if not isinstance(citations, list):
        return []
    safe: list[dict[str, str]] = []
    for citation in citations:
        if isinstance(citation, Citation):
            safe.append(citation_dict(citation))
        elif isinstance(citation, Mapping):
            law_id = citation.get("law_id")
            article_id = citation.get("article_id")
            if law_id and article_id:
                safe.append({"law_id": str(law_id), "article_id": str(article_id)})
    return safe


def prediction_to_demo_item(
    prediction: Prediction | Mapping[str, Any] | None,
    source: str | None = None,
    parse: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    if prediction is None:
        return None
    if isinstance(prediction, Mapping):
        error = prediction.get("error") if isinstance(prediction.get("error"), Mapping) else {}
        return {
            "answer": prediction.get("answer"),
            "citations": safe_prediction_citations(prediction.get("citations")),
            "explanation": prediction.get("explanation") or "",
            "confidence": prediction.get("confidence"),
            "abstained": bool(prediction.get("abstained", False)),
            "source": source,
            "parse": dict(parse or {}),
            "error": dict(error or {}),
        }
    return {
        "answer": prediction.answer,
        "citations": [citation_dict(citation) for citation in prediction.citations],
        "explanation": prediction.explanation,
        "confidence": prediction.confidence,
        "abstained": prediction.abstained,
        "source": source,
        "parse": dict(parse or {}),
        "error": {},
    }


def total_latency_ms(timings_ms: Mapping[str, Any] | None) -> float | None:
    if not isinstance(timings_ms, Mapping):
        return None
    values = [
        float(value)
        for value in timings_ms.values()
        if isinstance(value, int | float) and value >= 0
    ]
    return sum(values) if values else None


def load_cached_prediction_index(path: str | Path) -> dict[str, Mapping[str, Any]]:
    records, invalid_rows = read_prediction_jsonl(path)
    if invalid_rows:
        raise ValueError(f"Cached prediction artifact has invalid rows: {invalid_rows}")

    index: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row_index, record in enumerate(records):
        sample_id = extract_sample_id(record, row_index)
        if sample_id in index:
            duplicates.append(sample_id)
            continue
        index[sample_id] = record
    if duplicates:
        raise ValueError(f"Cached prediction artifact has duplicate sample IDs: {duplicates}")
    return index


def cached_prediction_demo_item(record: Mapping[str, Any]) -> dict[str, Any] | None:
    prediction = record.get("prediction")
    if isinstance(prediction, Mapping):
        return prediction_to_demo_item(
            prediction,
            source="cached_prediction",
            parse=record.get("parse") if isinstance(record.get("parse"), Mapping) else {},
        )
    if "predict" in record:
        return prediction_to_demo_item(
            {
                "answer": record.get("predict"),
                "citations": record.get("predicted_articles", []),
                "explanation": record.get("answer_explanation", ""),
                "confidence": None,
                "abstained": False,
            },
            source="cached_prediction",
        )
    return None


def make_demo_inspection_record(
    sample: dict[str, Any] | Query,
    evidence: list[Evidence],
    retrieval_strategy_name: str,
    top_k: int,
    diagnostics: list[dict[str, Any]] | None = None,
    prediction: Prediction | Mapping[str, Any] | None = None,
    model_status: Mapping[str, Any] | None = None,
    timings_ms: Mapping[str, Any] | None = None,
    prediction_source: str | None = None,
) -> dict[str, Any]:
    query = sample if isinstance(sample, Query) else Query.model_validate(sample)
    image_path = query.image_path
    timings = dict(timings_ms or {})
    parse = prediction.get("parse", {}) if isinstance(prediction, Mapping) else {}
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
        "disclaimer": DEMO_DISCLAIMER,
        "sample": query_to_demo_sample(query),
        "local_image_path": image_path,
        "retrieval": {
            "strategy": retrieval_strategy_name,
            "top_k": top_k,
            "evidence_count": len(evidence),
            "evidence": [evidence_to_demo_item(item) for item in evidence],
            "citation_ids": [item.uid for item in evidence],
            "diagnostics": list(diagnostics or []),
        },
        "prediction": prediction_to_demo_item(
            prediction,
            source=prediction_source,
            parse=parse if isinstance(parse, Mapping) else {},
        ),
        "latency_ms": {
            **timings,
            "total": total_latency_ms(timings),
        },
        "model": dict(model_status or {}),
    }


def default_answer_for_type(question_type: QuestionType | None) -> str:
    if question_type == QuestionType.YES_NO:
        return "Đúng"
    if question_type == QuestionType.FREE_FORM:
        return "Không đủ căn cứ để kết luận."
    return "A"


def mock_prediction(query: Query, evidence: list[Evidence]) -> Prediction:
    """Deterministic smoke predictor; never report this as real model accuracy."""
    abstained = not evidence
    citations = [] if abstained else [evidence[0].to_citation()]
    return Prediction(
        id=query.id,
        question_type=query.question_type,
        answer=default_answer_for_type(query.question_type),
        citations=citations,
        explanation=(
            "Mock smoke prediction; dùng để kiểm tra pipeline, không phải kết quả VLM thật."
            if not abstained
            else "Mock smoke prediction abstained because no evidence was provided."
        ),
        confidence=0.0,
        abstained=abstained,
        raw_response=None,
    )


def disable_model_backend(config: Mapping[str, Any]) -> dict[str, Any]:
    local_config = dict(config)
    local_config["model"] = {
        **dict(config.get("model", {})),
        "backend": "none",
    }
    return local_config


def merge_evidence(
    *groups: list[Evidence],
    top_k: int,
) -> list[Evidence]:
    merged: list[Evidence] = []
    seen: set[str] = set()
    for group in groups:
        for item in group:
            if item.uid in seen:
                continue
            data = item.model_dump(mode="json")
            data["rank"] = len(merged) + 1
            merged.append(Evidence.model_validate(data))
            seen.add(item.uid)
            if len(merged) >= top_k:
                return merged
    return merged


def retrieve_task1_evidence_for_sample(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    runtime: BenchmarkRuntime,
) -> tuple[list[Evidence], list[dict[str, Any]]]:
    article_index = load_processed_law_article_index(dict(config))
    feature_row = runtime.lowcost_feature_extractor().extract_one(sample)
    return retrieve_task1_citation_evidence(
        feature_row,
        config,
        article_index=article_index,
        vector_store=runtime.lowcost_vector_store(),
    )


def retrieve_for_sample(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    runtime: BenchmarkRuntime,
) -> tuple[list[Evidence], list[dict[str, Any]], list[dict[str, Any]]]:
    strategy = retrieval_strategy(config)
    if strategy == "none":
        return [], [], []

    top_k = int(config.get("retrieval", {}).get("top_k", 5))
    if strategy == "text":
        return retrieve_evidence(
            sample,
            dict(config),
            embedder=runtime.text_embedder(),
            vector_store=runtime.law_vector_store(),
            top_k=top_k,
        ), [], []

    if strategy in {"task1", "hybrid"}:
        diagnostics: list[dict[str, Any]] = []
        direct_evidence: list[Evidence] = []
        task1_evidence: list[Evidence] = []

        if strategy == "hybrid":
            direct_evidence = retrieve_evidence(
                sample,
                dict(config),
                embedder=runtime.text_embedder(),
                vector_store=runtime.law_vector_store(),
                top_k=top_k,
            )

        try:
            task1_evidence, task1_diagnostics = retrieve_task1_evidence_for_sample(
                sample,
                config,
                runtime,
            )
            diagnostics.extend(task1_diagnostics)
        except Exception as exc:
            if not bool(config.get("retrieval", {}).get("task1_allow_failure", True)):
                raise
            diagnostics.append(
                {
                    "type": "lowcost_task1_retrieval_failed",
                    "reason": str(exc),
                    "strategy": strategy,
                }
            )

        return merge_evidence(
            direct_evidence,
            task1_evidence,
            top_k=top_k,
        ), [], diagnostics

    retrieval_config = config.get("retrieval", {})
    example_top_k = int(retrieval_config.get("example_top_k", 3))
    example_mode = str(experiment_config(config).get("example_retrieval_mode", "fusion"))
    fused = retrieve_fused_evidence(
        sample,
        dict(config),
        text_embedder=runtime.text_embedder(),
        law_vector_store=runtime.law_vector_store(),
        example_text_embedder=runtime.text_embedder(),
        image_embedder=runtime.image_embedder(),
        example_vector_store=runtime.example_vector_store(),
        top_k=top_k,
        example_top_k=example_top_k,
        example_mode=example_mode,
        allow_example_failure=bool(
            retrieval_config.get("fusion_allow_example_failure", True)
        ),
    )
    return fused.evidence, [], list(fused.diagnostics)


def vlsp_output_path(
    config: Mapping[str, Any],
    set_name: str,
    task: str,
    output_path: str | Path | None = None,
) -> Path:
    if output_path:
        return Path(output_path)
    experiment = experiment_config(config)
    if experiment.get("output_path"):
        return Path(str(experiment["output_path"]))
    return Path(DEFAULT_COMPETITION_DIR) / f"{set_name}_{task}_predictions.jsonl"


def ensure_vlsp_task(task: str) -> str:
    if task not in {"task1", "task2"}:
        raise ValueError("task must be one of: task1, task2")
    return task


def ensure_vlsp_set_name(set_name: str) -> str:
    if set_name not in {"public_test", "private_test"}:
        raise ValueError("set_name must be one of: public_test, private_test")
    return set_name


def validate_vlsp_task2_config(config: Mapping[str, Any]) -> None:
    if is_mock_run(config):
        return
    model_config = config.get("model", {})
    if not bool(model_config.get("include_image", False)):
        raise RuntimeError("VLSP Task 2 real runs require model.include_image=true")


def vlsp_task1_record(
    sample: dict[str, Any],
    evidence: list[Evidence],
    timings_ms: Mapping[str, float],
    config: Mapping[str, Any],
    set_name: str,
    diagnostics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    query = Query.model_validate(sample)
    citations = evidence_citations(evidence)
    return {
        "schema_version": VLSP_TEST_SCHEMA_VERSION,
        "task": "task1",
        "set_name": set_name,
        "id": query.id,
        "image_id": query.image_id,
        "image_path": query.image_path,
        "question": query.question,
        "query": query.model_dump(mode="json", exclude={"answer", "relevant_articles"}),
        "relevant_articles": citations,
        "predicted_articles": citations,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "timings_ms": dict(timings_ms),
        "experiment": {
            "name": experiment_name(config),
            "label": experiment_config(config).get("label"),
            "mock": True,
            "retrieval_strategy": retrieval_strategy(config),
            "prompt_variant": None,
        },
        "mock": True,
        "model": {"backend": "none", "name": None, "include_image": False},
        "retrieval_config": retrieval_run_metadata(config),
        "parse": None,
        "diagnostics": list(diagnostics or []),
        "created_at": utc_now_iso(),
    }


def build_vlsp_task1_record(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    set_name: str,
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    runtime = runtime or BenchmarkRuntime(disable_model_backend(config))
    timings_ms: dict[str, float] = {}
    retrieval_start = time.perf_counter()
    evidence, _, diagnostics = retrieve_for_sample(sample, config, runtime)
    timings_ms["retrieval"] = (time.perf_counter() - retrieval_start) * 1000
    return vlsp_task1_record(
        sample=sample,
        evidence=evidence,
        timings_ms=timings_ms,
        config=config,
        set_name=set_name,
        diagnostics=diagnostics,
    )


def build_vlsp_task2_record(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    set_name: str,
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    runtime_config = disable_model_backend(config) if is_mock_run(config) else dict(config)
    runtime = runtime or BenchmarkRuntime(runtime_config)
    row = build_benchmark_record(sample, config, runtime=runtime)
    row.update(
        {
            "vlsp_schema_version": VLSP_TEST_SCHEMA_VERSION,
            "task": "task2",
            "set_name": set_name,
        }
    )
    return row


def build_vlsp_test_record(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    set_name: str,
    task: str,
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    task = ensure_vlsp_task(task)
    if task == "task1":
        return build_vlsp_task1_record(sample, config, set_name=set_name, runtime=runtime)
    return build_vlsp_task2_record(sample, config, set_name=set_name, runtime=runtime)


def run_vlsp_test(
    config: Mapping[str, Any],
    set_name: str,
    task: str,
    limit: int | None = None,
    output_path: str | Path | None = None,
    runtime: BenchmarkRuntime | None = None,
) -> Path:
    set_name = ensure_vlsp_set_name(set_name)
    task = ensure_vlsp_task(task)
    if task == "task2":
        validate_vlsp_task2_config(config)

    if runtime is None:
        runtime_config = (
            disable_model_backend(config)
            if task == "task1" or (task == "task2" and is_mock_run(config))
            else dict(config)
        )
        runtime = BenchmarkRuntime(runtime_config)

    samples = load_vlsp_samples(config, set_name=set_name, task=task, limit=limit)
    rows = [
        build_vlsp_test_record(
            sample,
            config,
            set_name=set_name,
            task=task,
            runtime=runtime,
        )
        for sample in samples
    ]
    path = vlsp_output_path(config, set_name=set_name, task=task, output_path=output_path)
    write_jsonl(rows, str(path))
    return path


def build_demo_inspection(
    sample_id: str,
    config: Mapping[str, Any],
    split: str = "val",
    top_k: int | None = None,
    retrieval_strategy_name: str = "text",
    include_prediction: bool = False,
    use_mock_prediction: bool = False,
    prediction_mode: str | None = None,
    cached_predictions_path: str | None = None,
    cached_predictions: Mapping[str, Mapping[str, Any]] | None = None,
    environ: Mapping[str, str] | None = None,
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    sample = load_demo_sample_by_id(config, sample_id=sample_id, split=split)
    query = Query.model_validate(sample)
    mode = normalize_demo_prediction_mode(
        prediction_mode,
        include_prediction=include_prediction,
        use_mock_prediction=use_mock_prediction,
    )
    model_status = demo_model_status(
        config,
        include_prediction=include_prediction,
        use_mock_prediction=use_mock_prediction,
        prediction_mode=mode,
        cached_predictions_path=cached_predictions_path,
        environ=environ,
    )
    if runtime is None:
        runtime_config = dict(config)
        if not (mode == "live" and model_status.get("available")):
            runtime_config["model"] = {
                **dict(config.get("model", {})),
                "backend": "none",
            }
        runtime = BenchmarkRuntime(runtime_config)

    local_config = dict(config)
    local_config["experiment"] = {
        **experiment_config(config),
        "retrieval_strategy": retrieval_strategy_name,
    }
    local_config["retrieval"] = dict(config.get("retrieval", {}))
    if top_k is not None:
        local_config["retrieval"]["top_k"] = int(top_k)
    resolved_top_k = int(local_config.get("retrieval", {}).get("top_k", 5))

    timings_ms: dict[str, float] = {}
    retrieval_start = time.perf_counter()
    evidence, _, diagnostics = retrieve_for_sample(sample, local_config, runtime)
    timings_ms["retrieval"] = (time.perf_counter() - retrieval_start) * 1000
    prediction: Prediction | Mapping[str, Any] | None = None
    prediction_source: str | None = None

    if mode == "mock":
        generation_start = time.perf_counter()
        prediction = mock_prediction(query, evidence)
        timings_ms["generation"] = (time.perf_counter() - generation_start) * 1000
        prediction_source = "mock_prediction"
    elif mode == "cached" and model_status.get("available"):
        cache = (
            dict(cached_predictions)
            if cached_predictions is not None
            else load_cached_prediction_index(str(cached_predictions_path))
        )
        cached_record = cache.get(query.id or "")
        if cached_record is None:
            model_status = {
                **model_status,
                "available": False,
                "reason": f"Cached artifact has no prediction for sample {query.id!r}.",
            }
        else:
            prediction = cached_prediction_demo_item(cached_record)
            timings = cached_record.get("timings_ms")
            if isinstance(timings, Mapping):
                for key, value in timings.items():
                    if isinstance(value, int | float):
                        timings_ms[f"cached_{key}"] = float(value)
            prediction_source = "cached_prediction"
    elif mode == "live" and model_status.get("available"):
        generation_start = time.perf_counter()
        try:
            prediction = runtime.vlm.answer(query, evidence)
            prediction_source = "live_vlm"
        except Exception as exc:
            prediction = {
                "answer": None,
                "citations": [],
                "explanation": "Live model call failed; showing retrieval evidence only.",
                "confidence": 0.0,
                "abstained": True,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "parse": parse_error_metadata(exc),
            }
            model_status = {
                **model_status,
                "available": False,
                "reason": f"Live model call failed: {type(exc).__name__}.",
            }
            prediction_source = "live_vlm_error"
        timings_ms["generation"] = (time.perf_counter() - generation_start) * 1000

    return make_demo_inspection_record(
        sample=query,
        evidence=evidence,
        retrieval_strategy_name=retrieval_strategy_name,
        top_k=resolved_top_k,
        diagnostics=diagnostics,
        prediction=prediction,
        model_status=model_status,
        timings_ms=timings_ms,
        prediction_source=prediction_source,
    )


def build_freeform_demo_inspection(
    image_path: str | Path,
    question: str,
    config: Mapping[str, Any],
    top_k: int | None = None,
    retrieval_strategy_name: str = "text",
    prediction_mode: str | None = None,
    include_prediction: bool = False,
    use_mock_prediction: bool = False,
    environ: Mapping[str, str] | None = None,
    runtime: BenchmarkRuntime | None = None,
    query_id: str = "freeform_demo",
    prompt_variant_name: str | None = None,
) -> dict[str, Any]:
    """Run the product-style demo: uploaded image + free-form question."""
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Free-form demo image not found: {image_path}")

    query = Query(
        id=query_id,
        image_id=image_path.stem,
        image_path=str(image_path),
        question=question,
        question_type=QuestionType.FREE_FORM,
    )
    mode = normalize_demo_prediction_mode(
        prediction_mode,
        include_prediction=include_prediction,
        use_mock_prediction=use_mock_prediction,
    )
    model_status = demo_model_status(
        config,
        include_prediction=include_prediction,
        use_mock_prediction=use_mock_prediction,
        prediction_mode=mode,
        environ=environ,
    )
    if runtime is None:
        runtime_config = dict(config)
        if not (mode == "live" and model_status.get("available")):
            runtime_config["model"] = {
                **dict(config.get("model", {})),
                "backend": "none",
            }
        runtime = BenchmarkRuntime(runtime_config)

    local_config = dict(config)
    local_config["experiment"] = {
        **experiment_config(config),
        "retrieval_strategy": retrieval_strategy_name,
        "prompt_variant": prompt_variant_name
        or demo_config(config).get("prompt_variant")
        or PromptVariant.STRUCTURED_LEGAL_RAG.value,
    }
    local_config["retrieval"] = dict(config.get("retrieval", {}))
    if top_k is not None:
        local_config["retrieval"]["top_k"] = int(top_k)
    resolved_top_k = int(local_config.get("retrieval", {}).get("top_k", 5))
    effective_prompt_variant = prompt_variant(local_config)

    timings_ms: dict[str, float] = {}
    retrieval_start = time.perf_counter()
    evidence, _, diagnostics = retrieve_for_sample(
        query.model_dump(mode="json", exclude_none=True),
        local_config,
        runtime,
    )
    timings_ms["retrieval"] = (time.perf_counter() - retrieval_start) * 1000

    prediction: Prediction | Mapping[str, Any] | None = None
    prediction_source: str | None = None
    if mode == "mock":
        generation_start = time.perf_counter()
        prediction = mock_prediction(query, evidence)
        timings_ms["generation"] = (time.perf_counter() - generation_start) * 1000
        prediction_source = "mock_prediction"
    elif mode == "live" and model_status.get("available"):
        generation_start = time.perf_counter()
        try:
            prediction = runtime.vlm.answer(
                query,
                evidence,
                variant=effective_prompt_variant,
            )
            prediction_source = "live_vlm"
        except Exception as exc:
            prediction = {
                "answer": None,
                "citations": [],
                "explanation": "Live model call failed; showing retrieval evidence only.",
                "confidence": 0.0,
                "abstained": True,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
                "parse": parse_error_metadata(exc),
            }
            model_status = {
                **model_status,
                "available": False,
                "reason": f"Live model call failed: {type(exc).__name__}.",
            }
            prediction_source = "live_vlm_error"
        timings_ms["generation"] = (time.perf_counter() - generation_start) * 1000

    return make_demo_inspection_record(
        sample=query,
        evidence=evidence,
        retrieval_strategy_name=retrieval_strategy_name,
        top_k=resolved_top_k,
        diagnostics=diagnostics,
        prediction=prediction,
        model_status=model_status,
        timings_ms=timings_ms,
        prediction_source=prediction_source,
    )


def retrieve_prompt_examples(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    runtime: BenchmarkRuntime,
) -> list[dict[str, Any]]:
    if not use_examples_for_prompt(config):
        return []

    retrieval_config = config.get("retrieval", {})
    top_k = int(config.get("prompt", {}).get("top_examples", 3))
    top_k = int(experiment_config(config).get("example_top_k", top_k))
    if prompt_variant(config) == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT:
        top_k *= max(1, int(retrieval_config.get("example_candidate_multiplier", 5)))
    mode = str(experiment_config(config).get("example_retrieval_mode", "fusion"))
    examples = retrieve_examples(
        sample,
        dict(config),
        mode=mode,
        text_embedder=runtime.text_embedder(),
        image_embedder=runtime.image_embedder(),
        vector_store=runtime.example_vector_store(),
        top_k=top_k,
    )
    return [example.to_prompt_example() for example in examples]


def build_prompt_metadata(
    runtime: LegalQAVLM,
    query: Query,
    evidence: list[Evidence],
    examples: list[dict[str, Any]],
    variant: PromptVariant,
) -> dict[str, Any]:
    messages = runtime.build_messages(
        query,
        evidence,
        examples=examples,
        variant=variant,
    )
    if variant == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT:
        example_count = max(
            0,
            sum(1 for message in messages if message.get("role") == "user") - 1,
        )
    else:
        example_count = len(examples)
    return {
        "variant": variant.value,
        "example_count": example_count,
        "message_hash": stable_json_hash(messages),
    }


def error_benchmark_record(
    query: Query,
    evidence: list[Evidence],
    timings_ms: dict[str, float],
    prompt_metadata: Mapping[str, Any],
    diagnostics: list[dict[str, Any]],
    config: Mapping[str, Any],
    variant: PromptVariant,
    error: Exception,
    model_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    error_type = type(error).__name__
    error_message = str(error)
    diagnostics = [
        *diagnostics,
        {
            "type": "model_error",
            "error_type": error_type,
            "message": error_message,
        },
    ]
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "query": query.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "detected_signs": [],
        "prediction": {
            "id": query.id,
            "question_type": query.question_type.value if query.question_type else None,
            "answer": None,
            "citations": [],
            "explanation": f"Model error: {error_type}: {error_message}",
            "confidence": 0.0,
            "abstained": True,
            "raw_response": None,
            "error": {
                "type": error_type,
                "message": error_message,
            },
        },
        "timings_ms": timings_ms,
        "experiment": {
            "name": experiment_name(config),
            "label": experiment_config(config).get("label"),
            "mock": is_mock_run(config),
            "retrieval_strategy": retrieval_strategy(config),
            "prompt_variant": variant.value,
        },
        "mock": is_mock_run(config),
        "model": dict(model_metadata or model_run_metadata(config)),
        "retrieval_config": retrieval_run_metadata(config),
        "parse": parse_error_metadata(error),
        "predicted_articles": evidence_citations(evidence),
        "prompt": dict(prompt_metadata),
        "diagnostics": diagnostics,
        "created_at": utc_now_iso(),
    }


def build_benchmark_record(
    sample: dict[str, Any],
    config: Mapping[str, Any],
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    runtime = runtime or BenchmarkRuntime(config)
    query = Query.model_validate(sample)
    variant = prompt_variant(config)
    timings_ms: dict[str, float] = {}
    diagnostics: list[dict[str, Any]] = []
    model_metadata = model_run_metadata(config, runtime=runtime)

    retrieval_start = time.perf_counter()
    evidence, _, retrieval_diagnostics = retrieve_for_sample(sample, config, runtime)
    timings_ms["retrieval"] = (time.perf_counter() - retrieval_start) * 1000
    diagnostics.extend(retrieval_diagnostics)

    examples_start = time.perf_counter()
    examples = retrieve_prompt_examples(sample, config, runtime)
    timings_ms["example_retrieval"] = (time.perf_counter() - examples_start) * 1000

    prompt_start = time.perf_counter()
    prompt_metadata = build_prompt_metadata(runtime.vlm, query, evidence, examples, variant)
    timings_ms["prompt"] = (time.perf_counter() - prompt_start) * 1000

    generation_start = time.perf_counter()
    if is_mock_run(config):
        prediction = mock_prediction(query, evidence)
    else:
        try:
            prediction = runtime.vlm.answer(
                query,
                evidence,
                examples=examples,
                variant=variant,
            )
        except Exception as exc:
            timings_ms["generation"] = (time.perf_counter() - generation_start) * 1000
            return error_benchmark_record(
                query=query,
                evidence=evidence,
                timings_ms=timings_ms,
                prompt_metadata=prompt_metadata,
                diagnostics=diagnostics,
                config=config,
                variant=variant,
                error=exc,
                model_metadata=model_metadata,
            )
    timings_ms["generation"] = (time.perf_counter() - generation_start) * 1000

    pipeline_result = PipelineResult(
        query=query,
        evidence=evidence,
        prediction=prediction,
        timings_ms=timings_ms,
    )
    row = pipeline_result.model_dump(mode="json")
    row.update(
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment": {
                "name": experiment_name(config),
                "label": experiment_config(config).get("label"),
                "mock": is_mock_run(config),
                "retrieval_strategy": retrieval_strategy(config),
                "prompt_variant": variant.value,
            },
            "mock": is_mock_run(config),
            "model": model_metadata,
            "retrieval_config": retrieval_run_metadata(config),
            "parse": parse_success_metadata(prediction),
            "predicted_articles": evidence_citations(evidence),
            "prompt": prompt_metadata,
            "diagnostics": diagnostics,
            "created_at": utc_now_iso(),
        }
    )
    return row


def run_benchmark(
    config: Mapping[str, Any],
    limit: int | None = None,
    output_path: str | Path | None = None,
) -> Path:
    samples = load_benchmark_samples(config, limit=limit)
    runtime = BenchmarkRuntime(config)
    rows = [
        build_benchmark_record(sample, config, runtime=runtime)
        for sample in samples
    ]
    path = Path(output_path) if output_path else benchmark_output_path(config)
    write_jsonl(rows, str(path))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experiment pipeline runner for traffic-legal-vlm."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--mode", choices=["benchmark", "vlsp-test"], default="benchmark")
    parser.add_argument("--set-name", choices=["public_test", "private_test"], default="public_test")
    parser.add_argument("--task", choices=["task1", "task2"], default="task2")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.mode == "benchmark":
        try:
            output_path = run_benchmark(config, limit=args.limit, output_path=args.output)
        except RuntimeError as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        print(
            json.dumps(
                {
                    "mode": "benchmark",
                    "config_name": experiment_name(config),
                    "mock": is_mock_run(config),
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.mode == "vlsp-test":
        try:
            output_path = run_vlsp_test(
                config,
                set_name=args.set_name,
                task=args.task,
                limit=args.limit,
                output_path=args.output,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        print(
            json.dumps(
                {
                    "mode": "vlsp-test",
                    "set_name": args.set_name,
                    "task": args.task,
                    "config_name": experiment_name(config),
                    "mock": is_mock_run(config),
                    "output_path": str(output_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
