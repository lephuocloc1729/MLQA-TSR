from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from src.data_utils import load_split_samples
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
DEFAULT_EXPERIMENT_DIR = "data/outputs/experiments"
DEMO_SCHEMA_VERSION = "demo-inspection-v1"


class BenchmarkRuntime:
    """Lazy dependencies reused across samples in one benchmark run."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.vlm = LegalQAVLM(config)
        self._text_embedder: TextEmbedder | None = None
        self._law_vector_store: TextVectorStore | None = None
        self._example_vector_store: ExampleVectorStore | None = None
        self._image_embedder: ImageEmbedder | None = None

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


def demo_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("demo", {}))


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
    allowed = {"none", "text", "fusion"}
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
    return prompt_variant(config) == PromptVariant.FEW_SHOT_RAG


def is_mock_run(config: Mapping[str, Any]) -> bool:
    return bool(experiment_config(config).get("mock", True))


def demo_model_status(
    config: Mapping[str, Any],
    include_prediction: bool = False,
    use_mock_prediction: bool = False,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe whether the demo can produce an answer without exposing secrets."""
    environ = environ or os.environ
    config_demo = demo_config(config)
    api_key_env = str(config_demo.get("vlm_api_key_env", "OPENAI_API_KEY"))

    if not include_prediction:
        return {
            "available": False,
            "mode": "retrieval_only",
            "reason": "Prediction disabled; showing retrieval evidence only.",
        }
    if use_mock_prediction:
        return {
            "available": True,
            "mode": "mock_prediction",
            "reason": "Using deterministic mock prediction for UI smoke testing.",
        }
    if config_demo.get("enable_vlm") and not environ.get(api_key_env):
        return {
            "available": False,
            "mode": "retrieval_only",
            "reason": f"Missing VLM credential environment variable: {api_key_env}.",
        }
    return {
        "available": False,
        "mode": "retrieval_only",
        "reason": "No VLM backend is configured in this demo build.",
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


def prediction_to_demo_item(prediction: Prediction | None) -> dict[str, Any] | None:
    if prediction is None:
        return None
    return {
        "answer": prediction.answer,
        "citations": [citation_dict(citation) for citation in prediction.citations],
        "explanation": prediction.explanation,
        "confidence": prediction.confidence,
        "abstained": prediction.abstained,
    }


def make_demo_inspection_record(
    sample: dict[str, Any] | Query,
    evidence: list[Evidence],
    retrieval_strategy_name: str,
    top_k: int,
    diagnostics: list[dict[str, Any]] | None = None,
    prediction: Prediction | None = None,
    model_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    query = sample if isinstance(sample, Query) else Query.model_validate(sample)
    image_path = query.image_path
    return {
        "schema_version": DEMO_SCHEMA_VERSION,
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
        "prediction": prediction_to_demo_item(prediction),
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


def build_demo_inspection(
    sample_id: str,
    config: Mapping[str, Any],
    split: str = "val",
    top_k: int | None = None,
    retrieval_strategy_name: str = "text",
    include_prediction: bool = False,
    use_mock_prediction: bool = False,
    runtime: BenchmarkRuntime | None = None,
) -> dict[str, Any]:
    runtime = runtime or BenchmarkRuntime(config)
    sample = load_demo_sample_by_id(config, sample_id=sample_id, split=split)
    query = Query.model_validate(sample)
    local_config = dict(config)
    local_config["experiment"] = {
        **experiment_config(config),
        "retrieval_strategy": retrieval_strategy_name,
    }
    local_config["retrieval"] = dict(config.get("retrieval", {}))
    if top_k is not None:
        local_config["retrieval"]["top_k"] = int(top_k)
    resolved_top_k = int(local_config.get("retrieval", {}).get("top_k", 5))

    evidence, _, diagnostics = retrieve_for_sample(sample, local_config, runtime)
    model_status = demo_model_status(
        config,
        include_prediction=include_prediction,
        use_mock_prediction=use_mock_prediction,
    )
    prediction = (
        mock_prediction(query, evidence)
        if include_prediction and use_mock_prediction
        else None
    )
    return make_demo_inspection_record(
        sample=query,
        evidence=evidence,
        retrieval_strategy_name=retrieval_strategy_name,
        top_k=resolved_top_k,
        diagnostics=diagnostics,
        prediction=prediction,
        model_status=model_status,
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
    return {
        "variant": variant.value,
        "example_count": len(examples),
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
    parser.add_argument("--mode", choices=["benchmark"], default="benchmark")
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


if __name__ == "__main__":
    main()
