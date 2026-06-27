from __future__ import annotations

import argparse
import json
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


def citation_dict(citation: Citation) -> dict[str, str]:
    return {"law_id": citation.law_id, "article_id": citation.article_id}


def evidence_citations(evidence: list[Evidence]) -> list[dict[str, str]]:
    return [citation_dict(item.to_citation()) for item in evidence]


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
        raise RuntimeError(
            "Real VLM generation is not configured in this pipeline runner. "
            "Set experiment.mock=true for a smoke benchmark or inject a VLM backend later."
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
        output_path = run_benchmark(config, limit=args.limit, output_path=args.output)
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
