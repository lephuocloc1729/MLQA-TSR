from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.utils import load_config, read_json, write_json


VALID_MULTIPLE_CHOICE_ANSWERS = {"A", "B", "C", "D"}
VALID_YES_NO_ANSWERS = {"Đúng", "Sai"}
UNKNOWN_QUESTION_TYPE = "Unknown"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def normalize_gold_answer(value: Any) -> str:
    """Normalize benchmark labels, including known VLSP legacy gold values."""
    if value == 40 or value == "40":
        return "A"

    answer = normalize_text(value)
    yes_no_mapping = {
        "yes": "Đúng",
        "true": "Đúng",
        "no": "Sai",
        "false": "Sai",
        "không": "Sai",
        "fail": "Sai",
    }
    return yes_no_mapping.get(answer.casefold(), answer)


def normalize_prediction_answer(value: Any) -> str:
    """Normalize prediction text without repairing invalid answer labels."""
    return normalize_text(value)


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    if not 0 <= percentile_value <= 100:
        raise ValueError("percentile_value must be between 0 and 100")

    ordered = sorted(values)
    index = math.ceil((percentile_value / 100) * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]


def latency_summary(latencies_ms: list[float]) -> dict[str, float | int | None]:
    ordered = sorted(latencies_ms)
    return {
        "count": len(ordered),
        "mean": mean(ordered) if ordered else None,
        "p50": percentile(ordered, 50),
        "p95": percentile(ordered, 95),
        "max": max(ordered) if ordered else None,
    }


def as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def normalize_question_type(value: Any) -> str:
    return normalize_text(value) or UNKNOWN_QUESTION_TYPE


def citation_uid(citation: Any) -> str | None:
    citation = as_mapping(citation)
    uid = normalize_text(citation.get("uid"))
    if uid and "#" in uid:
        return uid

    law_id = normalize_text(citation.get("law_id"))
    article_id = normalize_text(citation.get("article_id"))
    if not law_id or not article_id:
        return None
    return f"{law_id}#{article_id}"


def citation_uid_set(citations: Iterable[Any]) -> set[str]:
    return {uid for citation in citations if (uid := citation_uid(citation))}


def score_retrieval_sample(
    gold_citations: Iterable[Any],
    predicted_citations: Iterable[Any],
) -> dict[str, float | int]:
    """Return official-style Precision/Recall/F2 for one sample."""
    gold_uids = citation_uid_set(gold_citations)
    predicted_uids = citation_uid_set(predicted_citations)

    if not gold_uids and not predicted_uids:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f2": 1.0,
            "gold_count": 0,
            "predicted_count": 0,
            "hit_count": 0,
        }

    hit_count = len(gold_uids & predicted_uids)
    precision = hit_count / len(predicted_uids) if predicted_uids else 0.0
    recall = hit_count / len(gold_uids) if gold_uids else 0.0
    f2 = (
        5 * precision * recall / (4 * precision + recall)
        if precision + recall > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "gold_count": len(gold_uids),
        "predicted_count": len(predicted_uids),
        "hit_count": hit_count,
    }


def extract_sample_id(record: Mapping[str, Any], index: int) -> str:
    query = as_mapping(record.get("query"))
    prediction = as_mapping(record.get("prediction"))
    for value in (
        record.get("id"),
        record.get("sample_id"),
        query.get("id"),
        prediction.get("id"),
    ):
        sample_id = normalize_text(value)
        if sample_id:
            return sample_id
    return f"row_{index + 1}"


def extract_question_type(record: Mapping[str, Any]) -> str:
    query = as_mapping(record.get("query"))
    prediction = as_mapping(record.get("prediction"))
    for value in (
        query.get("question_type"),
        record.get("question_type"),
        prediction.get("question_type"),
    ):
        question_type = normalize_question_type(value)
        if question_type != UNKNOWN_QUESTION_TYPE:
            return question_type
    return UNKNOWN_QUESTION_TYPE


def extract_gold_answer(record: Mapping[str, Any]) -> Any:
    query = as_mapping(record.get("query"))
    for value in (query.get("answer"), record.get("answer"), record.get("gold_answer")):
        if value is not None:
            return value
    return None


def extract_prediction_answer(record: Mapping[str, Any]) -> Any:
    prediction = record.get("prediction")
    if isinstance(prediction, Mapping):
        if "answer" in prediction:
            return prediction["answer"]
    elif prediction is not None:
        return prediction

    for key in ("predict", "predicted_answer", "prediction_answer"):
        if key in record:
            return record[key]
    return None


def extract_gold_citations(record: Mapping[str, Any]) -> list[Any]:
    query = as_mapping(record.get("query"))
    for value in (
        query.get("relevant_articles"),
        record.get("relevant_articles"),
        record.get("gold_citations"),
    ):
        if isinstance(value, list):
            return value
    return []


def extract_predicted_citations(record: Mapping[str, Any]) -> list[Any]:
    prediction = as_mapping(record.get("prediction"))
    for value in (
        record.get("predicted_articles"),
        record.get("predicted_citations"),
        record.get("citations"),
        record.get("evidence"),
        prediction.get("citations"),
    ):
        if isinstance(value, list):
            return value
    return []


def extract_total_latency_ms(record: Mapping[str, Any]) -> float | None:
    timings = record.get("timings_ms")
    if isinstance(timings, Mapping):
        values = [
            float(value)
            for value in timings.values()
            if isinstance(value, int | float) and value >= 0
        ]
        return sum(values) if values else None
    if isinstance(timings, int | float) and timings >= 0:
        return float(timings)

    for key in ("latency_ms", "total_latency_ms", "time_ms"):
        value = record.get(key)
        if isinstance(value, int | float) and value >= 0:
            return float(value)

    value = record.get("time_second")
    if isinstance(value, int | float) and value >= 0:
        return float(value) * 1000
    return None


def first_record_experiment(records: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    for record in records:
        experiment = as_mapping(record.get("experiment"))
        if experiment:
            return experiment
    return {}


def is_valid_prediction_answer(answer: str, question_type: str) -> bool:
    if not answer:
        return False
    if question_type == "Multiple choice":
        return answer in VALID_MULTIPLE_CHOICE_ANSWERS
    if question_type == "Yes/No":
        return answer in VALID_YES_NO_ANSWERS
    return True


def evaluate_retrieval(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    sample_scores = [
        score_retrieval_sample(
            extract_gold_citations(record),
            extract_predicted_citations(record),
        )
        for record in records
    ]
    return {
        "sample_count": len(sample_scores),
        "precision": mean(score["precision"] for score in sample_scores),
        "recall": mean(score["recall"] for score in sample_scores),
        "f2": mean(score["f2"] for score in sample_scores),
    }


def evaluate_qa(
    records: list[Mapping[str, Any]],
    invalid_predictions: list[dict[str, str]],
) -> dict[str, Any]:
    total = 0
    correct = 0
    skipped_no_gold = 0
    question_type_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    gold_answer_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"correct": 0, "total": 0}
    )
    prediction_distribution: Counter[str] = Counter()

    for index, record in enumerate(records):
        sample_id = extract_sample_id(record, index)
        question_type = extract_question_type(record)
        raw_gold_answer = extract_gold_answer(record)
        if raw_gold_answer is None:
            skipped_no_gold += 1
            continue

        total += 1
        gold_answer = normalize_gold_answer(raw_gold_answer)
        raw_prediction_answer = extract_prediction_answer(record)
        predicted_answer = normalize_prediction_answer(raw_prediction_answer)
        question_type_stats[question_type]["total"] += 1
        gold_answer_stats[gold_answer]["total"] += 1

        if raw_prediction_answer is None:
            invalid_predictions.append(
                {"id": sample_id, "reason": "missing prediction answer"}
            )
            prediction_distribution["__MISSING__"] += 1
            continue

        if not is_valid_prediction_answer(predicted_answer, question_type):
            invalid_predictions.append(
                {
                    "id": sample_id,
                    "reason": (
                        f"invalid {question_type} prediction answer: "
                        f"{predicted_answer!r}"
                    ),
                }
            )
            prediction_distribution["__INVALID__"] += 1
            continue

        prediction_distribution[predicted_answer] += 1
        is_correct = predicted_answer == gold_answer
        if is_correct:
            correct += 1
            question_type_stats[question_type]["correct"] += 1
            gold_answer_stats[gold_answer]["correct"] += 1

    return {
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
        "skipped_no_gold": skipped_no_gold,
        "by_question_type": {
            question_type: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for question_type, stats in sorted(question_type_stats.items())
        },
        "prediction_distribution": dict(sorted(prediction_distribution.items())),
        "gold_answer_breakdown": {
            answer: {
                "accuracy": stats["correct"] / stats["total"] if stats["total"] else 0.0,
                "correct": stats["correct"],
                "total": stats["total"],
            }
            for answer, stats in sorted(gold_answer_stats.items())
        },
    }


def read_prediction_jsonl(path: str | Path) -> tuple[list[Mapping[str, Any]], list[dict[str, str]]]:
    records: list[Mapping[str, Any]] = []
    invalid_rows: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                invalid_rows.append(
                    {"id": f"line_{line_number}", "reason": f"malformed JSON: {exc.msg}"}
                )
                continue
            if not isinstance(record, Mapping):
                invalid_rows.append(
                    {"id": f"line_{line_number}", "reason": "prediction row is not an object"}
                )
                continue
            records.append(record)
    return records, invalid_rows


def build_split_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    split_manifest_path = (
        as_mapping(config.get("data")).get("split_manifest_path") if config else None
    )
    if not split_manifest_path:
        return {}

    path = Path(str(split_manifest_path))
    metadata: dict[str, Any] = {"manifest_path": str(path)}
    if not path.exists():
        metadata["available"] = False
        return metadata

    manifest = read_json(str(path))
    metadata.update(
        {
            "available": True,
            "manifest_sha256": file_sha256(path),
            "split_hash": manifest.get("split_hash"),
            "train_count": manifest.get("train_count"),
            "val_count": manifest.get("val_count"),
        }
    )
    return metadata


def build_evaluation_artifact(
    records: list[Mapping[str, Any]],
    invalid_rows: list[dict[str, str]] | None = None,
    config: Mapping[str, Any] | None = None,
    predictions_path: str | Path | None = None,
) -> dict[str, Any]:
    config = config or {}
    invalid_predictions = list(invalid_rows or [])
    latencies = [
        latency
        for record in records
        if (latency := extract_total_latency_ms(record)) is not None
    ]

    retrieval = evaluate_retrieval(records)
    qa = evaluate_qa(records, invalid_predictions)
    project_config = as_mapping(config.get("project"))
    experiment_config = as_mapping(config.get("experiment"))
    record_experiment = first_record_experiment(records)
    path = Path(predictions_path) if predictions_path else None
    resolved_experiment_name = (
        experiment_config.get("name")
        or record_experiment.get("name")
        or project_config.get("name")
    )

    return {
        "schema_version": "w2-ablation-metrics-v1",
        "config_name": resolved_experiment_name,
        "seed": project_config.get("seed"),
        "experiment": {
            "name": resolved_experiment_name,
            "label": experiment_config.get("label") or record_experiment.get("label"),
            "mock": experiment_config.get("mock", record_experiment.get("mock")),
            "retrieval_strategy": experiment_config.get("retrieval_strategy")
            or record_experiment.get("retrieval_strategy"),
            "prompt_variant": experiment_config.get("prompt_variant")
            or record_experiment.get("prompt_variant")
            or as_mapping(config.get("prompt")).get("variant"),
            "output_path": experiment_config.get("output_path"),
        },
        "mock": experiment_config.get("mock", record_experiment.get("mock")),
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "predictions_path": str(path) if path else None,
        "predictions_sha256": file_sha256(path) if path and path.exists() else None,
        "split": build_split_metadata(config),
        "sample_count": len(records),
        "latency_ms": latency_summary(latencies),
        "retrieval": retrieval,
        "qa": qa,
        "invalid_prediction_count": len(invalid_predictions),
        "invalid_predictions": invalid_predictions,
        "failed_sample_ids": sorted(
            {
                invalid_prediction["id"]
                for invalid_prediction in invalid_predictions
                if invalid_prediction.get("id")
            }
        ),
    }


def default_output_path(config: Mapping[str, Any], predictions_path: str | Path) -> Path:
    experiment_config = as_mapping(config.get("experiment"))
    if experiment_config.get("metrics_path"):
        return Path(str(experiment_config["metrics_path"]))
    path = Path(predictions_path)
    return path.with_name(f"{path.stem}_metrics.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval and QA predictions for traffic-legal-vlm."
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="JSONL file containing PipelineResult-style or flat prediction rows.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Project config path used for seed/output/split metadata.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the metrics artifact. Defaults to <predictions_stem>_metrics.json beside the predictions file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    records, invalid_rows = read_prediction_jsonl(args.predictions)
    artifact = build_evaluation_artifact(
        records,
        invalid_rows=invalid_rows,
        config=config,
        predictions_path=args.predictions,
    )
    output_path = Path(args.output) if args.output else default_output_path(
        config,
        args.predictions,
    )
    write_json(artifact, str(output_path))
    print(json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved metrics to {output_path}")


if __name__ == "__main__":
    main()
