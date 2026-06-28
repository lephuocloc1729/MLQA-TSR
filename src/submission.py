from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Mapping

from src.evaluate import (
    extract_predicted_citations,
    extract_prediction_answer,
    extract_question_type,
    extract_sample_id,
    normalize_prediction_answer,
    read_prediction_jsonl,
)
from src.utils import load_config, read_json, write_json


VALID_MULTIPLE_CHOICE = {"A", "B", "C", "D"}
VALID_YES_NO = {"Đúng", "Sai"}
SUBMISSION_SCHEMA_VERSION = "submission-task2-v1"
VALIDATION_SCHEMA_VERSION = "submission-validation-v1"


class SubmissionValidationError(ValueError):
    """Raised when predictions are unsafe to package as a submission."""

    def __init__(self, message: str, summary: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.summary = dict(summary)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return unicodedata.normalize("NFC", str(value)).strip()


def load_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not all(isinstance(record, dict) for record in records):
            raise ValueError(f"Every sample in {path} must be a JSON object")
        return [dict(record) for record in records]

    data = read_json(str(path))
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("samples"), list):
        records = data["samples"]
    else:
        raise ValueError(f"Expected a list of sample objects in {path}")

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"Every sample in {path} must be a JSON object")
    return [dict(record) for record in records]


def private_task2_candidates(config: Mapping[str, Any]) -> list[Path]:
    data_config = config.get("data", {})
    candidates: list[Path] = []
    if data_config.get("private_test_task2_path"):
        candidates.append(Path(str(data_config["private_test_task2_path"])))
    private_dir = Path(str(data_config.get("private_test_dir", "data/raw/private_test")))
    candidates.extend(
        [
            private_dir / "Task 2 Submission File" / "vlsp2025_submission_task2.json",
            private_dir / "vlsp2025_submission_task2.json",
            private_dir / "vlsp_2025_private_test_task2.json",
        ]
    )
    return candidates


def resolve_required_samples_path(
    config: Mapping[str, Any],
    set_name: str | None = None,
    explicit_path: str | Path | None = None,
) -> Path | None:
    if explicit_path:
        return Path(explicit_path)

    if not set_name:
        return None

    data_config = config.get("data", {})
    if set_name == "public_test":
        value = data_config.get("public_test_task2_path")
        return Path(str(value)) if value else None
    if set_name == "private_test":
        for candidate in private_task2_candidates(config):
            if candidate.exists():
                return candidate
        return private_task2_candidates(config)[0]
    if set_name == "val":
        value = data_config.get("val_split_path")
        return Path(str(value)) if value else None
    if set_name == "train":
        value = data_config.get("train_split_path")
        return Path(str(value)) if value else None

    raise ValueError("set_name must be one of public_test, private_test, val, train")


def question_type_for_record(
    prediction_record: Mapping[str, Any] | None,
    required_sample: Mapping[str, Any] | None = None,
) -> str:
    if required_sample and required_sample.get("question_type"):
        return normalize_text(required_sample["question_type"])
    if prediction_record:
        return extract_question_type(prediction_record)
    return "Unknown"


def validate_answer(answer: Any, question_type: str, sample_id: str) -> str:
    normalized = normalize_prediction_answer(answer)
    if not normalized:
        raise ValueError(f"{sample_id}: missing prediction answer")

    question_type = normalize_text(question_type)
    if question_type == "Multiple choice":
        if normalized not in VALID_MULTIPLE_CHOICE:
            raise ValueError(
                f"{sample_id}: invalid Multiple choice answer {normalized!r}; "
                "expected A, B, C or D"
            )
        return normalized

    if question_type == "Yes/No":
        if normalized not in VALID_YES_NO:
            raise ValueError(
                f"{sample_id}: invalid Yes/No answer {normalized!r}; "
                "expected Đúng or Sai"
            )
        return normalized

    if normalized not in VALID_MULTIPLE_CHOICE | VALID_YES_NO:
        raise ValueError(
            f"{sample_id}: cannot validate answer {normalized!r} for "
            f"unknown question type {question_type!r}"
        )
    return normalized


def prediction_rows_by_id(
    records: list[Mapping[str, Any]],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]], list[str]]:
    rows_by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[dict[str, str]] = []
    order: list[str] = []
    for index, record in enumerate(records):
        sample_id = extract_sample_id(record, index)
        if sample_id in rows_by_id:
            duplicates.append({"id": sample_id, "reason": "duplicate prediction sample ID"})
            continue
        rows_by_id[sample_id] = record
        order.append(sample_id)
    return rows_by_id, duplicates, order


def sample_ids(samples: list[Mapping[str, Any]]) -> list[str]:
    ids = [normalize_text(sample.get("id")) for sample in samples]
    missing_id_rows = [index + 1 for index, sample_id in enumerate(ids) if not sample_id]
    if missing_id_rows:
        raise ValueError(f"Required sample file has rows without id: {missing_id_rows}")
    return ids


def make_submission_row(
    sample_id: str,
    answer: str,
    template_sample: Mapping[str, Any] | None = None,
    citations: list[Any] | None = None,
    include_citations: bool = False,
) -> dict[str, Any]:
    row = dict(template_sample or {"id": sample_id})
    row["id"] = sample_id
    row["answer"] = answer
    if include_citations:
        row["citations"] = list(citations or [])
    else:
        row.pop("citations", None)
        row.pop("predicted_articles", None)
        row.pop("relevant_articles", None)
    return row


def build_submission(
    records: list[Mapping[str, Any]],
    required_samples: list[Mapping[str, Any]] | None = None,
    allow_missing: bool = False,
    include_citations: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_id, duplicate_errors, prediction_order = prediction_rows_by_id(records)
    required_samples = required_samples or []
    required_ids = sample_ids(required_samples) if required_samples else []
    required_by_id = {normalize_text(sample["id"]): sample for sample in required_samples}
    output_order = required_ids or prediction_order

    missing = [sample_id for sample_id in required_ids if sample_id not in rows_by_id]
    invalid: list[dict[str, str]] = list(duplicate_errors)
    submission_rows: list[dict[str, Any]] = []

    for sample_id in output_order:
        record = rows_by_id.get(sample_id)
        template = required_by_id.get(sample_id)
        if record is None:
            if not allow_missing:
                invalid.append({"id": sample_id, "reason": "missing prediction"})
            continue

        try:
            question_type = question_type_for_record(record, template)
            answer = validate_answer(extract_prediction_answer(record), question_type, sample_id)
        except ValueError as exc:
            invalid.append({"id": sample_id, "reason": str(exc)})
            continue

        submission_rows.append(
            make_submission_row(
                sample_id=sample_id,
                answer=answer,
                template_sample=template,
                citations=extract_predicted_citations(record),
                include_citations=include_citations,
            )
        )

    extra_ids = [
        sample_id for sample_id in prediction_order if required_ids and sample_id not in required_by_id
    ]
    summary = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "submission_schema_version": SUBMISSION_SCHEMA_VERSION,
        "prediction_count": len(records),
        "required_count": len(required_ids) if required_ids else None,
        "output_count": len(submission_rows),
        "missing_count": len(missing),
        "missing_sample_ids": missing,
        "invalid_count": len(invalid),
        "invalid_predictions": invalid,
        "duplicate_count": len(duplicate_errors),
        "extra_count": len(extra_ids),
        "extra_sample_ids": extra_ids,
        "include_citations": include_citations,
        "allow_missing": allow_missing,
    }

    if invalid:
        raise SubmissionValidationError("Submission validation failed", summary)
    return submission_rows, summary


def convert_prediction_file(
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    required_samples_path: str | Path | None = None,
    allow_missing: bool = False,
    include_citations: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    records, invalid_rows = read_prediction_jsonl(predictions_path)
    if invalid_rows:
        summary = {
            "schema_version": VALIDATION_SCHEMA_VERSION,
            "prediction_count": len(records),
            "invalid_count": len(invalid_rows),
            "invalid_predictions": invalid_rows,
        }
        raise SubmissionValidationError("Prediction JSONL contains invalid rows", summary)

    required_samples = (
        load_json_records(required_samples_path) if required_samples_path else None
    )
    rows, summary = build_submission(
        records,
        required_samples=required_samples,
        allow_missing=allow_missing,
        include_citations=include_citations,
    )
    summary.update(
        {
            "predictions_path": str(predictions_path),
            "required_samples_path": str(required_samples_path)
            if required_samples_path
            else None,
            "output_path": str(output_path) if output_path else None,
            "dry_run": dry_run,
        }
    )

    if not dry_run:
        if not output_path:
            raise ValueError("--output is required unless --dry-run is used")
        write_json(rows, str(output_path))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert internal PipelineResult JSONL into VLSP task-2 submission JSON."
    )
    parser.add_argument("--predictions", required=True, help="Internal prediction JSONL")
    parser.add_argument(
        "--output",
        default=None,
        help="Submission JSON path. Required unless --dry-run is set.",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config used to resolve public/private test paths.",
    )
    parser.add_argument(
        "--required-samples",
        default=None,
        help="Optional public/private/test JSON whose IDs must be covered.",
    )
    parser.add_argument(
        "--set-name",
        choices=["public_test", "private_test", "val", "train"],
        default=None,
        help="Resolve required sample IDs from config for a known split.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing required IDs for dry-run diagnostics only.",
    )
    parser.add_argument(
        "--include-citations",
        action="store_true",
        help="Include citations in output. Leave off for the default task-2 answer format.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only and do not write the submission JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.allow_missing and not args.dry_run:
        raise SystemExit("ERROR: --allow-missing is only allowed with --dry-run")

    config = load_config(args.config)
    required_path = resolve_required_samples_path(
        config,
        set_name=args.set_name,
        explicit_path=args.required_samples,
    )
    if required_path is not None and not required_path.exists():
        raise SystemExit(f"ERROR: Required sample file not found: {required_path}")

    try:
        summary = convert_prediction_file(
            predictions_path=args.predictions,
            output_path=args.output,
            required_samples_path=required_path,
            allow_missing=args.allow_missing,
            include_citations=args.include_citations,
            dry_run=args.dry_run,
        )
    except SubmissionValidationError as exc:
        print(json.dumps(exc.summary, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(f"ERROR: {exc}") from None
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    if not args.dry_run and args.output:
        print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
