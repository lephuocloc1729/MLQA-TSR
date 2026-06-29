from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.evaluate import (
    as_mapping,
    extract_predicted_citations,
    extract_prediction_answer,
    extract_sample_id,
    read_prediction_jsonl,
)
from src.submission import (
    SubmissionValidationError,
    load_json_records,
    normalize_text,
    prediction_rows_by_id,
    private_task2_candidates,
    question_type_for_record,
    sample_ids,
    validate_answer,
)
from src.utils import load_config, write_json


COMPETITION_SCHEMA_VERSION = "vlsp-competition-submission-v1"
TASK1_FILENAME = "submission_task1.json"
TASK2_FILENAME = "submission_task2.json"
ZIP_FILENAMES = (TASK1_FILENAME, TASK2_FILENAME)
VALID_TASKS = {"task1", "task2"}


def private_task1_candidates(config: Mapping[str, Any]) -> list[Path]:
    data_config = config.get("data", {})
    candidates: list[Path] = []
    if data_config.get("private_test_task1_path"):
        candidates.append(Path(str(data_config["private_test_task1_path"])))
    private_dir = Path(str(data_config.get("private_test_dir", "data/raw/private_test")))
    candidates.extend(
        [
            private_dir / "Task 1 Submission File" / "vlsp2025_submission_task1.json",
            private_dir / "vlsp2025_submission_task1.json",
            private_dir / "vlsp_2025_private_test_task1.json",
        ]
    )
    return candidates


def resolve_task_samples_path(
    config: Mapping[str, Any],
    task: str,
    set_name: str,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)

    if task not in VALID_TASKS:
        raise ValueError("task must be one of task1 or task2")

    data_config = config.get("data", {})
    if set_name == "public_test":
        key = "public_test_task1_path" if task == "task1" else "public_test_task2_path"
        value = data_config.get(key)
        if not value:
            raise ValueError(f"Missing data.{key} in config")
        return Path(str(value))

    if set_name == "private_test":
        candidates = (
            private_task1_candidates(config)
            if task == "task1"
            else private_task2_candidates(config)
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    raise ValueError("set_name must be one of public_test or private_test")


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix == ".jsonl":
        records, invalid_rows = read_prediction_jsonl(path)
        if invalid_rows:
            summary = {
                "schema_version": COMPETITION_SCHEMA_VERSION,
                "prediction_count": len(records),
                "invalid_count": len(invalid_rows),
                "invalid_predictions": invalid_rows,
            }
            raise SubmissionValidationError("Prediction JSONL contains invalid rows", summary)
        return [dict(record) for record in records]
    return load_json_records(path)


def _citation_from_uid(uid: str) -> tuple[str, str] | None:
    if "#" not in uid:
        return None
    law_id, article_id = uid.split("#", 1)
    law_id = normalize_text(law_id)
    article_id = normalize_text(article_id)
    if not law_id or not article_id:
        return None
    return law_id, article_id


def normalize_citation(citation: Any, sample_id: str) -> dict[str, str]:
    if not isinstance(citation, Mapping):
        raise ValueError(f"{sample_id}: citation must be an object")

    law_id = normalize_text(citation.get("law_id"))
    article_id = normalize_text(citation.get("article_id"))
    if not law_id or not article_id:
        uid_parts = _citation_from_uid(normalize_text(citation.get("uid")))
        if uid_parts:
            law_id, article_id = uid_parts

    if not law_id or not article_id:
        raise ValueError(
            f"{sample_id}: citation must include non-empty law_id and article_id"
        )
    return {"law_id": law_id, "article_id": article_id}


def normalize_citations(citations: Any, sample_id: str) -> list[dict[str, str]]:
    if not isinstance(citations, list):
        raise ValueError(f"{sample_id}: relevant_articles must be a list")

    normalized: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for citation in citations:
        item = normalize_citation(citation, sample_id)
        key = (item["law_id"], item["article_id"])
        if key not in seen:
            normalized.append(item)
            seen.add(key)

    if not normalized:
        raise ValueError(f"{sample_id}: relevant_articles must contain at least one item")
    return normalized


def extract_submission_citations(record: Mapping[str, Any]) -> list[Any]:
    citations = extract_predicted_citations(record)
    if citations:
        return citations

    prediction = as_mapping(record.get("prediction"))
    if isinstance(prediction.get("relevant_articles"), list):
        return list(prediction["relevant_articles"])

    # Direct submission-shaped prediction files often use relevant_articles as
    # the prediction field. PipelineResult rows keep gold labels under query, so
    # we only use the top-level field when the row is not a PipelineResult.
    if "query" not in record and isinstance(record.get("relevant_articles"), list):
        return list(record["relevant_articles"])

    return []


def extract_submission_answer(record: Mapping[str, Any]) -> Any:
    answer = extract_prediction_answer(record)
    if answer is not None:
        return answer

    direct_prediction_keys = {
        "prediction",
        "predict",
        "predicted_answer",
        "prediction_answer",
        "query",
    }
    if "answer" in record and not any(key in record for key in direct_prediction_keys):
        return record["answer"]
    return None


def _query_mapping(record: Mapping[str, Any]) -> Mapping[str, Any]:
    return as_mapping(record.get("query"))


def first_present(
    key: str,
    template: Mapping[str, Any] | None,
    record: Mapping[str, Any],
) -> Any:
    query = _query_mapping(record)
    for source in (template, record, query):
        if source and key in source and source[key] not in (None, ""):
            return source[key]
    return None


def make_task1_row(
    sample_id: str,
    citations: list[dict[str, str]],
    record: Mapping[str, Any],
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": sample_id}
    for key in ("image_id", "question"):
        value = first_present(key, template, record)
        if value is not None:
            row[key] = value
    row["relevant_articles"] = citations
    return row


def make_task2_row(
    sample_id: str,
    answer: str,
    citations: list[dict[str, str]],
    record: Mapping[str, Any],
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"id": sample_id}
    for key in ("image_id", "question", "question_type", "choices"):
        value = first_present(key, template, record)
        if value is not None and not (key == "choices" and value == {}):
            row[key] = value
    row["relevant_articles"] = citations
    row["answer"] = answer
    return row


def build_competition_submission(
    task: str,
    records: list[Mapping[str, Any]],
    required_samples: list[Mapping[str, Any]] | None = None,
    allow_missing: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if task not in VALID_TASKS:
        raise ValueError("task must be one of task1 or task2")

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
            citations = normalize_citations(extract_submission_citations(record), sample_id)
            if task == "task1":
                submission_rows.append(
                    make_task1_row(sample_id, citations, record, template)
                )
                continue

            question_type = question_type_for_record(record, template)
            answer = validate_answer(extract_submission_answer(record), question_type, sample_id)
            submission_rows.append(
                make_task2_row(sample_id, answer, citations, record, template)
            )
        except ValueError as exc:
            invalid.append({"id": sample_id, "reason": str(exc)})

    extra_ids = [
        sample_id for sample_id in prediction_order if required_ids and sample_id not in required_by_id
    ]
    summary = {
        "schema_version": COMPETITION_SCHEMA_VERSION,
        "task": task,
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
        "allow_missing": allow_missing,
    }

    if invalid:
        raise SubmissionValidationError(f"{task} submission validation failed", summary)
    return submission_rows, summary


def convert_task_predictions(
    task: str,
    predictions_path: str | Path,
    output_path: str | Path | None = None,
    required_samples_path: str | Path | None = None,
    allow_missing: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    records = load_prediction_records(predictions_path)
    required_samples = (
        load_json_records(required_samples_path) if required_samples_path else None
    )
    rows, summary = build_competition_submission(
        task=task,
        records=records,
        required_samples=required_samples,
        allow_missing=allow_missing,
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
            raise ValueError("output_path is required unless dry_run is true")
        write_json(rows, str(output_path))
    return summary


def pack_submission(input_dir: str | Path, output_path: str | Path) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_path = Path(output_path)
    missing = [filename for filename in ZIP_FILENAMES if not (input_dir / filename).exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot create submission zip; missing required file(s): {', '.join(missing)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in ZIP_FILENAMES:
            archive.write(input_dir / filename, arcname=filename)

    with zipfile.ZipFile(output_path) as archive:
        entries = archive.namelist()
    if entries != list(ZIP_FILENAMES):
        raise ValueError(f"submission.zip contains unexpected entries: {entries}")

    return {
        "schema_version": COMPETITION_SCHEMA_VERSION,
        "mode": "pack",
        "input_dir": str(input_dir),
        "output_path": str(output_path),
        "zip_entries": entries,
        "zip_entry_count": len(entries),
    }


def tasks_from_cli(value: str) -> list[str]:
    if value == "both":
        return ["task1", "task2"]
    if value not in VALID_TASKS:
        raise ValueError("--task must be one of task1, task2, both")
    return [value]


def ensure_allow_missing_is_dry_run(allow_missing: bool, dry_run: bool) -> None:
    if allow_missing and not dry_run:
        raise ValueError("--allow-missing is only accepted with --dry-run")


def output_path_for_task(output_dir: str | Path | None, task: str, dry_run: bool) -> Path | None:
    if dry_run:
        return None
    if not output_dir:
        raise ValueError("--output-dir is required unless --dry-run is set")
    filename = TASK1_FILENAME if task == "task1" else TASK2_FILENAME
    return Path(output_dir) / filename


def prediction_path_for_task(args: argparse.Namespace, task: str) -> str | None:
    return args.task1_predictions if task == "task1" else args.task2_predictions


def explicit_required_path_for_task(args: argparse.Namespace, task: str) -> str | None:
    return args.task1_required_samples if task == "task1" else args.task2_required_samples


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build exact VLSP post-submission files: submission_task1.json, "
            "submission_task2.json, and submission.zip."
        )
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config used to resolve public/private VLSP input paths.",
    )
    parser.add_argument(
        "--set-name",
        choices=["public_test", "private_test"],
        default="public_test",
        help="VLSP set whose IDs and template fields must be preserved.",
    )
    parser.add_argument(
        "--task",
        choices=["task1", "task2", "both"],
        default="both",
        help="Task file(s) to validate and write.",
    )
    parser.add_argument("--task1-predictions", help="Task 1 prediction JSON/JSONL")
    parser.add_argument("--task2-predictions", help="Task 2 prediction JSON/JSONL")
    parser.add_argument(
        "--task1-required-samples",
        default=None,
        help="Optional Task 1 template/required-sample JSON override.",
    )
    parser.add_argument(
        "--task2-required-samples",
        default=None,
        help="Optional Task 2 template/required-sample JSON override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where submission_task1.json/submission_task2.json are written.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Allow missing required IDs for dry-run diagnostics only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate only and do not write task JSON files.",
    )
    parser.add_argument(
        "--pack",
        default=None,
        help="Pack an existing directory containing submission_task1.json and submission_task2.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Zip output path used with --pack.",
    )
    return parser.parse_args(argv)


def run_conversion(args: argparse.Namespace) -> dict[str, Any]:
    ensure_allow_missing_is_dry_run(args.allow_missing, args.dry_run)
    config = load_config(args.config)

    summaries: dict[str, Any] = {
        "schema_version": COMPETITION_SCHEMA_VERSION,
        "mode": "convert",
        "set_name": args.set_name,
        "dry_run": args.dry_run,
        "tasks": {},
    }
    for task in tasks_from_cli(args.task):
        predictions_path = prediction_path_for_task(args, task)
        if not predictions_path:
            raise ValueError(f"--{task}-predictions is required for --task {args.task}")

        required_path = resolve_task_samples_path(
            config=config,
            task=task,
            set_name=args.set_name,
            explicit_path=explicit_required_path_for_task(args, task),
        )
        if not required_path.exists():
            raise FileNotFoundError(f"Required sample file not found: {required_path}")

        output_path = output_path_for_task(args.output_dir, task, args.dry_run)
        summaries["tasks"][task] = convert_task_predictions(
            task=task,
            predictions_path=predictions_path,
            output_path=output_path,
            required_samples_path=required_path,
            allow_missing=args.allow_missing,
            dry_run=args.dry_run,
        )

    return summaries


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        if args.pack:
            if not args.output:
                raise ValueError("--output is required with --pack")
            summary = pack_submission(args.pack, args.output)
        else:
            summary = run_conversion(args)
    except SubmissionValidationError as exc:
        print(json.dumps(exc.summary, ensure_ascii=False, indent=2, sort_keys=True))
        raise SystemExit(f"ERROR: {exc}") from None
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from None

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
