import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from src.competition_submission import (
    TASK1_FILENAME,
    TASK2_FILENAME,
    SubmissionValidationError,
    build_competition_submission,
    convert_task_predictions,
    pack_submission,
    resolve_task_samples_path,
)
from src.utils import write_json, write_jsonl


LAW_ID = "QCVN 41:2024/BGTVT"


def citation(article_id: str = "22") -> dict:
    return {"law_id": LAW_ID, "article_id": article_id}


def task1_record(sample_id: str = "public_test_1", **overrides) -> dict:
    record = {
        "id": sample_id,
        "image_id": "public_test_2_2",
        "question": "Đây là biển báo gì?",
        "relevant_articles": [citation("B.13")],
        "answer": "A",
    }
    record.update(overrides)
    return record


def task2_record(sample_id: str = "public_test_51", **overrides) -> dict:
    record = {
        "id": sample_id,
        "image_id": "public_test_5_6",
        "question": "Trong hình, xe gắn máy bị cấm vào khung giờ nào?",
        "question_type": "Multiple choice",
        "choices": {
            "A": "23:00 - 04:00",
            "B": "04:00 - 09:00",
            "C": "09:00 - 16:00",
            "D": "16:00 - 23:00",
        },
        "relevant_articles": [citation("22")],
        "answer": "A",
    }
    record.update(overrides)
    return record


def test_valid_task1_conversion_preserves_fields_and_omits_answer():
    rows, summary = build_competition_submission(
        "task1",
        [task1_record()],
        required_samples=[task1_record(answer="ignored")],
    )

    assert summary["invalid_count"] == 0
    assert rows == [
        {
            "id": "public_test_1",
            "image_id": "public_test_2_2",
            "question": "Đây là biển báo gì?",
            "relevant_articles": [citation("B.13")],
        }
    ]
    assert "answer" not in rows[0]


def test_valid_task2_conversion_includes_answer_and_relevant_articles():
    rows, summary = build_competition_submission(
        "task2",
        [task2_record()],
        required_samples=[task2_record(answer="ignored")],
    )

    assert summary["invalid_count"] == 0
    assert rows[0]["answer"] == "A"
    assert rows[0]["relevant_articles"] == [citation("22")]
    assert rows[0]["choices"]["D"] == "16:00 - 23:00"


def test_missing_task1_citation_fails_validation():
    record = task1_record(relevant_articles=[])

    with pytest.raises(SubmissionValidationError) as exc_info:
        build_competition_submission("task1", [record], required_samples=[task1_record()])

    assert exc_info.value.summary["invalid_count"] == 1
    assert "relevant_articles must contain at least one item" in exc_info.value.summary[
        "invalid_predictions"
    ][0]["reason"]


def test_malformed_citation_fails_validation():
    record = task1_record(relevant_articles=[{"law_id": LAW_ID}])

    with pytest.raises(SubmissionValidationError) as exc_info:
        build_competition_submission("task1", [record], required_samples=[task1_record()])

    assert "law_id and article_id" in exc_info.value.summary["invalid_predictions"][0][
        "reason"
    ]


def test_invalid_task2_answer_fails_validation():
    record = task2_record(answer="E")

    with pytest.raises(SubmissionValidationError) as exc_info:
        build_competition_submission("task2", [record], required_samples=[task2_record()])

    assert "invalid Multiple choice answer 'E'" in exc_info.value.summary[
        "invalid_predictions"
    ][0]["reason"]


def test_missing_ids_fail_unless_allow_missing_dry_run(tmp_path):
    predictions_path = tmp_path / "predictions.jsonl"
    required_path = tmp_path / "required.json"
    write_jsonl([task1_record("public_test_1")], predictions_path)
    write_json([task1_record("public_test_1"), task1_record("public_test_2")], required_path)

    with pytest.raises(SubmissionValidationError):
        convert_task_predictions(
            "task1",
            predictions_path=predictions_path,
            required_samples_path=required_path,
            dry_run=True,
        )

    summary = convert_task_predictions(
        "task1",
        predictions_path=predictions_path,
        required_samples_path=required_path,
        allow_missing=True,
        dry_run=True,
    )

    assert summary["missing_sample_ids"] == ["public_test_2"]
    assert summary["invalid_count"] == 0


def test_zip_file_contains_exactly_required_file_names(tmp_path):
    input_dir = tmp_path / "vlsp_private"
    write_json([task1_record()], input_dir / TASK1_FILENAME)
    write_json([task2_record()], input_dir / TASK2_FILENAME)
    write_json({"ignored": True}, input_dir / "notes.json")

    output_path = tmp_path / "submission.zip"
    summary = pack_submission(input_dir, output_path)

    with zipfile.ZipFile(output_path) as archive:
        assert archive.namelist() == [TASK1_FILENAME, TASK2_FILENAME]
    assert summary["zip_entries"] == [TASK1_FILENAME, TASK2_FILENAME]


def test_private_task1_path_resolution_supports_spaces():
    config = {
        "data": {
            "private_test_task1_path": (
                "data/raw/private_test/Task 1 Submission File/"
                "vlsp2025_submission_task1.json"
            )
        }
    }

    assert resolve_task_samples_path(config, "task1", "private_test") == Path(
        "data/raw/private_test/Task 1 Submission File/vlsp2025_submission_task1.json"
    )


def test_allow_missing_cli_requires_dry_run(tmp_path):
    predictions_path = tmp_path / "task1.jsonl"
    write_jsonl([task1_record()], predictions_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.competition_submission",
            "--set-name",
            "public_test",
            "--task",
            "task1",
            "--task1-predictions",
            str(predictions_path),
            "--allow-missing",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "--allow-missing is only accepted with --dry-run" in result.stderr
