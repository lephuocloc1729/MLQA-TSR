import json
from pathlib import Path

import pytest

from src.submission import (
    SubmissionValidationError,
    build_submission,
    convert_prediction_file,
    load_json_records,
    resolve_required_samples_path,
)
from src.utils import write_json, write_jsonl


LAW_ID = "QCVN 41:2024/BGTVT"


def pipeline_record(sample_id: str = "s1", **overrides) -> dict:
    record = {
        "query": {
            "id": sample_id,
            "image_id": "img_1",
            "question": "Biển báo này có ý nghĩa gì?",
            "question_type": "Multiple choice",
            "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        },
        "prediction": {
            "id": sample_id,
            "question_type": "Multiple choice",
            "answer": "B",
            "citations": [{"law_id": LAW_ID, "article_id": "22"}],
            "explanation": "Dựa trên Điều 22.",
        },
        "predicted_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    record.update(overrides)
    return record


def required_sample(sample_id: str = "s1", **overrides) -> dict:
    sample = {
        "id": sample_id,
        "image_id": "img_1",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    sample.update(overrides)
    return sample


def test_valid_multiple_choice_conversion_preserves_sample_id():
    rows, summary = build_submission(
        [pipeline_record()],
        required_samples=[required_sample()],
    )

    assert summary["invalid_count"] == 0
    assert summary["required_count"] == 1
    assert rows == [
        {
            "id": "s1",
            "image_id": "img_1",
            "question": "Biển báo này có ý nghĩa gì?",
            "question_type": "Multiple choice",
            "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
            "answer": "B",
        }
    ]


def test_valid_yes_no_conversion_normalizes_nfc_answer():
    record = pipeline_record(
        "yn_1",
        query={
            "id": "yn_1",
            "image_id": "img_2",
            "question": "Đúng hay sai?",
            "question_type": "Yes/No",
        },
        prediction={
            "id": "yn_1",
            "question_type": "Yes/No",
            "answer": "Đúng",
            "citations": [{"law_id": LAW_ID, "article_id": "41"}],
            "explanation": "Dựa trên Điều 41.",
        },
    )

    rows, summary = build_submission(
        [record],
        required_samples=[
            required_sample(
                "yn_1",
                question_type="Yes/No",
                choices={},
                relevant_articles=[{"law_id": LAW_ID, "article_id": "41"}],
            )
        ],
    )

    assert summary["invalid_count"] == 0
    assert rows[0]["id"] == "yn_1"
    assert rows[0]["answer"] == "Đúng"


def test_missing_sample_id_fails_validation():
    with pytest.raises(SubmissionValidationError) as exc_info:
        build_submission(
            [pipeline_record("s1")],
            required_samples=[required_sample("s1"), required_sample("s2")],
        )

    summary = exc_info.value.summary
    assert summary["missing_sample_ids"] == ["s2"]
    assert summary["invalid_predictions"] == [
        {"id": "s2", "reason": "missing prediction"}
    ]


def test_allow_missing_dry_run_reports_missing_without_failing(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    required_path = tmp_path / "required.json"
    write_jsonl([pipeline_record("s1")], prediction_path)
    write_json([required_sample("s1"), required_sample("s2")], str(required_path))

    summary = convert_prediction_file(
        prediction_path,
        required_samples_path=required_path,
        allow_missing=True,
        dry_run=True,
    )

    assert summary["dry_run"] is True
    assert summary["missing_sample_ids"] == ["s2"]
    assert summary["invalid_count"] == 0
    assert not (tmp_path / "submission.json").exists()


def test_invalid_answer_fails_without_silent_replacement():
    record = pipeline_record(
        prediction={
            "id": "s1",
            "question_type": "Multiple choice",
            "answer": "E",
            "citations": [{"law_id": LAW_ID, "article_id": "22"}],
            "explanation": "Invalid label.",
        }
    )

    with pytest.raises(SubmissionValidationError) as exc_info:
        build_submission([record], required_samples=[required_sample()])

    assert exc_info.value.summary["invalid_count"] == 1
    assert "invalid Multiple choice answer 'E'" in exc_info.value.summary[
        "invalid_predictions"
    ][0]["reason"]


def test_citations_remain_internal_and_are_omitted_by_default(tmp_path):
    prediction_path = tmp_path / "predictions.jsonl"
    output_path = tmp_path / "submission.json"
    original_record = pipeline_record()
    write_jsonl([original_record], prediction_path)

    summary = convert_prediction_file(prediction_path, output_path=output_path)

    output_rows = json.loads(output_path.read_text(encoding="utf-8"))
    input_rows = [
        json.loads(line)
        for line in prediction_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["output_count"] == 1
    assert output_rows == [{"id": "s1", "answer": "B"}]
    assert input_rows[0]["prediction"]["citations"] == [
        {"law_id": LAW_ID, "article_id": "22"}
    ]
    assert "citations" not in output_rows[0]


def test_private_test_path_is_configurable():
    config = {
        "data": {
            "private_test_task2_path": (
                "data/raw/private_test/Task 2 Submission File/"
                "vlsp2025_submission_task2.json"
            )
        }
    }

    assert resolve_required_samples_path(config, set_name="private_test") == Path(
        "data/raw/private_test/Task 2 Submission File/vlsp2025_submission_task2.json"
    )


def test_required_samples_loader_supports_jsonl_splits(tmp_path):
    path = tmp_path / "val_split.jsonl"
    write_jsonl([required_sample("s1"), required_sample("s2")], path)

    rows = load_json_records(path)

    assert [row["id"] for row in rows] == ["s1", "s2"]
