import json
from pathlib import Path

import pytest

from src.evaluate import (
    build_evaluation_artifact,
    default_output_path,
    normalize_gold_answer,
    normalize_prediction_answer,
    read_prediction_jsonl,
    score_retrieval_sample,
)


LAW_ID = "QCVN 41:2024/BGTVT"


def citation(article_id: str) -> dict:
    return {"law_id": LAW_ID, "article_id": article_id}


def test_perfect_retrieval_returns_one():
    score = score_retrieval_sample(
        [citation("22"), citation("41")],
        [citation("41"), citation("22")],
    )

    assert score["precision"] == 1.0
    assert score["recall"] == 1.0
    assert score["f2"] == 1.0


def test_partial_retrieval_uses_f2_formula():
    score = score_retrieval_sample(
        [citation("22"), citation("41")],
        [citation("22")],
    )

    assert score["precision"] == 1.0
    assert score["recall"] == 0.5
    assert score["f2"] == pytest.approx(5 * 1.0 * 0.5 / ((4 * 1.0) + 0.5))


def test_empty_predicted_citations_returns_zero_without_crashing():
    score = score_retrieval_sample([citation("22")], [])

    assert score["precision"] == 0.0
    assert score["recall"] == 0.0
    assert score["f2"] == 0.0


def test_answer_normalization_handles_decomposed_vietnamese_accents():
    assert normalize_gold_answer("Đúng") == "Đúng"
    assert normalize_prediction_answer("Đúng") == "Đúng"


def test_invalid_prediction_is_counted_not_silently_corrected():
    records = [
        {
            "id": "s1",
            "question_type": "Multiple choice",
            "answer": "A",
            "predict": "40",
            "relevant_articles": [citation("22")],
            "predicted_articles": [citation("22")],
        }
    ]

    artifact = build_evaluation_artifact(records)

    assert artifact["qa"]["accuracy"] == 0.0
    assert artifact["qa"]["total"] == 1
    assert artifact["invalid_prediction_count"] == 1
    assert artifact["invalid_predictions"][0]["id"] == "s1"
    assert "invalid Multiple choice prediction answer" in artifact["invalid_predictions"][0][
        "reason"
    ]


def test_pipeline_result_shape_is_supported():
    records = [
        {
            "query": {
                "id": "s1",
                "image_id": "img1",
                "question": "Chọn đáp án đúng?",
                "question_type": "Multiple choice",
                "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
                "answer": "B",
                "relevant_articles": [citation("22")],
            },
            "prediction": {
                "id": "s1",
                "question_type": "Multiple choice",
                "answer": "B",
                "citations": [citation("22")],
                "explanation": "Dựa trên Điều 22.",
            },
            "timings_ms": {"retrieval": 5.0, "generation": 10.0},
        }
    ]

    artifact = build_evaluation_artifact(records)

    assert artifact["retrieval"]["f2"] == 1.0
    assert artifact["qa"]["accuracy"] == 1.0
    assert artifact["latency_ms"]["mean"] == 15.0


def test_w3_artifact_includes_model_parse_retrieval_and_adapter_metadata():
    config = {
        "project": {"seed": 42},
        "experiment": {
            "name": "w3_adapter_diag",
            "label": "W3 adapter diagnostic",
            "mock": False,
            "retrieval_strategy": "fusion",
            "prompt_variant": "structured_legal_rag",
        },
        "model": {
            "backend": "openai_compatible",
            "name": "Qwen/Qwen2.5-VL-3B-Instruct",
            "max_new_tokens": 320,
            "include_image": True,
        },
        "retrieval": {"top_k": 5, "example_top_k": 3},
        "retrieval_freeze": {"version": "retrieval-final-v1"},
        "adapter_diagnostic": {
            "metadata_path": "checkpoints/qlora_adapter/adapter_metadata.json",
            "effective_train_count": 80,
            "status": "diagnostic",
        },
    }
    records = [
        {
            "query": {
                "id": "s1",
                "question_type": "Multiple choice",
                "answer": "B",
                "relevant_articles": [citation("22")],
            },
            "prediction": {
                "id": "s1",
                "question_type": "Multiple choice",
                "answer": "B",
                "citations": [citation("22")],
                "explanation": "Dựa trên điều luật.",
                "raw_response": '{"answer":"B"}',
            },
            "predicted_articles": [citation("22")],
            "model": {
                "backend": "openai_compatible",
                "name": "qwen2.5-vl-local",
                "max_new_tokens": 320,
                "include_image": True,
            },
            "retrieval_config": {
                "strategy": "fusion",
                "freeze_version": "retrieval-final-v1",
                "top_k": 5,
                "example_top_k": 3,
            },
            "parse": {
                "status": "success",
                "success": True,
                "invalid_json": False,
                "truncated_output": False,
            },
        }
    ]

    artifact = build_evaluation_artifact(records, config=config)

    assert artifact["mock"] is False
    assert artifact["model"]["backend"] == "openai_compatible"
    assert artifact["model"]["name"] == "qwen2.5-vl-local"
    assert artifact["model"]["max_new_tokens"] == 320
    assert artifact["retrieval_config"]["strategy"] == "fusion"
    assert artifact["retrieval_config"]["freeze_version"] == "retrieval-final-v1"
    assert artifact["parse"]["parse_success_count"] == 1
    assert artifact["parse"]["invalid_json_count"] == 0
    assert artifact["parse"]["truncated_output_count"] == 0
    assert artifact["adapter_diagnostic"]["effective_train_count"] == 80


def test_invalid_json_and_truncated_predictions_are_counted_as_incorrect():
    records = [
        {
            "query": {
                "id": "bad_json",
                "question_type": "Multiple choice",
                "answer": "B",
                "relevant_articles": [citation("22")],
            },
            "prediction": {
                "id": "bad_json",
                "question_type": "Multiple choice",
                "answer": None,
                "citations": [],
                "explanation": "Model error: ValueError: VLM response does not contain a JSON object",
                "error": {
                    "type": "ValueError",
                    "message": "VLM response does not contain a JSON object",
                },
            },
            "parse": {
                "status": "invalid_json",
                "success": False,
                "invalid_json": True,
                "truncated_output": False,
            },
        },
        {
            "query": {
                "id": "truncated",
                "question_type": "Yes/No",
                "answer": "Đúng",
                "relevant_articles": [citation("41")],
            },
            "prediction": {
                "id": "truncated",
                "question_type": "Yes/No",
                "answer": None,
                "citations": [],
                "explanation": "Model output truncated at max_new_tokens=160",
                "error": {
                    "type": "ValueError",
                    "message": "Model output truncated at max_new_tokens=160",
                },
            },
            "parse": {
                "status": "error",
                "success": False,
                "invalid_json": False,
                "truncated_output": True,
            },
        },
    ]

    artifact = build_evaluation_artifact(records)

    assert artifact["qa"]["accuracy"] == 0.0
    assert artifact["qa"]["total"] == 2
    assert artifact["invalid_prediction_count"] == 2
    assert artifact["parse"]["parse_failure_count"] == 2
    assert artifact["parse"]["invalid_json_count"] == 1
    assert artifact["parse"]["truncated_output_count"] == 1
    assert artifact["parse"]["invalid_json_sample_ids"] == ["bad_json"]
    assert artifact["parse"]["truncated_sample_ids"] == ["truncated"]


def test_malformed_jsonl_row_is_reported(tmp_path: Path):
    path = tmp_path / "predictions.jsonl"
    path.write_text('{"id": "ok"}\n{bad json\n', encoding="utf-8")

    records, invalid_rows = read_prediction_jsonl(path)

    assert len(records) == 1
    assert invalid_rows[0]["id"] == "line_2"
    assert invalid_rows[0]["reason"].startswith("malformed JSON:")


def test_tiny_fixture_evaluates_and_is_serializable():
    path = Path("tests/fixtures/tiny_predictions.jsonl")
    records, invalid_rows = read_prediction_jsonl(path)
    artifact = build_evaluation_artifact(records, invalid_rows=invalid_rows)

    json.dumps(artifact, ensure_ascii=False)
    assert artifact["sample_count"] == 3
    assert artifact["invalid_prediction_count"] == 1
    assert artifact["qa"]["by_question_type"]["Multiple choice"]["total"] == 2


def test_default_metrics_path_is_saved_beside_predictions():
    path = Path("data/outputs/experiments/run.jsonl")

    assert default_output_path({}, path) == Path("data/outputs/experiments/run_metrics.json")


def test_configured_metrics_path_wins_over_default():
    config = {"experiment": {"metrics_path": "data/outputs/custom_metrics.json"}}

    assert default_output_path(config, "data/outputs/experiments/run.jsonl") == Path(
        "data/outputs/custom_metrics.json"
    )
