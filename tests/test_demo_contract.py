import json
from pathlib import Path

from src.pipeline import (
    build_demo_inspection,
    demo_model_status,
    make_demo_inspection_record,
)
from src.schemas import Evidence, Query


LAW_ID = "QCVN 41:2024/BGTVT"


def sample(**overrides) -> dict:
    data = {
        "id": "val_1",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    data.update(overrides)
    return data


def evidence(article_id: str = "22") -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id=article_id,
        title=f"Điều {article_id}",
        content=f"Nội dung pháp lý của Điều {article_id}.",
        score=0.82,
        rank=1,
        retrieval_method="text",
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_demo_inspection_record_contract_is_ui_ready_without_raw_path_leakage():
    query = Query.model_validate(sample())
    record = make_demo_inspection_record(
        sample=query,
        evidence=[evidence()],
        retrieval_strategy_name="text",
        top_k=5,
        model_status={"available": False, "mode": "retrieval_only"},
    )

    assert record["schema_version"] == "demo-inspection-v1"
    assert record["sample"]["id"] == "val_1"
    assert record["sample"]["image_display_name"] == "img_1.jpg"
    assert "image_path" not in record["sample"]
    assert record["local_image_path"].endswith("img_1.jpg")
    assert record["retrieval"]["citation_ids"] == [f"{LAW_ID}#22"]
    assert record["retrieval"]["evidence"][0]["score"] == 0.82
    assert record["prediction"] is None
    assert "api_key" not in json.dumps(record, ensure_ascii=False).lower()


def test_missing_model_credentials_fall_back_to_retrieval_only_mode():
    status = demo_model_status(
        {"demo": {"enable_vlm": True, "vlm_api_key_env": "TRAFFIC_VLM_API_KEY"}},
        include_prediction=True,
        use_mock_prediction=False,
        environ={},
    )

    assert status["available"] is False
    assert status["mode"] == "retrieval_only"
    assert "TRAFFIC_VLM_API_KEY" in status["reason"]


def test_mock_prediction_status_is_explicitly_marked():
    status = demo_model_status(
        {},
        include_prediction=True,
        use_mock_prediction=True,
        environ={},
    )

    assert status["available"] is True
    assert status["mode"] == "mock_prediction"
    assert "mock" in status["reason"].lower()


def test_build_demo_inspection_retrieval_none_runs_without_docker_or_models(tmp_path):
    split_path = tmp_path / "val_split.jsonl"
    write_jsonl(split_path, [sample()])
    config = {
        "data": {
            "val_split_path": str(split_path),
            "train_split_path": str(tmp_path / "train_split.jsonl"),
        },
        "retrieval": {"top_k": 5},
        "model": {"name": "mock-vlm"},
    }

    record = build_demo_inspection(
        sample_id="val_1",
        config=config,
        split="val",
        top_k=3,
        retrieval_strategy_name="none",
        include_prediction=True,
        use_mock_prediction=False,
    )

    assert record["sample"]["id"] == "val_1"
    assert record["retrieval"]["strategy"] == "none"
    assert record["retrieval"]["top_k"] == 3
    assert record["retrieval"]["evidence"] == []
    assert record["prediction"] is None
    assert record["model"]["mode"] == "retrieval_only"
