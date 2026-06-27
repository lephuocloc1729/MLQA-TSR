import json

import pytest

from src.data_utils import (
    build_law_article_index,
    build_sft_record,
    validate_sft_split_leakage,
)
from src.schemas import Evidence, Query
from src.vlm import parse_prediction


LAW_ID = "QCVN 41:2024/BGTVT"


def article(article_id: str = "22") -> dict:
    return {
        "uid": f"{LAW_ID}#{article_id}",
        "law_id": LAW_ID,
        "law_title": "Quy chuẩn báo hiệu đường bộ",
        "article_id": article_id,
        "title": f"Điều {article_id}",
        "content": f"Nội dung pháp lý của Điều {article_id}.",
        "images": [],
        "tables": [],
    }


def article_index() -> dict:
    return build_law_article_index([article("22"), article("32")])


def multiple_choice_sample(**overrides) -> dict:
    data = {
        "id": "train_1",
        "image_id": "train_1_3",
        "image_path": "data/raw/train_data/train_images/train_1_3.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    data.update(overrides)
    return data


def yes_no_sample(**overrides) -> dict:
    data = {
        "id": "train_2",
        "image_id": "train_2_1",
        "image_path": "data/raw/train_data/train_images/train_2_1.jpg",
        "question": "Xe máy được phép đi vào đường này, đúng hay sai?",
        "question_type": "Yes/No",
        "answer": "Sai",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "32"}],
    }
    data.update(overrides)
    return data


def parse_target(record: dict):
    raw_response = record["messages"][1]["content"]
    query = Query.model_validate(
        {
            "id": record["id"],
            "image_id": record["image_id"],
            "image_path": record["image_path"],
            "question": "placeholder question",
            "question_type": record["question_type"],
            "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"}
            if record["question_type"] == "Multiple choice"
            else {},
        }
    )
    evidence = [Evidence.model_validate(item) for item in record["evidence"]]
    return parse_prediction(raw_response, query=query, evidence=evidence)


def test_sft_record_creation_for_multiple_choice():
    record = build_sft_record(multiple_choice_sample(), article_index(), split="train")

    assert record["split"] == "train"
    assert record["image_id"] == "train_1_3"
    assert record["messages"][0]["role"] == "user"
    assert record["messages"][1]["role"] == "assistant"
    assert "Biển báo này có ý nghĩa gì?" in record["messages"][0]["content"]
    assert "Retrieved legal evidence" in record["messages"][0]["content"]
    assert json.loads(record["messages"][1]["content"]) == record["target"]
    assert record["target"]["answer"] == "B"
    assert record["target"]["citations"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert record["evidence"][0]["retrieval_method"] == "oracle"


def test_sft_record_creation_for_yes_no():
    record = build_sft_record(yes_no_sample(), article_index(), split="val")
    prediction = parse_target(record)

    assert record["split"] == "val"
    assert record["target"]["answer"] == "Sai"
    assert prediction.answer == "Sai"
    assert prediction.citations[0].uid == f"{LAW_ID}#32"


def test_target_json_parses_with_project_parser():
    record = build_sft_record(multiple_choice_sample(), article_index(), split="train")
    prediction = parse_target(record)

    assert prediction.answer == "B"
    assert prediction.confidence == 1.0
    assert prediction.abstained is False


def test_split_leakage_guard_by_image_id():
    train_record = build_sft_record(multiple_choice_sample(), article_index(), split="train")
    val_record = build_sft_record(
        yes_no_sample(image_id="train_1_3"),
        article_index(),
        split="val",
    )

    with pytest.raises(ValueError, match="overlap by image_id"):
        validate_sft_split_leakage([train_record], [val_record])


def test_unknown_target_citation_fails_clearly():
    bad_sample = multiple_choice_sample(
        relevant_articles=[{"law_id": LAW_ID, "article_id": "999"}]
    )

    with pytest.raises(KeyError, match="Unknown law article UID"):
        build_sft_record(bad_sample, article_index(), split="train")
