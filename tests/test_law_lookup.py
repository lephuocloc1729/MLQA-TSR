import pytest

from src.data_utils import (
    build_law_article_index,
    load_processed_law_article_index,
    reference_uid,
    resolve_law_reference,
)


LAW_ID = "QCVN 41:2024/BGTVT"


def article(article_id: str = "22") -> dict:
    return {
        "uid": f"{LAW_ID}#{article_id}",
        "law_id": LAW_ID,
        "law_title": "Quy chuẩn báo hiệu đường bộ",
        "article_id": article_id,
        "title": "Tác dụng của biển báo",
        "content": "Nội dung điều luật.",
        "images": [],
        "tables": [],
    }


def test_reference_uid_normalizes_law_and_article_ids():
    assert (
        reference_uid({"law_id": f" {LAW_ID} ", "article_id": " 22 "})
        == f"{LAW_ID}#22"
    )


def test_reference_uid_requires_law_and_article_id():
    with pytest.raises(ValueError, match="law_id and article_id"):
        reference_uid({"law_id": LAW_ID, "article_id": ""})


def test_resolve_known_law_reference_returns_article_content():
    index = build_law_article_index([article("22")])

    resolved = resolve_law_reference(
        {"law_id": LAW_ID, "article_id": "22"},
        index,
    )

    assert resolved["title"] == "Tác dụng của biển báo"
    assert resolved["content"] == "Nội dung điều luật."


def test_unknown_law_reference_raises_clear_error():
    index = build_law_article_index([article("22")])

    with pytest.raises(KeyError, match="Unknown law article UID"):
        resolve_law_reference({"law_id": LAW_ID, "article_id": "999"}, index)


def test_load_processed_law_article_index_missing_file_is_helpful(tmp_path):
    config = {"data": {"processed_law_path": str(tmp_path / "missing.jsonl")}}

    with pytest.raises(FileNotFoundError, match="mode preprocess"):
        load_processed_law_article_index(config)
