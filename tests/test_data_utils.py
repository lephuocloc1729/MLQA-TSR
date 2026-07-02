import json
from pathlib import Path

import pytest

from src.data_utils import (
    build_freeform_val_record,
    build_freeform_validation_dataset,
    build_law_article_index,
    build_law_articles,
    clean_html,
    extract_images,
    extract_tables,
    freeform_expected_answer,
    freeform_question_from_sample,
    get_law_article,
    iter_law_articles,
    load_law_articles,
    normalize_article_id,
)


REAL_LAWDB_PATH = Path("data/raw/law_db/vlsp2025_law_new.json")


def test_marker_parsing_with_table_and_image():
    raw_text = """
    Điều thử nghiệm.
    <<TABLE:<table><tr><th>Loại</th><td>Biển cấm</td></tr></table>/TABLE>>
    <<IMAGE: image001.jpg /IMAGE>>
    Kết thúc.
    """

    assert extract_images(raw_text) == ["image001.jpg"]
    assert extract_tables(raw_text) == ["Loại Biển cấm"]
    assert clean_html(raw_text) == "Điều thử nghiệm. Kết thúc."


def test_iter_law_articles_flattens_document_structure():
    raw_data = [
        {
            "id": "LAW-1",
            "title": "Luật thử nghiệm",
            "articles": [
                {
                    "id": "1",
                    "title": "Điều 1",
                    "text": "Nội dung <<IMAGE: sign.png /IMAGE>>",
                }
            ],
        }
    ]

    articles = list(iter_law_articles(raw_data))

    assert articles == [
        {
            "uid": "LAW-1#1",
            "law_id": "LAW-1",
            "law_title": "Luật thử nghiệm",
            "article_id": "1",
            "title": "Điều 1",
            "content": "Nội dung",
            "images": ["sign.png"],
            "tables": [],
        }
    ]


def test_title_subsection_prefix_recovers_specific_article_id():
    assert normalize_article_id("K.1", "K.1.1 Chữ in hoa") == "K.1.1"
    assert normalize_article_id("K.1", "K.1 Kiểu chữ thường") == "K.1"
    assert normalize_article_id("22", "Tác dụng của biển báo") == "22"


def test_build_law_article_index_rejects_duplicate_uid():
    article = {
        "uid": "LAW#1",
        "law_id": "LAW",
        "article_id": "1",
        "title": "Điều 1",
        "content": "Nội dung",
        "images": [],
        "tables": [],
    }

    with pytest.raises(ValueError, match="Duplicate law article UID"):
        build_law_article_index([article, article])


def test_lookup_returns_article_and_unknown_uid_raises():
    article = {
        "uid": "QCVN 41:2024/BGTVT#22",
        "law_id": "QCVN 41:2024/BGTVT",
        "law_title": "Quy chuẩn",
        "article_id": "22",
        "title": "Tác dụng của biển báo",
        "content": "Nội dung điều 22",
        "images": [],
        "tables": [],
    }
    index = build_law_article_index([article])

    assert get_law_article("QCVN 41:2024/BGTVT", "22", index) == article

    with pytest.raises(KeyError, match="Unknown law article UID"):
        get_law_article("QCVN 41:2024/BGTVT", "999", index)


def test_build_law_articles_writes_expected_jsonl_shape(tmp_path):
    law_path = tmp_path / "law.json"
    output_path = tmp_path / "law_articles.jsonl"
    law_path.write_text(
        json.dumps(
            [
                {
                    "id": "LAW",
                    "title": "Luật",
                    "articles": [
                        {
                            "id": "1",
                            "title": "Điều 1",
                            "text": "Nội dung <b>quan trọng</b>",
                        }
                    ],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    articles = build_law_articles(
        {"data": {"law_path": str(law_path), "processed_law_path": str(output_path)}}
    )
    rows = load_law_articles(output_path)

    assert articles == rows
    assert rows[0]["content"] == "Nội dung quan trọng"
    assert "raw" not in rows[0]


def test_freeform_question_and_expected_answer_from_multiple_choice():
    sample = {
        "question": "Đây là biển báo gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Biển cấm", "B": "Biển nguy hiểm", "C": "Biển hiệu lệnh", "D": "Biển chỉ dẫn"},
        "answer": "B",
    }

    assert "không chỉ chọn A/B/C/D" in freeform_question_from_sample(sample)
    assert freeform_expected_answer(sample) == "Kết luận: Biển nguy hiểm."


def test_build_freeform_val_record_contains_gold_citations_and_target():
    article = {
        "uid": "QCVN 41:2024/BGTVT#22",
        "law_id": "QCVN 41:2024/BGTVT",
        "law_title": "Quy chuẩn",
        "article_id": "22",
        "title": "Điều 22",
        "content": "Nội dung điều 22",
        "images": [],
        "tables": [],
    }
    sample = {
        "id": "val_1",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Yes/No",
        "answer": "Đúng",
        "relevant_articles": [{"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}],
    }

    record = build_freeform_val_record(sample, build_law_article_index([article]), index=1)

    assert record["id"] == "freeform_val_0001"
    assert record["source_id"] == "val_1"
    assert record["question_type"] == "Free-form"
    assert record["expected_answer"] == "Kết luận: Đúng."
    assert record["expected_citations"] == [
        {"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}
    ]
    assert record["target"]["citations"] == record["expected_citations"]


def test_build_freeform_validation_dataset_writes_jsonl(tmp_path):
    processed_law_path = tmp_path / "law_articles.jsonl"
    val_split_path = tmp_path / "val_split.jsonl"
    output_path = tmp_path / "freeform_val.jsonl"
    article = {
        "uid": "QCVN 41:2024/BGTVT#22",
        "law_id": "QCVN 41:2024/BGTVT",
        "law_title": "Quy chuẩn",
        "article_id": "22",
        "title": "Điều 22",
        "content": "Nội dung điều 22",
        "images": [],
        "tables": [],
    }
    sample = {
        "id": "val_1",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question": "Đây là biển báo gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": "QCVN 41:2024/BGTVT", "article_id": "22"}],
    }
    processed_law_path.write_text(json.dumps(article, ensure_ascii=False) + "\n", encoding="utf-8")
    val_split_path.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = build_freeform_validation_dataset(
        {
            "data": {
                "val_split_path": str(val_split_path),
                "processed_law_path": str(processed_law_path),
                "freeform_val_path": str(output_path),
            }
        }
    )

    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert summary["record_count"] == 1
    assert rows[0]["target"]["answer"] == "Kết luận: Hai."


@pytest.mark.skipif(not REAL_LAWDB_PATH.exists(), reason="Local VLSP LawDB is unavailable")
def test_real_lawdb_flattens_to_402_articles():
    raw_data = json.loads(REAL_LAWDB_PATH.read_text(encoding="utf-8"))
    articles = list(iter_law_articles(raw_data))

    assert len(articles) == 402
    assert len(build_law_article_index(articles)) == 402
    assert {article["law_id"] for article in articles} == {
        "QCVN 41:2024/BGTVT",
        "36/2024/QH15",
    }
    assert all(article["content"] for article in articles)
    assert all(isinstance(article["images"], list) for article in articles)
    assert all(isinstance(article["tables"], list) for article in articles)


@pytest.mark.skipif(not REAL_LAWDB_PATH.exists(), reason="Local VLSP LawDB is unavailable")
def test_real_lawdb_lookup_article_22():
    raw_data = json.loads(REAL_LAWDB_PATH.read_text(encoding="utf-8"))
    index = build_law_article_index(iter_law_articles(raw_data))
    article = get_law_article("QCVN 41:2024/BGTVT", "22", index)

    assert article["uid"] == "QCVN 41:2024/BGTVT#22"
    assert article["title"]
    assert article["content"]


@pytest.mark.skipif(not REAL_LAWDB_PATH.exists(), reason="Local VLSP LawDB is unavailable")
def test_real_lawdb_recovers_duplicate_appendix_subsection_ids():
    raw_data = json.loads(REAL_LAWDB_PATH.read_text(encoding="utf-8"))
    index = build_law_article_index(iter_law_articles(raw_data))

    assert "QCVN 41:2024/BGTVT#K.1.1" in index
    assert "QCVN 41:2024/BGTVT#K.2.2" in index
