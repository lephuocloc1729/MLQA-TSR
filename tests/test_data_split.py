import json
from pathlib import Path

import pytest

from src.data_utils import (
    build_law_article_index,
    grouped_train_val_split,
    load_law_articles,
    load_validated_train_samples,
    normalize_article_reference,
    normalize_train_sample,
    validate_split_integrity,
)


LAW_ID = "QCVN 41:2024/BGTVT"
REAL_TRAIN_PATH = Path("data/raw/train_data/vlsp_2025_train.json")
PROCESSED_LAW_PATH = Path("data/processed/law_articles.jsonl")


def _sample(sample_id: str, image_id: str) -> dict:
    return {
        "id": sample_id,
        "image_id": image_id,
        "question": f"Câu hỏi {sample_id}?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "A",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }


def test_grouped_split_is_deterministic():
    samples = [_sample("s1", "img1"), _sample("s2", "img1"), _sample("s3", "img2")]

    first = grouped_train_val_split(samples, seed=42, validation_ratio=0.34)
    second = grouped_train_val_split(samples, seed=42, validation_ratio=0.34)

    assert first == second


def test_grouped_split_has_no_image_leakage():
    samples = [
        _sample("s1", "img1"),
        _sample("s2", "img1"),
        _sample("s3", "img2"),
        _sample("s4", "img3"),
    ]

    train_samples, val_samples = grouped_train_val_split(
        samples,
        seed=7,
        validation_ratio=0.5,
    )

    validate_split_integrity(train_samples, val_samples, samples)
    train_images = {sample["image_id"] for sample in train_samples}
    val_images = {sample["image_id"] for sample in val_samples}
    assert train_images.isdisjoint(val_images)


def test_answer_normalization_and_image_path_for_training_sample():
    sample = {
        "id": "s1",
        "image_id": "img1",
        "question": "Đúng hay sai?",
        "question_type": "Yes/No",
        "answer": "Đúng",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    config = {"data": {"train_image_dir": "data/raw/train_data/train_images"}}
    article_index = {
        f"{LAW_ID}#22": {"law_id": LAW_ID, "article_id": "22"},
    }

    normalized = normalize_train_sample(sample, config, article_index)

    assert normalized["answer"] == "Đúng"
    assert normalized["image_path"].endswith("data/raw/train_data/train_images/img1.jpg")


def test_legacy_numeric_answer_40_is_audited_and_mapped_to_a():
    sample = {
        "id": "s1",
        "image_id": "img1",
        "question": "Chọn đáp án?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": 40,
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    config = {"data": {"train_image_dir": "data/raw/train_data/train_images"}}
    article_index = {
        f"{LAW_ID}#22": {"law_id": LAW_ID, "article_id": "22"},
    }

    from collections import Counter

    answer_audit = Counter()
    normalized = normalize_train_sample(
        sample,
        config,
        article_index,
        answer_audit=answer_audit,
    )

    assert normalized["answer"] == "A"
    assert answer_audit["40 -> A"] == 1


def test_article_reference_alias_splits_compound_reference():
    article_index = {
        f"{LAW_ID}#B.3": {},
        f"{LAW_ID}#41": {},
    }

    normalized = normalize_article_reference(
        {"law_id": LAW_ID, "article_id": "B.3; 41"},
        article_index,
    )

    assert normalized == [
        {"law_id": LAW_ID, "article_id": "B.3"},
        {"law_id": LAW_ID, "article_id": "41"},
    ]


def test_real_train_references_resolve_after_normalization():
    if not REAL_TRAIN_PATH.exists() or not PROCESSED_LAW_PATH.exists():
        pytest.skip("Local train data or processed LawDB is unavailable")

    article_index = build_law_article_index(load_law_articles(PROCESSED_LAW_PATH))
    config = {
        "data": {
            "train_path": str(REAL_TRAIN_PATH),
            "train_image_dir": "data/raw/train_data/train_images",
        }
    }

    samples, audit = load_validated_train_samples(config, article_index)

    assert len(samples) == 530
    assert audit["answer_normalizations"]["40 -> A"] == 1
    for sample in samples:
        for reference in sample["relevant_articles"]:
            uid = f"{reference['law_id']}#{reference['article_id']}"
            assert uid in article_index


def test_real_split_has_expected_sample_and_image_counts():
    if not REAL_TRAIN_PATH.exists() or not PROCESSED_LAW_PATH.exists():
        pytest.skip("Local train data or processed LawDB is unavailable")

    raw_samples = json.loads(REAL_TRAIN_PATH.read_text(encoding="utf-8"))
    image_count = len({sample["image_id"] for sample in raw_samples})

    assert len(raw_samples) == 530
    assert image_count == 304
