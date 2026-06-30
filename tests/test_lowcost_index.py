from pathlib import Path

import pytest

from src.lowcost_retrieval import (
    IMAGE_VECTOR_NAME,
    OBJECT_DIMENSION_NAME,
    OBJECT_VECTOR_NAME,
    TEXT_VECTOR_NAME,
    build_named_vector_config,
    index_lowcost_train_examples,
    payload_from_feature_row,
    point_from_feature_row,
)
from src.utils import write_json, write_jsonl


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeLowCostVectorStore:
    def __init__(self) -> None:
        self.collection_name = "traffic_train_examples_lowcost"
        self.uses_multivector = True
        self.recreated_dimensions = None
        self.points = []

    def recreate_collection(self, dimensions):
        self.recreated_dimensions = dict(dimensions)

    def upsert_points(self, points):
        self.points.extend(points)


def feature_row(**overrides) -> dict:
    row = {
        "id": "train_1",
        "sample_id": "train_1",
        "image_id": "train_1_3",
        "image_path": "data/raw/train_data/train_images/train_1_3.jpg",
        "question": "Đây là biển báo gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
        "text_vector": [0.1, 0.2],
        "image_general_feature_vector": [0.3, 0.4, 0.5],
        "image_object_feature_list_vector": [[0.6, 0.7, 0.8]],
    }
    row.update(overrides)
    return row


def dimensions() -> dict:
    return {
        TEXT_VECTOR_NAME: 2,
        IMAGE_VECTOR_NAME: 3,
        OBJECT_DIMENSION_NAME: 3,
    }


def config(tmp_path: Path, **dimension_overrides) -> dict:
    dims = dimensions()
    dims.update(dimension_overrides)
    return {
        "qdrant": {"url": "http://localhost:6333"},
        "lowcost_retrieval": {
            "collection_name": "traffic_train_examples_lowcost",
            "index_manifest_path": str(tmp_path / "index_manifest.json"),
            "dimensions": dims,
        },
    }


def write_feature_files(tmp_path: Path, rows=None, manifest_dimensions=None, set_name="train") -> Path:
    rows = rows or [feature_row()]
    manifest_dimensions = manifest_dimensions or dimensions()
    features_path = tmp_path / "train_features.jsonl"
    write_jsonl(rows, features_path)
    write_json(
        {
            "schema_version": "lowcost-features-manifest-v1",
            "set_name": set_name,
            "models": {
                "text": "fake-jina",
                "image": "fake-c-radio",
                "object_detector": "fake-owlv2",
                "object_encoder": "fake-c-radio",
            },
            "dimensions": manifest_dimensions,
        },
        str(features_path.with_suffix(".manifest.json")),
    )
    return features_path


def test_vector_config_uses_three_named_vectors_and_multivector_object():
    vector_config = build_named_vector_config(dimensions())

    assert set(vector_config) == {
        TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME,
        OBJECT_VECTOR_NAME,
    }
    assert vector_config[TEXT_VECTOR_NAME].size == 2
    assert vector_config[IMAGE_VECTOR_NAME].size == 3
    assert vector_config[OBJECT_VECTOR_NAME].size == 3
    assert vector_config[OBJECT_VECTOR_NAME].multivector_config is not None
    assert vector_config[OBJECT_VECTOR_NAME].multivector_config.comparator == "max_sim"


def test_payload_conversion_preserves_train_relevant_articles():
    payload = payload_from_feature_row(feature_row())

    assert payload == {
        "sample_id": "train_1",
        "image_id": "train_1_3",
        "image_path": "data/raw/train_data/train_images/train_1_3.jpg",
        "question": "Đây là biển báo gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
        "split": "train",
    }


def test_missing_relevant_articles_fails_for_train_indexing():
    row = feature_row()
    row.pop("relevant_articles")

    with pytest.raises(ValueError, match="must include relevant_articles"):
        payload_from_feature_row(row)


def test_dimension_mismatch_fails_for_feature_row():
    with pytest.raises(ValueError, match="feature dimension mismatch for text_vector"):
        point_from_feature_row(feature_row(text_vector=[0.1]), dimensions())


def test_manifest_dimension_mismatch_fails(tmp_path):
    features_path = write_feature_files(
        tmp_path,
        manifest_dimensions={
            TEXT_VECTOR_NAME: 99,
            IMAGE_VECTOR_NAME: 3,
            OBJECT_DIMENSION_NAME: 3,
        },
    )

    with pytest.raises(ValueError, match="Feature dimension mismatch for text_vector"):
        index_lowcost_train_examples(
            config(tmp_path),
            features_path,
            vector_store=FakeLowCostVectorStore(),
        )


def test_fake_vector_store_receives_expected_named_vectors(tmp_path):
    features_path = write_feature_files(tmp_path)
    store = FakeLowCostVectorStore()

    summary = index_lowcost_train_examples(
        config(tmp_path),
        features_path,
        vector_store=store,
    )

    assert summary["indexed_examples"] == 1
    assert summary["collection"] == "traffic_train_examples_lowcost"
    assert summary["vectors"] == [
        TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME,
        OBJECT_VECTOR_NAME,
    ]
    assert store.recreated_dimensions == dimensions()
    assert len(store.points) == 1
    point = store.points[0]
    assert set(point["vector"]) == {
        TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME,
        OBJECT_VECTOR_NAME,
    }
    assert point["vector"][OBJECT_VECTOR_NAME] == [[0.6, 0.7, 0.8]]
    assert point["payload"]["relevant_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert (tmp_path / "index_manifest.json").exists()


def test_refuses_non_train_feature_manifest(tmp_path):
    features_path = write_feature_files(tmp_path, set_name="public_test")

    with pytest.raises(ValueError, match="requires train features"):
        index_lowcost_train_examples(
            config(tmp_path),
            features_path,
            vector_store=FakeLowCostVectorStore(),
        )
