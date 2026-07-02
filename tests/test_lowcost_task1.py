from pathlib import Path

from src.lowcost_retrieval import (
    IMAGE_VECTOR_NAME,
    OBJECT_DIMENSION_NAME,
    OBJECT_VECTOR_NAME,
    TEXT_VECTOR_NAME,
    lowcost_task1_limits,
    retrieve_task1_citation_evidence,
    run_lowcost_task1_predictions,
    summarize_task1_ablation,
    task1_evidence_from_citations,
    task1_prediction_row,
    union_relevant_articles_from_examples,
)
from src.utils import read_jsonl, write_json, write_jsonl


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeTask1VectorStore:
    def __init__(self, results):
        self.collection_name = "traffic_train_examples_lowcost"
        self.uses_multivector = True
        self.results = results
        self.calls = []

    def recreate_collection(self, dimensions):
        raise AssertionError("Task 1 runner must not recreate the train index")

    def upsert_points(self, points):
        raise AssertionError("Task 1 runner must not upsert train points")

    def query_task1(self, query_vectors, limits, query_mode):
        self.calls.append(
            {
                "query_vectors": query_vectors,
                "limits": dict(limits),
                "query_mode": query_mode,
            }
        )
        return self.results


def dimensions() -> dict:
    return {
        TEXT_VECTOR_NAME: 2,
        IMAGE_VECTOR_NAME: 3,
        OBJECT_DIMENSION_NAME: 3,
    }


def config(**task1_overrides) -> dict:
    task1_config = {
        "query_mode": "text_image_object",
        "text_limit": 10,
        "image_limit": 5,
        "object_limit": 3,
    }
    task1_config.update(task1_overrides)
    return {
        "qdrant": {"url": "http://localhost:6333"},
        "lowcost_retrieval": {
            "collection_name": "traffic_train_examples_lowcost",
            "dimensions": dimensions(),
        },
        "lowcost_task1": task1_config,
    }


def query_feature_row(**overrides) -> dict:
    row = {
        "id": "public_test_1",
        "sample_id": "public_test_1",
        "image_id": "public_test_1_1",
        "image_path": "data/raw/public_test/images/public_test_1_1.jpg",
        "question": "Đây là biển báo gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "text_vector": [0.1, 0.2],
        "image_general_feature_vector": [0.3, 0.4, 0.5],
        "image_object_feature_list_vector": [[0.6, 0.7, 0.8]],
    }
    row.update(overrides)
    return row


def retrieved_example(sample_id: str, citations: list[dict[str, str]]) -> dict:
    return {
        "score": 0.9,
        "payload": {
            "sample_id": sample_id,
            "image_id": f"{sample_id}_image",
            "question": "Train question",
            "answer": "A",
            "relevant_articles": citations,
            "split": "train",
        },
    }


def article_index() -> dict:
    return {
        f"{LAW_ID}#22": {
            "uid": f"{LAW_ID}#22",
            "law_id": LAW_ID,
            "law_title": "Quy chuẩn báo hiệu đường bộ",
            "article_id": "22",
            "title": "Điều 22",
            "content": "Nội dung Điều 22.",
            "images": [],
            "tables": [],
        },
        f"{LAW_ID}#B.13": {
            "uid": f"{LAW_ID}#B.13",
            "law_id": LAW_ID,
            "law_title": "Quy chuẩn báo hiệu đường bộ",
            "article_id": "B.13",
            "title": "Biển B.13",
            "content": "Nội dung biển B.13.",
            "images": [],
            "tables": [],
        },
    }


def write_feature_files(tmp_path: Path, rows=None, set_name="public_test") -> Path:
    rows = rows or [query_feature_row()]
    features_path = tmp_path / f"{set_name}_features.jsonl"
    write_jsonl(rows, features_path)
    write_json(
        {
            "schema_version": "lowcost-features-manifest-v1",
            "set_name": set_name,
            "dimensions": dimensions(),
        },
        str(features_path.with_suffix(".manifest.json")),
    )
    return features_path


def test_union_of_train_relevant_articles_from_retrieved_examples():
    citations = union_relevant_articles_from_examples(
        [
            retrieved_example("train_1", [{"law_id": LAW_ID, "article_id": "22"}]),
            retrieved_example("train_2", [{"law_id": LAW_ID, "article_id": "B.13"}]),
        ]
    )

    assert citations == [
        {"law_id": LAW_ID, "article_id": "22"},
        {"law_id": LAW_ID, "article_id": "B.13"},
    ]


def test_deduplication_preserves_stable_order():
    citations = union_relevant_articles_from_examples(
        [
            retrieved_example(
                "train_1",
                [
                    {"law_id": LAW_ID, "article_id": "22"},
                    {"law_id": LAW_ID, "article_id": "B.13"},
                ],
            ),
            retrieved_example(
                "train_2",
                [
                    {"law_id": LAW_ID, "article_id": "22"},
                    {"law_id": LAW_ID, "article_id": "46"},
                ],
            ),
        ]
    )

    assert citations == [
        {"law_id": LAW_ID, "article_id": "22"},
        {"law_id": LAW_ID, "article_id": "B.13"},
        {"law_id": LAW_ID, "article_id": "46"},
    ]


def test_task1_output_row_never_contains_answer():
    row = task1_prediction_row(
        query_feature_row(answer="B"),
        [retrieved_example("train_1", [{"law_id": LAW_ID, "article_id": "22"}])],
    )

    assert row == {
        "id": "public_test_1",
        "image_id": "public_test_1_1",
        "question": "Đây là biển báo gì?",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    assert "answer" not in row


def test_task1_citations_resolve_to_prompt_ready_evidence():
    evidence, diagnostics = task1_evidence_from_citations(
        [{"law_id": LAW_ID, "article_id": "22"}],
        article_index(),
        retrieved_examples=[
            retrieved_example("train_1", [{"law_id": LAW_ID, "article_id": "22"}])
        ],
    )

    assert diagnostics == []
    assert evidence[0].uid == f"{LAW_ID}#22"
    assert evidence[0].content == "Nội dung Điều 22."
    assert evidence[0].retrieval_method == "example"
    assert evidence[0].metadata["source"] == "lowcost_task1_retrieved_examples"


def test_task1_evidence_retrieval_queries_examples_and_resolves_articles():
    store = FakeTask1VectorStore(
        [
            retrieved_example("train_1", [{"law_id": LAW_ID, "article_id": "22"}]),
            retrieved_example("train_2", [{"law_id": LAW_ID, "article_id": "B.13"}]),
        ]
    )

    evidence, diagnostics = retrieve_task1_citation_evidence(
        query_feature_row(),
        config(),
        article_index(),
        vector_store=store,
    )

    assert [item.uid for item in evidence] == [f"{LAW_ID}#22", f"{LAW_ID}#B.13"]
    assert store.calls[0]["query_mode"] == "text_image_object"
    assert diagnostics[-1]["type"] == "lowcost_task1_retrieval"
    assert diagnostics[-1]["citation_count"] == 2


def test_task1_runner_passes_configurable_limits_and_writes_packager_shape(tmp_path):
    features_path = write_feature_files(tmp_path)
    output_path = tmp_path / "public_task1_lowcost.jsonl"
    store = FakeTask1VectorStore(
        [
            retrieved_example("train_1", [{"law_id": LAW_ID, "article_id": "22"}]),
            retrieved_example("train_2", [{"law_id": LAW_ID, "article_id": "B.13"}]),
        ]
    )

    summary = run_lowcost_task1_predictions(
        config=config(text_limit=2, image_limit=1, object_limit=1, max_articles=1),
        set_name="public_test",
        features_path=features_path,
        output_path=output_path,
        vector_store=store,
    )

    assert summary["prediction_count"] == 1
    assert summary["query_mode"] == "text_image_object"
    assert store.calls[0]["limits"] == {
        "text_limit": 2,
        "image_limit": 1,
        "object_limit": 1,
        "max_articles": 1,
    }
    assert set(store.calls[0]["query_vectors"]) == {
        TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME,
        OBJECT_VECTOR_NAME,
    }
    rows = read_jsonl(str(output_path))
    assert rows == [
        {
            "id": "public_test_1",
            "image_id": "public_test_1_1",
            "question": "Đây là biển báo gì?",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
        }
    ]
    assert "answer" not in rows[0]


def test_text_image_mode_does_not_require_object_query_vector(tmp_path):
    features_path = write_feature_files(tmp_path)
    output_path = tmp_path / "public_task1_lowcost_text_image.jsonl"
    store = FakeTask1VectorStore(
        [retrieved_example("train_1", [{"law_id": LAW_ID, "article_id": "22"}])]
    )

    run_lowcost_task1_predictions(
        config=config(query_mode="text_image", text_limit=4, image_limit=2),
        set_name="public_test",
        features_path=features_path,
        output_path=output_path,
        vector_store=store,
    )

    assert store.calls[0]["query_mode"] == "text_image"
    assert set(store.calls[0]["query_vectors"]) == {
        TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME,
    }
    assert lowcost_task1_limits(config(query_mode="text_image"))["object_limit"] == 3


def test_validation_ablation_chooses_best_f2_setting():
    gold_rows = [
        {
            "id": "val_1",
            "image_id": "image_1",
            "question": "Q1",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
        },
        {
            "id": "val_2",
            "image_id": "image_2",
            "question": "Q2",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "46"}],
        },
    ]
    weak_predictions = [
        {
            "id": "val_1",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "B.13"}],
        },
        {
            "id": "val_2",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "46"}],
        },
    ]
    strong_predictions = [
        {
            "id": "val_1",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
        },
        {
            "id": "val_2",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "46"}],
        },
    ]

    summary = summarize_task1_ablation(
        gold_rows,
        {
            "weak_t1_i1_o1": weak_predictions,
            "strong_t10_i5_o3": strong_predictions,
        },
    )

    assert summary["best_setting"] == "strong_t10_i5_o3"
    assert summary["settings"][0]["f2"] == 1.0
