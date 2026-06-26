import json
from pathlib import Path

import pytest

from src.retrieval import (
    ExampleSearchResult,
    build_example_text,
    example_payload,
    index_examples,
    retrieve_examples,
)


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeTextEmbedder:
    model_name = "fake-text"

    def __init__(self) -> None:
        self.texts = []

    def embed_texts(self, texts):
        self.texts.extend(texts)
        return [[float(len(text)), 1.0] for text in texts]


class FakeImageEmbedder:
    model_name = "fake-image"

    def __init__(self) -> None:
        self.paths = []

    def embed_images(self, paths):
        self.paths.extend([str(path) for path in paths])
        return [[float(index + 1), 1.0] for index, _ in enumerate(paths)]


class FakeExampleStore:
    def __init__(self, text_points=None, image_points=None) -> None:
        self.text_points = text_points or []
        self.image_points = image_points or []
        self.recreated = None
        self.upserted = None
        self.queries = []

    def recreate_collection(self, text_vector_size: int, image_vector_size: int) -> None:
        self.recreated = (text_vector_size, image_vector_size)

    def upsert_examples(self, text_embeddings, image_embeddings, payloads, ids) -> None:
        self.upserted = {
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings,
            "payloads": payloads,
            "ids": ids,
        }

    def query(self, query_vector, vector_name: str, top_k: int):
        self.queries.append(
            {"query_vector": query_vector, "vector_name": vector_name, "top_k": top_k}
        )
        points = self.text_points if vector_name == "text" else self.image_points
        return points[:top_k]


def sample(
    sample_id: str,
    image_id: str,
    answer: str = "A",
    question: str = "Biển báo này có ý nghĩa gì?",
) -> dict:
    return {
        "id": sample_id,
        "image_id": image_id,
        "image_path": f"data/raw/train_data/train_images/{image_id}.jpg",
        "question": question,
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": answer,
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }


def point(payload: dict, score: float) -> dict:
    return {"payload": payload, "score": score}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def config(tmp_path: Path) -> dict:
    return {
        "data": {
            "train_split_path": str(tmp_path / "train_split.jsonl"),
            "val_split_path": str(tmp_path / "val_split.jsonl"),
            "train_image_dir": str(tmp_path / "train_images"),
        },
        "retrieval": {
            "text_weight": 0.7,
            "image_weight": 0.3,
            "example_candidate_multiplier": 4,
        },
        "embeddings": {"normalize": False},
        "qdrant": {"url": "http://localhost:6333"},
    }


def test_train_only_indexing_preserves_payload_and_blocks_validation_leakage(tmp_path):
    cfg = config(tmp_path)
    train_rows = [sample("train_1", "img_1", answer="A")]
    val_rows = [sample("val_1", "img_val", answer="SECRET")]
    write_jsonl(Path(cfg["data"]["train_split_path"]), train_rows)
    write_jsonl(Path(cfg["data"]["val_split_path"]), val_rows)
    store = FakeExampleStore()

    count = index_examples(
        cfg,
        split="train",
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        vector_store=store,
    )

    assert count == 1
    assert store.recreated == (2, 2)
    assert [payload["sample_id"] for payload in store.upserted["payloads"]] == ["train_1"]
    assert store.upserted["payloads"][0]["split"] == "train"
    assert store.upserted["payloads"][0]["answer"] == "A"
    assert "SECRET" not in json.dumps(store.upserted["payloads"], ensure_ascii=False)


def test_same_image_group_is_excluded_from_results(tmp_path):
    cfg = config(tmp_path)
    same_image = example_payload(sample("train_same", "img_1"), "train", cfg)
    other_image = example_payload(sample("train_other", "img_2"), "train", cfg)
    store = FakeExampleStore(
        text_points=[point(same_image, 0.99), point(other_image, 0.5)],
    )

    results = retrieve_examples(
        sample("query", "img_1"),
        cfg,
        mode="text",
        text_embedder=FakeTextEmbedder(),
        vector_store=store,
        top_k=3,
    )

    assert [result.payload["sample_id"] for result in results] == ["train_other"]


def test_validation_examples_are_filtered_even_if_store_returns_them(tmp_path):
    cfg = config(tmp_path)
    val_payload = example_payload(sample("val_1", "img_val", answer="SECRET"), "val", cfg)
    train_payload = example_payload(sample("train_1", "img_2", answer="B"), "train", cfg)
    store = FakeExampleStore(text_points=[point(val_payload, 1.0), point(train_payload, 0.4)])

    results = retrieve_examples(
        sample("query", "img_1"),
        cfg,
        mode="text",
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        vector_store=store,
        top_k=5,
    )

    assert [result.payload["sample_id"] for result in results] == ["train_1"]
    assert "SECRET" not in json.dumps(
        [result.to_prompt_example() for result in results],
        ensure_ascii=False,
    )


def test_fusion_score_combines_text_and_image_scores_deterministically(tmp_path):
    cfg = config(tmp_path)
    payload_a = example_payload(sample("train_a", "img_a"), "train", cfg)
    payload_b = example_payload(sample("train_b", "img_b"), "train", cfg)
    store = FakeExampleStore(
        text_points=[point(payload_a, 0.8), point(payload_b, 0.3)],
        image_points=[point(payload_b, 0.9), point(payload_a, 0.1)],
    )

    results = retrieve_examples(
        sample("query", "img_query"),
        cfg,
        mode="fusion",
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        vector_store=store,
        top_k=2,
    )

    assert [result.payload["sample_id"] for result in results] == ["train_a", "train_b"]
    assert results[0]["score"] == pytest.approx((0.7 * 0.8) + (0.3 * 0.1))
    assert results[1]["score"] == pytest.approx((0.7 * 0.3) + (0.3 * 0.9))
    assert [result["rank"] for result in results] == [1, 2]
    assert [query["vector_name"] for query in store.queries] == ["text", "image"]


def test_image_only_mode_uses_image_vector(tmp_path):
    cfg = config(tmp_path)
    payload_a = example_payload(sample("train_a", "img_a"), "train", cfg)
    store = FakeExampleStore(image_points=[point(payload_a, 0.77)])

    results = retrieve_examples(
        sample("query", "img_query"),
        cfg,
        mode="image",
        text_embedder=FakeTextEmbedder(),
        image_embedder=FakeImageEmbedder(),
        vector_store=store,
        top_k=1,
    )

    assert results[0]["retrieval_mode"] == "image"
    assert results[0]["score"] == 0.77
    assert [query["vector_name"] for query in store.queries] == ["image"]


def test_payload_round_trip_into_prompt_ready_example(tmp_path):
    cfg = config(tmp_path)
    payload = example_payload(sample("train_1", "img_1", answer="C"), "train", cfg)
    result = ExampleSearchResult(
        payload=payload,
        score=0.42,
        rank=1,
        retrieval_mode="text",
        text_score=0.42,
    )

    prompt_example = result.to_prompt_example()

    assert prompt_example["sample_id"] == "train_1"
    assert prompt_example["answer"] == "C"
    assert prompt_example["relevant_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert prompt_example["score"] == 0.42


def test_example_text_contains_question_choices_and_type():
    text = build_example_text(sample("train_1", "img_1"))

    assert "Question type: Multiple choice" in text
    assert "Biển báo này có ý nghĩa gì?" in text
    assert "A. Một" in text
