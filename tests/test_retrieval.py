from pathlib import Path

import pytest

from src.retrieval import (
    QdrantTextVectorStore,
    article_payload,
    build_article_text,
    build_query_text,
    evidence_from_payload,
    index_law_articles,
    points_to_evidence,
    retrieve_evidence,
)


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class FakeVectorStore:
    def __init__(self, points=None) -> None:
        self.points = points or []
        self.vector_size = None
        self.upserted = None
        self.query_vector = None
        self.query_top_k = None

    def recreate_collection(self, vector_size: int) -> None:
        self.vector_size = vector_size

    def upsert(self, embeddings, payloads, ids) -> None:
        self.upserted = {
            "embeddings": embeddings,
            "payloads": payloads,
            "ids": ids,
        }

    def query(self, query_vector, top_k: int):
        self.query_vector = query_vector
        self.query_top_k = top_k
        return self.points[:top_k]


class FakeQdrantClient:
    def __init__(self) -> None:
        self.deleted = []
        self.created = None
        self.exists = True

    def collection_exists(self, collection_name):
        assert collection_name == "traffic_law"
        return self.exists

    def delete_collection(self, collection_name):
        self.deleted.append(collection_name)

    def create_collection(self, **kwargs):
        self.created = kwargs


def _article(article_id="22", title="Ý nghĩa sử dụng các biển báo cấm"):
    return {
        "uid": f"{LAW_ID}#{article_id}",
        "law_id": LAW_ID,
        "law_title": "Quy chuẩn báo hiệu đường bộ",
        "article_id": article_id,
        "title": title,
        "content": "Nội dung điều luật.",
        "images": [],
        "tables": [],
    }


def _config(tmp_path: Path):
    return {
        "data": {"processed_law_path": str(tmp_path / "law_articles.jsonl")},
        "qdrant": {"url": "http://localhost:6333", "collection_name": "traffic_law"},
        "retrieval": {"top_k": 5, "batch_size": 64},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(__import__("json").dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_query_text_for_multiple_choice_uses_only_visible_fields():
    query = {
        "id": "train_1",
        "image_id": "image_1",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"B": "Đáp án B", "A": "Đáp án A", "D": "Đáp án D", "C": "Đáp án C"},
        "answer": "E",
        "relevant_articles": [{"law_id": "UNKNOWN", "article_id": "missing"}],
    }

    text = build_query_text(query)

    assert text == (
        "Biển báo này có ý nghĩa gì?\n"
        "A. Đáp án A\n"
        "B. Đáp án B\n"
        "C. Đáp án C\n"
        "D. Đáp án D"
    )
    assert "relevant_articles" not in text
    assert "\nB\n" not in text


def test_query_text_for_yes_no_has_no_choices():
    query = {
        "id": "train_2",
        "image_id": "image_2",
        "question": "Phát biểu này đúng hay sai?",
        "question_type": "Yes/No",
        "answer": "Đúng",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }

    assert build_query_text(query) == "Phát biểu này đúng hay sai?"


def test_article_payload_contains_required_qdrant_fields():
    article = _article()
    payload = article_payload(article, source_path="data/processed/law_articles.jsonl")

    assert payload["uid"] == f"{LAW_ID}#22"
    assert payload["law_id"] == LAW_ID
    assert payload["law_title"] == "Quy chuẩn báo hiệu đường bộ"
    assert payload["article_id"] == "22"
    assert payload["title"] == "Ý nghĩa sử dụng các biển báo cấm"
    assert payload["content"] == "Nội dung điều luật."
    assert payload["source_path"] == "data/processed/law_articles.jsonl"


def test_payload_conversion_to_evidence():
    payload = article_payload(_article(), source_path="law.jsonl")
    evidence = evidence_from_payload(payload, score=0.91, rank=1)

    assert evidence.law_id == LAW_ID
    assert evidence.article_id == "22"
    assert evidence.score == 0.91
    assert evidence.rank == 1
    assert evidence.retrieval_method == "text"
    assert evidence.metadata["source_path"] == "law.jsonl"


def test_points_to_evidence_deduplicates_and_reranks():
    payload_22 = article_payload(_article("22"), source_path="law.jsonl")
    payload_41 = article_payload(_article("41", "Biển phụ"), source_path="law.jsonl")
    points = [
        {"payload": payload_22, "score": 0.9},
        {"payload": payload_22, "score": 0.8},
        {"payload": payload_41, "score": 0.7},
    ]

    evidence = points_to_evidence(points, top_k=5)

    assert [item.uid for item in evidence] == [f"{LAW_ID}#22", f"{LAW_ID}#41"]
    assert [item.rank for item in evidence] == [1, 2]


def test_index_law_articles_fails_helpfully_when_preprocess_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="mode preprocess"):
        index_law_articles(_config(tmp_path), embedder=FakeEmbedder(), vector_store=FakeVectorStore())


def test_index_law_articles_uses_one_cosine_text_collection_shape(tmp_path):
    path = tmp_path / "law_articles.jsonl"
    _write_jsonl(path, [_article("22"), _article("41", "Biển phụ")])
    embedder = FakeEmbedder()
    store = FakeVectorStore()

    count = index_law_articles(_config(tmp_path), embedder=embedder, vector_store=store)

    assert count == 2
    assert store.vector_size == 3
    assert len(store.upserted["payloads"]) == 2
    assert store.upserted["payloads"][0]["uid"] == f"{LAW_ID}#22"
    assert build_article_text(_article("22")) in embedder.texts[0]


def test_qdrant_collection_uses_one_cosine_text_vector():
    client = FakeQdrantClient()
    store = object.__new__(QdrantTextVectorStore)
    store.collection_name = "traffic_law"
    store.batch_size = 64
    store.client = client

    store.recreate_collection(vector_size=384)

    assert client.deleted == ["traffic_law"]
    assert client.created["collection_name"] == "traffic_law"
    vector_config = client.created["vectors_config"]
    assert vector_config.size == 384
    assert vector_config.distance == "Cosine"


def test_retrieve_evidence_queries_top_k_unique_text_results():
    payload_22 = article_payload(_article("22"), source_path="law.jsonl")
    payload_41 = article_payload(_article("41", "Biển phụ"), source_path="law.jsonl")
    embedder = FakeEmbedder()
    store = FakeVectorStore(
        points=[
            {"payload": payload_22, "score": 0.9},
            {"payload": payload_22, "score": 0.8},
            {"payload": payload_41, "score": 0.7},
        ]
    )

    evidence = retrieve_evidence(
        {
            "id": "train_1",
            "image_id": "image_1",
            "question": "Câu hỏi?",
        },
        {"retrieval": {"top_k": 1}},
        embedder=embedder,
        vector_store=store,
        top_k=1,
    )

    assert len(evidence) == 1
    assert evidence[0].uid == f"{LAW_ID}#22"
    assert store.query_top_k == 3
