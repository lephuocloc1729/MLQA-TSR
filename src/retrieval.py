from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models

from src.data_utils import load_law_articles
from src.schemas import Evidence, Query
from src.utils import load_config, read_json


class TextEmbedder(Protocol):
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class TextVectorStore(Protocol):
    def recreate_collection(self, vector_size: int) -> None:
        ...

    def upsert(
        self,
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        ...

    def query(self, query_vector: list[float], top_k: int) -> list[Any]:
        ...


class SentenceTransformerEmbedder:
    """Thin wrapper so tests can inject a fake embedder without model downloads."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32").tolist()


class QdrantTextVectorStore:
    """Qdrant adapter for one unnamed cosine text vector."""

    def __init__(
        self,
        url: str,
        collection_name: str,
        batch_size: int = 64,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.client = QdrantClient(url=url, api_key=api_key)

    def recreate_collection(self, vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
            on_disk_payload=True,
        )

    def upsert(
        self,
        embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        if len(embeddings) != len(payloads) or len(embeddings) != len(ids):
            raise ValueError("Embeddings, payloads, and ids must have the same length")

        for start in range(0, len(embeddings), self.batch_size):
            end = start + self.batch_size
            points = [
                models.PointStruct(
                    id=ids[index],
                    vector=embeddings[index],
                    payload=payloads[index],
                )
                for index in range(start, min(end, len(embeddings)))
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def query(self, query_vector: list[float], top_k: int) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return list(response.points)


def build_query_text(query: Query | dict[str, Any]) -> str:
    """Build retrieval text from user-visible inputs only, never gold labels."""
    if isinstance(query, Query):
        question = query.question
        choices = query.choices
    else:
        question = query.get("question")
        choices = query.get("choices") or {}

    if not question:
        raise ValueError("A retrieval query requires a non-empty question")
    if not isinstance(choices, dict):
        raise ValueError("Query choices must be a dictionary when provided")

    parts = [unicodedata.normalize("NFC", str(question)).strip()]
    for key in sorted(choices):
        choice_key = unicodedata.normalize("NFC", str(key)).strip().upper()
        choice_text = unicodedata.normalize("NFC", str(choices[key])).strip()
        parts.append(f"{choice_key}. {choice_text}")
    return "\n".join(parts)


def build_article_text(article: dict[str, Any]) -> str:
    return "\n".join(
        [
            f"{article['law_id']} - {article['law_title']}",
            f"{article['article_id']}. {article['title']}",
            article["content"],
        ]
    )


def article_payload(article: dict[str, Any], source_path: str) -> dict[str, Any]:
    return {
        "uid": article["uid"],
        "law_id": article["law_id"],
        "law_title": article["law_title"],
        "article_id": article["article_id"],
        "title": article["title"],
        "content": article["content"],
        "images": article.get("images", []),
        "tables": article.get("tables", []),
        "source_path": source_path,
    }


def point_id_from_uid(uid: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"traffic-legal-vlm:{uid}"))


def _point_payload(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        return point["payload"]
    return point.payload or {}


def _point_score(point: Any) -> float | None:
    if isinstance(point, dict):
        return point.get("score")
    return point.score


def evidence_from_payload(
    payload: dict[str, Any],
    score: float | None,
    rank: int,
) -> Evidence:
    return Evidence(
        law_id=payload["law_id"],
        article_id=payload["article_id"],
        title=payload.get("title"),
        content=payload["content"],
        score=score,
        rank=rank,
        retrieval_method="text",
        metadata={
            "uid": payload.get("uid"),
            "law_title": payload.get("law_title"),
            "source_path": payload.get("source_path"),
            "images": payload.get("images", []),
            "tables": payload.get("tables", []),
        },
    )


def points_to_evidence(points: Iterable[Any], top_k: int) -> list[Evidence]:
    evidence: list[Evidence] = []
    seen_uids: set[str] = set()
    for point in points:
        payload = _point_payload(point)
        uid = payload.get("uid") or f"{payload.get('law_id')}#{payload.get('article_id')}"
        if uid in seen_uids:
            continue
        seen_uids.add(uid)
        evidence.append(
            evidence_from_payload(
                payload=payload,
                score=_point_score(point),
                rank=len(evidence) + 1,
            )
        )
        if len(evidence) >= top_k:
            break
    return evidence


def make_embedder(config: dict) -> SentenceTransformerEmbedder:
    retrieval_config = config.get("retrieval", {})
    return SentenceTransformerEmbedder(
        model_name=retrieval_config.get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
        device=retrieval_config.get("device"),
    )


def make_vector_store(config: dict) -> QdrantTextVectorStore:
    qdrant_config = config["qdrant"]
    retrieval_config = config.get("retrieval", {})
    return QdrantTextVectorStore(
        url=qdrant_config["url"],
        collection_name=qdrant_config["collection_name"],
        batch_size=retrieval_config.get("batch_size", 64),
        api_key=qdrant_config.get("api_key"),
    )


def index_law_articles(
    config: dict,
    embedder: TextEmbedder | None = None,
    vector_store: TextVectorStore | None = None,
) -> int:
    processed_law_path = Path(config["data"]["processed_law_path"])
    if not processed_law_path.exists():
        raise FileNotFoundError(
            f"Processed LawDB not found at {processed_law_path}. "
            "Run `python -m src.data_utils --mode preprocess` first."
        )

    articles = load_law_articles(processed_law_path)
    if not articles:
        raise ValueError(f"No law articles found in {processed_law_path}")

    embedder = embedder or make_embedder(config)
    vector_store = vector_store or make_vector_store(config)

    texts = [build_article_text(article) for article in articles]
    embeddings = embedder.embed_texts(texts)
    if not embeddings or not embeddings[0]:
        raise ValueError("Text embedder returned empty embeddings")

    vector_store.recreate_collection(vector_size=len(embeddings[0]))
    vector_store.upsert(
        embeddings=embeddings,
        payloads=[
            article_payload(article, source_path=str(processed_law_path))
            for article in articles
        ],
        ids=[point_id_from_uid(article["uid"]) for article in articles],
    )
    return len(articles)


def retrieve_evidence(
    query: Query | dict[str, Any],
    config: dict,
    embedder: TextEmbedder | None = None,
    vector_store: TextVectorStore | None = None,
    top_k: int | None = None,
) -> list[Evidence]:
    top_k = top_k or config.get("retrieval", {}).get("top_k", 5)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    embedder = embedder or make_embedder(config)
    vector_store = vector_store or make_vector_store(config)

    query_text = build_query_text(query)
    query_vector = embedder.embed_texts([query_text])[0]
    points = vector_store.query(query_vector=query_vector, top_k=top_k * 3)
    return points_to_evidence(points, top_k=top_k)


def load_sample_by_id(config: dict, sample_id: str) -> dict[str, Any]:
    samples = read_json(config["data"]["train_path"])
    for sample in samples:
        if sample.get("id") == sample_id:
            return sample
    raise KeyError(f"Sample {sample_id!r} not found in {config['data']['train_path']}")


def run_index(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    count = index_law_articles(config)
    print(f"Indexed {count} law articles into Qdrant collection {config['qdrant']['collection_name']!r}")


def run_retrieve(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sample = load_sample_by_id(config, args.sample_id)
    evidence = retrieve_evidence(sample, config, top_k=args.top_k)
    print(
        json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Text retrieval over processed LawDB")
    parser.add_argument("--mode", choices=["index", "retrieve"], default="index")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample-id", help="Training sample ID for retrieve mode")
    parser.add_argument("--top-k", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "index":
        run_index(args)
    elif args.mode == "retrieve":
        if not args.sample_id:
            parser.error("--sample-id is required when --mode retrieve")
        run_retrieve(args)


if __name__ == "__main__":
    main()
