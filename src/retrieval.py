from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Protocol
from uuid import uuid5, NAMESPACE_URL

from qdrant_client import QdrantClient, models

from src.data_utils import (
    attach_train_image_path,
    load_law_articles,
    load_processed_law_article_index,
    load_split_samples,
    make_article_uid,
    reference_uid,
    resolve_law_reference,
)
from src.schemas import Evidence, Query
from src.utils import (
    EmbeddingCache,
    file_sha256,
    l2_normalize_vectors,
    load_config,
    read_json,
    stable_json_hash,
)
from src.vision import ImageEmbeddingAdapter, make_image_embedder


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


class ImageEmbedder(Protocol):
    model_name: str

    def embed_images(self, paths: list[str | Path]) -> list[list[float]]:
        ...


class ExampleVectorStore(Protocol):
    def recreate_collection(self, text_vector_size: int, image_vector_size: int) -> None:
        ...

    def upsert_examples(
        self,
        text_embeddings: list[list[float]],
        image_embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        ...

    def query(self, query_vector: list[float], vector_name: str, top_k: int) -> list[Any]:
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


def normalize_embedding_batch(
    embeddings: list[list[float]],
    normalize: bool = True,
) -> list[list[float]]:
    if not embeddings:
        return []
    return l2_normalize_vectors(embeddings) if normalize else embeddings


def text_data_hash(texts: list[str]) -> str:
    return stable_json_hash({"texts": texts})


def embedding_model_name(embedder: TextEmbedder, fallback: str) -> str:
    return str(getattr(embedder, "model_name", fallback))


def embedding_cache_config(config: dict) -> dict[str, Any]:
    return config.get("embeddings", {})


def text_embedding_config(config: dict) -> dict[str, Any]:
    return embedding_cache_config(config).get("text", {})


def text_embedding_model_name(config: dict) -> str:
    return text_embedding_config(config).get(
        "model_name",
        config.get("retrieval", {}).get(
            "embedding_model",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        ),
    )


def embedding_cache_dir(config: dict) -> str:
    return embedding_cache_config(config).get("cache_dir", "data/outputs/embeddings")


def should_normalize_embeddings(config: dict) -> bool:
    return bool(embedding_cache_config(config).get("normalize", True))


def text_cache_enabled(config: dict, flag_name: str) -> bool:
    return bool(text_embedding_config(config).get(flag_name, False))


def embed_texts(
    texts: list[str],
    embedder: TextEmbedder,
    normalize: bool = True,
) -> list[list[float]]:
    if not texts:
        return []
    return normalize_embedding_batch(embedder.embed_texts(texts), normalize=normalize)


def embed_texts_cached(
    texts: list[str],
    embedder: TextEmbedder,
    cache_dir: str | Path,
    cache_key: str,
    model_name: str,
    data_hash: str | None = None,
    normalize: bool = True,
) -> list[list[float]]:
    """Embed text with a metadata-validated local cache."""
    data_hash = data_hash or text_data_hash(texts)
    cache = EmbeddingCache(cache_dir)
    metadata = {
        "model_name": model_name,
        "modality": "text",
        "data_hash": data_hash,
        "normalized": normalize,
    }

    if cache.exists(cache_key):
        manifest = cache.read_manifest(cache_key)
        return cache.read(
            cache_key,
            {
                **metadata,
                "dimension": manifest.get("dimension"),
            },
        )

    embeddings = embed_texts(texts, embedder=embedder, normalize=normalize)
    cache.write(cache_key, embeddings, metadata=metadata)
    return embeddings


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


class QdrantExampleVectorStore:
    """Qdrant adapter for train-example search with named text/image vectors."""

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

    def recreate_collection(self, text_vector_size: int, image_vector_size: int) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                "text": models.VectorParams(
                    size=text_vector_size,
                    distance=models.Distance.COSINE,
                ),
                "image": models.VectorParams(
                    size=image_vector_size,
                    distance=models.Distance.COSINE,
                ),
            },
            on_disk_payload=True,
        )

    def upsert_examples(
        self,
        text_embeddings: list[list[float]],
        image_embeddings: list[list[float]],
        payloads: list[dict[str, Any]],
        ids: list[str],
    ) -> None:
        sizes = {len(text_embeddings), len(image_embeddings), len(payloads), len(ids)}
        if len(sizes) != 1:
            raise ValueError(
                "Text embeddings, image embeddings, payloads, and ids must have "
                "the same length"
            )

        for start in range(0, len(payloads), self.batch_size):
            end = min(start + self.batch_size, len(payloads))
            points = [
                models.PointStruct(
                    id=ids[index],
                    vector={
                        "text": text_embeddings[index],
                        "image": image_embeddings[index],
                    },
                    payload=payloads[index],
                )
                for index in range(start, end)
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)

    def query(
        self,
        query_vector: list[float],
        vector_name: str,
        top_k: int,
    ) -> list[models.ScoredPoint]:
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=vector_name,
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


def build_example_text(sample: dict[str, Any]) -> str:
    question_type = sample.get("question_type", "")
    return "\n".join(
        part
        for part in [
            f"Question type: {question_type}" if question_type else "",
            build_query_text(sample),
        ]
        if part
    )


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


def example_point_id(sample_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"traffic-legal-vlm:example:{sample_id}"))


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
        model_name=text_embedding_model_name(config),
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


def make_example_vector_store(config: dict) -> QdrantExampleVectorStore:
    qdrant_config = config["qdrant"]
    retrieval_config = config.get("retrieval", {})
    return QdrantExampleVectorStore(
        url=qdrant_config["url"],
        collection_name=qdrant_config.get(
            "example_collection_name",
            "traffic_qa_examples",
        ),
        batch_size=retrieval_config.get("batch_size", 64),
        api_key=qdrant_config.get("api_key"),
    )


def example_collection_exists(config: dict) -> bool:
    store = make_example_vector_store(config)
    return bool(store.client.collection_exists(store.collection_name))


def example_payload(sample: dict[str, Any], split: str, config: dict) -> dict[str, Any]:
    sample = attach_train_image_path(sample, config)
    return {
        "sample_id": sample["id"],
        "image_id": sample["image_id"],
        "question": sample["question"],
        "question_type": sample.get("question_type"),
        "choices": sample.get("choices", {}),
        "answer": sample.get("answer"),
        "relevant_articles": sample.get("relevant_articles", []),
        "image_path": sample["image_path"],
        "split": split,
    }


def load_example_samples(config: dict, split: str = "train") -> list[dict[str, Any]]:
    if split != "train":
        raise ValueError("Example index is leakage-safe and only supports split='train'")
    samples = load_split_samples(config, split)
    return [attach_train_image_path(sample, config) for sample in samples]


def make_image_embedder_from_config(config: dict) -> ImageEmbeddingAdapter:
    return make_image_embedder(config)


def index_examples(
    config: dict,
    split: str = "train",
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    vector_store: ExampleVectorStore | None = None,
) -> int:
    samples = load_example_samples(config, split=split)
    if not samples:
        raise ValueError(f"No samples found for split {split!r}")

    text_embedder = text_embedder or make_embedder(config)
    image_embedder = image_embedder or make_image_embedder_from_config(config)
    vector_store = vector_store or make_example_vector_store(config)

    payloads = [example_payload(sample, split=split, config=config) for sample in samples]
    texts = [build_example_text(payload) for payload in payloads]
    image_paths = [payload["image_path"] for payload in payloads]
    text_embeddings = embed_texts(
        texts,
        embedder=text_embedder,
        normalize=should_normalize_embeddings(config),
    )
    image_embeddings = image_embedder.embed_images(image_paths)
    if not text_embeddings or not text_embeddings[0]:
        raise ValueError("Text embedder returned empty example embeddings")
    if not image_embeddings or not image_embeddings[0]:
        raise ValueError("Image embedder returned empty example embeddings")

    vector_store.recreate_collection(
        text_vector_size=len(text_embeddings[0]),
        image_vector_size=len(image_embeddings[0]),
    )
    vector_store.upsert_examples(
        text_embeddings=text_embeddings,
        image_embeddings=image_embeddings,
        payloads=payloads,
        ids=[example_point_id(payload["sample_id"]) for payload in payloads],
    )
    return len(samples)


class ExampleSearchResult(dict):
    """Prompt-ready solved example result."""

    def __init__(
        self,
        payload: dict[str, Any],
        score: float,
        rank: int,
        retrieval_mode: str,
        text_score: float | None = None,
        image_score: float | None = None,
    ) -> None:
        super().__init__(
            payload=payload,
            score=score,
            rank=rank,
            retrieval_mode=retrieval_mode,
            text_score=text_score,
            image_score=image_score,
        )

    @property
    def payload(self) -> dict[str, Any]:
        return self["payload"]

    def to_prompt_example(self) -> dict[str, Any]:
        payload = self.payload
        return {
            "sample_id": payload["sample_id"],
            "image_id": payload["image_id"],
            "question": payload["question"],
            "question_type": payload.get("question_type"),
            "choices": payload.get("choices", {}),
            "answer": payload.get("answer"),
            "relevant_articles": payload.get("relevant_articles", []),
            "image_path": payload.get("image_path"),
            "split": payload.get("split"),
            "score": self["score"],
            "retrieval_mode": self["retrieval_mode"],
        }


class FusedEvidenceResult(dict):
    """Structured retrieval artifact with final `Evidence` plus diagnostics."""

    @property
    def evidence(self) -> list[Evidence]:
        return self["evidence"]

    @property
    def diagnostics(self) -> list[dict[str, Any]]:
        return self["diagnostics"]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "direct_law": self["direct_law"],
            "example_votes": self["example_votes"],
            "diagnostics": self["diagnostics"],
            "evidence": [
                evidence.model_dump(mode="json") for evidence in self["evidence"]
            ],
        }


def _query_points_by_mode(
    vector_store: ExampleVectorStore,
    mode: str,
    text_vector: list[float] | None,
    image_vector: list[float] | None,
    top_k: int,
) -> dict[str, list[Any]]:
    if mode == "text":
        if text_vector is None:
            raise ValueError("text_vector is required for text example retrieval")
        return {"text": vector_store.query(text_vector, vector_name="text", top_k=top_k)}
    if mode == "image":
        if image_vector is None:
            raise ValueError("image_vector is required for image example retrieval")
        return {"image": vector_store.query(image_vector, vector_name="image", top_k=top_k)}
    if mode == "fusion":
        if text_vector is None or image_vector is None:
            raise ValueError("text_vector and image_vector are required for fusion")
        return {
            "text": vector_store.query(text_vector, vector_name="text", top_k=top_k),
            "image": vector_store.query(image_vector, vector_name="image", top_k=top_k),
        }
    raise ValueError("mode must be one of: text, image, fusion")


def _merge_example_points(
    point_groups: dict[str, list[Any]],
    query_image_id: str | None,
    mode: str,
    top_k: int,
    text_weight: float,
    image_weight: float,
) -> list[ExampleSearchResult]:
    merged: dict[str, dict[str, Any]] = {}
    for vector_name, points in point_groups.items():
        for point in points:
            payload = _point_payload(point)
            if payload.get("split") != "train":
                continue
            if query_image_id and payload.get("image_id") == query_image_id:
                continue

            sample_id = payload.get("sample_id")
            if not sample_id:
                continue
            item = merged.setdefault(
                sample_id,
                {"payload": payload, "text_score": None, "image_score": None},
            )
            score = float(_point_score(point) or 0.0)
            score_key = f"{vector_name}_score"
            item[score_key] = max(score, item[score_key] or score)

    results: list[ExampleSearchResult] = []
    for item in merged.values():
        text_score = item["text_score"]
        image_score = item["image_score"]
        if mode == "text":
            score = float(text_score or 0.0)
        elif mode == "image":
            score = float(image_score or 0.0)
        else:
            score = (text_weight * float(text_score or 0.0)) + (
                image_weight * float(image_score or 0.0)
            )
        results.append(
            ExampleSearchResult(
                payload=item["payload"],
                score=score,
                rank=0,
                retrieval_mode=mode,
                text_score=text_score,
                image_score=image_score,
            )
        )

    results.sort(key=lambda result: (-result["score"], result.payload["sample_id"]))
    for rank, result in enumerate(results[:top_k], start=1):
        result["rank"] = rank
    return results[:top_k]


def _as_example_payload(example: ExampleSearchResult | dict[str, Any]) -> dict[str, Any]:
    if isinstance(example, ExampleSearchResult):
        return example.payload
    if "payload" in example and isinstance(example["payload"], dict):
        return example["payload"]
    return example


def _as_example_score(example: ExampleSearchResult | dict[str, Any]) -> float:
    try:
        score = example.get("score", 1.0)
    except AttributeError:
        score = 1.0
    return float(score if score is not None else 1.0)


def _as_example_rank(
    example: ExampleSearchResult | dict[str, Any],
    default_rank: int,
) -> int:
    try:
        rank = example.get("rank", default_rank)
    except AttributeError:
        rank = default_rank
    return max(1, int(rank or default_rank))


def rank_weighted_example_votes(
    examples: list[ExampleSearchResult | dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate example citations using example score divided by example rank."""
    votes: dict[str, dict[str, Any]] = {}
    diagnostics: list[dict[str, Any]] = []
    for default_rank, example in enumerate(examples, start=1):
        payload = _as_example_payload(example)
        sample_id = payload.get("sample_id") or payload.get("id") or f"example_{default_rank}"
        rank = _as_example_rank(example, default_rank)
        example_score = _as_example_score(example)
        vote_increment = example_score / rank

        for reference in payload.get("relevant_articles", []):
            try:
                uid = reference_uid(reference)
            except ValueError as exc:
                diagnostics.append(
                    {
                        "type": "invalid_example_reference",
                        "sample_id": sample_id,
                        "reference": dict(reference) if isinstance(reference, dict) else reference,
                        "reason": str(exc),
                    }
                )
                continue
            item = votes.setdefault(
                uid,
                {
                    "uid": uid,
                    "law_id": reference["law_id"],
                    "article_id": reference["article_id"],
                    "score": 0.0,
                    "count": 0,
                    "sources": [],
                },
            )
            item["score"] += vote_increment
            item["count"] += 1
            item["sources"].append(
                {
                    "sample_id": sample_id,
                    "rank": rank,
                    "example_score": example_score,
                    "vote_increment": vote_increment,
                }
            )
    return votes, diagnostics


def _direct_evidence_components(
    direct_evidence: list[Evidence],
) -> dict[str, dict[str, Any]]:
    components: dict[str, dict[str, Any]] = {}
    for evidence in direct_evidence:
        uid = evidence.uid
        score = float(evidence.score if evidence.score is not None else 0.0)
        if score == 0.0 and evidence.rank:
            score = 1.0 / evidence.rank
        components[uid] = {
            "uid": uid,
            "law_id": evidence.law_id,
            "article_id": evidence.article_id,
            "score": score,
            "rank": evidence.rank,
            "source": "direct_law",
        }
    return components


def _resolve_fused_evidence(
    uid: str,
    law_id: str,
    article_id: str,
    article_index: dict[str, dict],
    score: float,
    rank: int,
    metadata: dict[str, Any],
) -> Evidence:
    article = resolve_law_reference(
        {"law_id": law_id, "article_id": article_id},
        article_index,
    )
    return Evidence(
        law_id=article["law_id"],
        article_id=article["article_id"],
        title=article["title"],
        content=article["content"],
        score=score,
        rank=rank,
        retrieval_method="fusion",
        metadata={
            "uid": uid,
            "law_title": article.get("law_title"),
            "images": article.get("images", []),
            "tables": article.get("tables", []),
            **metadata,
        },
    )


def fuse_legal_evidence(
    direct_evidence: list[Evidence],
    examples: list[ExampleSearchResult | dict[str, Any]],
    article_index: dict[str, dict],
    top_k: int = 5,
    direct_weight: float = 1.0,
    example_vote_weight: float = 1.0,
) -> FusedEvidenceResult:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    direct_components = _direct_evidence_components(direct_evidence)
    example_votes, diagnostics = rank_weighted_example_votes(examples)
    candidate_uids = sorted(set(direct_components) | set(example_votes))
    ranked_items: list[dict[str, Any]] = []

    for uid in candidate_uids:
        direct = direct_components.get(uid)
        vote = example_votes.get(uid)
        law_id = (direct or vote)["law_id"]
        article_id = (direct or vote)["article_id"]
        direct_score = float(direct["score"]) if direct else 0.0
        example_vote_score = float(vote["score"]) if vote else 0.0
        fusion_score = (direct_weight * direct_score) + (
            example_vote_weight * example_vote_score
        )
        ranked_items.append(
            {
                "uid": uid,
                "law_id": law_id,
                "article_id": article_id,
                "direct_score": direct_score,
                "direct_rank": direct.get("rank") if direct else None,
                "example_vote_score": example_vote_score,
                "example_vote_count": vote.get("count", 0) if vote else 0,
                "example_sources": vote.get("sources", []) if vote else [],
                "fusion_score": fusion_score,
                "source_mode": (
                    "direct+examples"
                    if direct and vote
                    else "direct"
                    if direct
                    else "examples"
                ),
            }
        )

    ranked_items.sort(key=lambda item: (-item["fusion_score"], item["uid"]))
    evidence: list[Evidence] = []
    for item in ranked_items:
        try:
            evidence.append(
                _resolve_fused_evidence(
                    uid=item["uid"],
                    law_id=item["law_id"],
                    article_id=item["article_id"],
                    article_index=article_index,
                    score=item["fusion_score"],
                    rank=len(evidence) + 1,
                    metadata={
                        "direct_score": item["direct_score"],
                        "direct_rank": item["direct_rank"],
                        "example_vote_score": item["example_vote_score"],
                        "example_vote_count": item["example_vote_count"],
                        "example_sources": item["example_sources"],
                        "fusion_score": item["fusion_score"],
                        "source_mode": item["source_mode"],
                    },
                )
            )
        except KeyError as exc:
            diagnostics.append(
                {
                    "type": "unknown_law_article",
                    "uid": item["uid"],
                    "law_id": item["law_id"],
                    "article_id": item["article_id"],
                    "reason": str(exc),
                    "source_mode": item["source_mode"],
                }
            )
        if len(evidence) >= top_k:
            break

    return FusedEvidenceResult(
        direct_law=[evidence_item.uid for evidence_item in direct_evidence],
        example_votes=[
            item["uid"]
            for item in sorted(
                example_votes.values(),
                key=lambda item: (-item["score"], item["uid"]),
            )
        ],
        diagnostics=diagnostics,
        evidence=evidence,
    )


def retrieve_examples(
    query: dict[str, Any],
    config: dict,
    mode: str = "fusion",
    text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    vector_store: ExampleVectorStore | None = None,
    top_k: int | None = None,
) -> list[ExampleSearchResult]:
    top_k = top_k or config.get("retrieval", {}).get("example_top_k", 3)
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    query = attach_train_image_path(query, config)
    vector_store = vector_store or make_example_vector_store(config)
    candidate_multiplier = max(
        1,
        int(config.get("retrieval", {}).get("example_candidate_multiplier", 5)),
    )
    candidate_k = top_k * candidate_multiplier

    text_vector = None
    image_vector = None
    if mode in {"text", "fusion"}:
        text_embedder = text_embedder or make_embedder(config)
        text_vector = embed_texts(
            [build_example_text(query)],
            embedder=text_embedder,
            normalize=should_normalize_embeddings(config),
        )[0]
    if mode in {"image", "fusion"}:
        image_embedder = image_embedder or make_image_embedder_from_config(config)
        image_vector = image_embedder.embed_images([query["image_path"]])[0]

    point_groups = _query_points_by_mode(
        vector_store=vector_store,
        mode=mode,
        text_vector=text_vector,
        image_vector=image_vector,
        top_k=candidate_k,
    )
    retrieval_config = config.get("retrieval", {})
    return _merge_example_points(
        point_groups=point_groups,
        query_image_id=query.get("image_id"),
        mode=mode,
        top_k=top_k,
        text_weight=float(retrieval_config.get("text_weight", 0.7)),
        image_weight=float(retrieval_config.get("image_weight", 0.3)),
    )


def retrieve_fused_evidence(
    query: dict[str, Any],
    config: dict,
    text_embedder: TextEmbedder | None = None,
    law_vector_store: TextVectorStore | None = None,
    example_text_embedder: TextEmbedder | None = None,
    image_embedder: ImageEmbedder | None = None,
    example_vector_store: ExampleVectorStore | None = None,
    top_k: int | None = None,
    example_top_k: int | None = None,
    example_mode: str = "fusion",
    allow_example_failure: bool | None = None,
) -> FusedEvidenceResult:
    retrieval_config = config.get("retrieval", {})
    top_k = top_k or retrieval_config.get("top_k", 5)
    example_top_k = example_top_k or retrieval_config.get("example_top_k", 3)
    allow_example_failure = (
        retrieval_config.get("fusion_allow_example_failure", True)
        if allow_example_failure is None
        else allow_example_failure
    )
    article_index = load_processed_law_article_index(config)
    direct_evidence = retrieve_evidence(
        query,
        config,
        embedder=text_embedder,
        vector_store=law_vector_store,
        top_k=top_k,
    )

    diagnostics: list[dict[str, Any]] = []
    examples: list[ExampleSearchResult | dict[str, Any]] = []
    try:
        if (
            example_vector_store is None
            and allow_example_failure
            and not example_collection_exists(config)
        ):
            raise RuntimeError(
                "Example collection is unavailable. "
                "Run `python -m src.retrieval --mode index-examples --split train` first."
            )
        examples = retrieve_examples(
            query,
            config,
            mode=example_mode,
            text_embedder=example_text_embedder or text_embedder,
            image_embedder=image_embedder,
            vector_store=example_vector_store,
            top_k=example_top_k,
        )
    except Exception as exc:
        if not allow_example_failure:
            raise
        diagnostics.append(
            {
                "type": "example_retrieval_failed",
                "reason": str(exc),
                "example_mode": example_mode,
            }
        )

    result = fuse_legal_evidence(
        direct_evidence=direct_evidence,
        examples=examples,
        article_index=article_index,
        top_k=top_k,
        direct_weight=float(retrieval_config.get("fusion_direct_weight", 1.0)),
        example_vote_weight=float(retrieval_config.get("fusion_example_vote_weight", 1.0)),
    )
    result["diagnostics"].extend(diagnostics)
    return result


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
    model_name = embedding_model_name(embedder, text_embedding_model_name(config))
    if text_cache_enabled(config, "cache_articles"):
        embeddings = embed_texts_cached(
            texts,
            embedder=embedder,
            cache_dir=embedding_cache_dir(config),
            cache_key="law_articles_text",
            model_name=model_name,
            data_hash=file_sha256(processed_law_path),
            normalize=should_normalize_embeddings(config),
        )
    else:
        embeddings = embed_texts(
            texts,
            embedder=embedder,
            normalize=should_normalize_embeddings(config),
        )
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
    model_name = embedding_model_name(embedder, text_embedding_model_name(config))
    if text_cache_enabled(config, "cache_queries"):
        query_vector = embed_texts_cached(
            [query_text],
            embedder=embedder,
            cache_dir=embedding_cache_dir(config),
            cache_key=f"query_text_{text_data_hash([query_text])}",
            model_name=model_name,
            data_hash=text_data_hash([query_text]),
            normalize=should_normalize_embeddings(config),
        )[0]
    else:
        query_vector = embed_texts(
            [query_text],
            embedder=embedder,
            normalize=should_normalize_embeddings(config),
        )[0]
    points = vector_store.query(query_vector=query_vector, top_k=top_k * 3)
    return points_to_evidence(points, top_k=top_k)


def load_sample_by_id(config: dict, sample_id: str) -> dict[str, Any]:
    samples = read_json(config["data"]["train_path"])
    for sample in samples:
        if sample.get("id") == sample_id:
            return sample
    raise KeyError(f"Sample {sample_id!r} not found in {config['data']['train_path']}")


def load_query_sample_by_id(config: dict, sample_id: str) -> dict[str, Any]:
    for split in ("train", "val"):
        try:
            for sample in load_split_samples(config, split):
                if sample.get("id") == sample_id:
                    return attach_train_image_path(sample, config)
        except FileNotFoundError:
            continue
    return attach_train_image_path(load_sample_by_id(config, sample_id), config)


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


def run_index_examples(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    count = index_examples(config, split=args.split)
    collection_name = config["qdrant"].get("example_collection_name", "traffic_qa_examples")
    print(f"Indexed {count} training examples into Qdrant collection {collection_name!r}")


def run_retrieve_examples(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sample = load_query_sample_by_id(config, args.sample_id)
    examples = retrieve_examples(
        sample,
        config,
        mode=args.retrieval_mode,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            [item.to_prompt_example() for item in examples],
            ensure_ascii=False,
            indent=2,
        )
    )


def run_retrieve_fusion(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    sample = load_query_sample_by_id(config, args.sample_id)
    result = retrieve_fused_evidence(
        sample,
        config,
        top_k=args.top_k,
        example_mode=args.retrieval_mode,
    )
    print(json.dumps(result.to_jsonable(), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Text retrieval over processed LawDB")
    parser.add_argument(
        "--mode",
        choices=[
            "index",
            "retrieve",
            "index-examples",
            "retrieve-examples",
            "retrieve-fusion",
        ],
        default="index",
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--sample-id", help="Training sample ID for retrieve mode")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument(
        "--retrieval-mode",
        default="fusion",
        choices=["text", "image", "fusion"],
        help="Example retrieval mode for retrieve-examples.",
    )
    args = parser.parse_args()

    if args.mode == "index":
        run_index(args)
    elif args.mode == "retrieve":
        if not args.sample_id:
            parser.error("--sample-id is required when --mode retrieve")
        run_retrieve(args)
    elif args.mode == "index-examples":
        run_index_examples(args)
    elif args.mode == "retrieve-examples":
        if not args.sample_id:
            parser.error("--sample-id is required when --mode retrieve-examples")
        run_retrieve_examples(args)
    elif args.mode == "retrieve-fusion":
        if not args.sample_id:
            parser.error("--sample-id is required when --mode retrieve-fusion")
        run_retrieve_fusion(args)


if __name__ == "__main__":
    main()
