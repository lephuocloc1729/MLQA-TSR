from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.evaluate import mean, score_retrieval_sample
from src.data_utils import resolve_law_reference
from src.lowcost_features import manifest_output_path
from src.schemas import Evidence, RetrievalMethod
from src.utils import (
    load_config,
    read_json,
    read_jsonl,
    stable_json_hash,
    utc_now_iso,
    write_json,
    write_jsonl,
)


TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_general_feature_vector"
OBJECT_VECTOR_NAME = "image_object_feature_list_vector"
OBJECT_DIMENSION_NAME = "image_object_feature_vector"
LOWCOST_VECTOR_NAMES = [TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, OBJECT_VECTOR_NAME]
DEFAULT_COLLECTION_NAME = "traffic_train_examples_lowcost"
INDEX_MANIFEST_SCHEMA_VERSION = "lowcost-qdrant-index-v1"
TASK1_QUERY_MODES = {"text_image", "text_image_object"}
TASK1_DEFAULT_QUERY_MODE = "text_image_object"
TASK1_DEFAULT_LIMITS = {
    "text_limit": 10,
    "image_limit": 5,
    "object_limit": 3,
}
TASK1_EVIDENCE_SCHEMA_VERSION = "lowcost-task1-evidence-v1"


class LowCostVectorStore(Protocol):
    collection_name: str
    uses_multivector: bool

    def recreate_collection(self, dimensions: Mapping[str, int]) -> None:
        ...

    def upsert_points(self, points: list[dict[str, Any]]) -> None:
        ...

    def query_task1(
        self,
        query_vectors: Mapping[str, Any],
        limits: Mapping[str, int | None],
        query_mode: str,
    ) -> list[dict[str, Any]]:
        ...


def lowcost_retrieval_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("lowcost_retrieval", {}))


def lowcost_collection_name(config: Mapping[str, Any]) -> str:
    retrieval_config = lowcost_retrieval_config(config)
    qdrant_config = config.get("qdrant", {})
    return str(
        retrieval_config.get("collection_name")
        or qdrant_config.get("lowcost_collection_name")
        or DEFAULT_COLLECTION_NAME
    )


def lowcost_task1_config(config: Mapping[str, Any]) -> dict[str, Any]:
    retrieval_config = lowcost_retrieval_config(config)
    task1_config = dict(retrieval_config.get("task1", {}))
    task1_config.update(dict(config.get("lowcost_task1", {})))
    return task1_config


def lowcost_task1_query_mode(config: Mapping[str, Any]) -> str:
    query_mode = str(
        lowcost_task1_config(config).get("query_mode", TASK1_DEFAULT_QUERY_MODE)
    )
    if query_mode not in TASK1_QUERY_MODES:
        raise ValueError(
            f"Unsupported low-cost Task 1 query_mode={query_mode!r}; "
            f"expected one of {sorted(TASK1_QUERY_MODES)}"
        )
    return query_mode


def positive_int(value: Any, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return normalized


def lowcost_task1_limits(config: Mapping[str, Any]) -> dict[str, int | None]:
    task1_config = lowcost_task1_config(config)
    limits: dict[str, int | None] = {}
    for key, default in TASK1_DEFAULT_LIMITS.items():
        limits[key] = positive_int(task1_config.get(key, default), key)

    max_articles = task1_config.get("max_articles")
    limits["max_articles"] = (
        None if max_articles in (None, "") else positive_int(max_articles, "max_articles")
    )
    return limits


def point_id_from_sample_id(sample_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"traffic-legal-vlm:lowcost:{sample_id}"))


def vector_dimensions_from_row(row: Mapping[str, Any]) -> dict[str, int]:
    object_vectors = row.get(OBJECT_VECTOR_NAME)
    first_object_vector = object_vectors[0] if isinstance(object_vectors, list) and object_vectors else []
    return {
        TEXT_VECTOR_NAME: len(row.get(TEXT_VECTOR_NAME) or []),
        IMAGE_VECTOR_NAME: len(row.get(IMAGE_VECTOR_NAME) or []),
        OBJECT_DIMENSION_NAME: len(first_object_vector or []),
    }


def normalize_dimensions(dimensions: Mapping[str, Any]) -> dict[str, int]:
    aliases = {
        TEXT_VECTOR_NAME: TEXT_VECTOR_NAME,
        IMAGE_VECTOR_NAME: IMAGE_VECTOR_NAME,
        OBJECT_VECTOR_NAME: OBJECT_DIMENSION_NAME,
        OBJECT_DIMENSION_NAME: OBJECT_DIMENSION_NAME,
    }
    normalized: dict[str, int] = {}
    for raw_key, raw_value in dimensions.items():
        key = aliases.get(str(raw_key), str(raw_key))
        if key not in {TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, OBJECT_DIMENSION_NAME}:
            continue
        value = int(raw_value)
        if value <= 0:
            raise ValueError(f"Feature dimension for {key} must be positive")
        normalized[key] = value
    return normalized


def dimensions_from_config(config: Mapping[str, Any]) -> dict[str, int]:
    return normalize_dimensions(lowcost_retrieval_config(config).get("dimensions", {}))


def dimensions_from_manifest(manifest: Mapping[str, Any]) -> dict[str, int]:
    return normalize_dimensions(manifest.get("dimensions", {}))


def resolve_feature_dimensions(
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
) -> dict[str, int]:
    config_dimensions = dimensions_from_config(config)
    manifest_dimensions = dimensions_from_manifest(manifest)
    row_dimensions = vector_dimensions_from_row(rows[0]) if rows else {}

    dimensions = manifest_dimensions or row_dimensions
    if config_dimensions:
        dimensions = config_dimensions
        for key, expected in config_dimensions.items():
            actual = manifest_dimensions.get(key)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"Feature dimension mismatch for {key}: "
                    f"config expects {expected}, manifest has {actual}"
                )

    missing = [
        key
        for key in (TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, OBJECT_DIMENSION_NAME)
        if not dimensions.get(key)
    ]
    if missing:
        raise ValueError(f"Feature manifest is missing dimensions: {missing}")
    return dimensions


def vector_param(size: int, multivector: bool = False) -> models.VectorParams:
    kwargs: dict[str, Any] = {
        "size": size,
        "distance": models.Distance.COSINE,
    }
    if hasattr(models, "Datatype"):
        kwargs["datatype"] = models.Datatype.FLOAT32
    if multivector:
        kwargs["multivector_config"] = models.MultiVectorConfig(
            comparator=models.MultiVectorComparator.MAX_SIM
        )
        kwargs["hnsw_config"] = models.HnswConfigDiff(m=0)
    return models.VectorParams(**kwargs)


def build_named_vector_config(dimensions: Mapping[str, int]) -> dict[str, models.VectorParams]:
    return {
        TEXT_VECTOR_NAME: vector_param(int(dimensions[TEXT_VECTOR_NAME])),
        IMAGE_VECTOR_NAME: vector_param(int(dimensions[IMAGE_VECTOR_NAME])),
        OBJECT_VECTOR_NAME: vector_param(int(dimensions[OBJECT_DIMENSION_NAME]), multivector=True),
    }


def validate_vector(values: Any, expected_dimension: int, vector_name: str, sample_id: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{sample_id}: {vector_name} must be a non-empty vector")
    if len(values) != expected_dimension:
        raise ValueError(
            f"{sample_id}: feature dimension mismatch for {vector_name}; "
            f"expected {expected_dimension}, got {len(values)}"
        )
    return [float(value) for value in values]


def validate_object_vectors(values: Any, expected_dimension: int, sample_id: str) -> list[list[float]]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{sample_id}: {OBJECT_VECTOR_NAME} must be a non-empty multivector")
    normalized: list[list[float]] = []
    for index, vector in enumerate(values):
        if not isinstance(vector, list) or len(vector) != expected_dimension:
            got = len(vector) if isinstance(vector, list) else "not-a-vector"
            raise ValueError(
                f"{sample_id}: feature dimension mismatch for "
                f"{OBJECT_VECTOR_NAME}[{index}]; expected {expected_dimension}, got {got}"
            )
        normalized.append([float(value) for value in vector])
    return normalized


def sample_id_from_feature_row(row: Mapping[str, Any]) -> str:
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    if not sample_id:
        raise ValueError("Feature row is missing sample_id/id")
    return sample_id


def query_vectors_from_feature_row(
    row: Mapping[str, Any],
    dimensions: Mapping[str, int],
    query_mode: str,
) -> dict[str, Any]:
    sample_id = sample_id_from_feature_row(row)
    vectors: dict[str, Any] = {
        TEXT_VECTOR_NAME: validate_vector(
            row.get(TEXT_VECTOR_NAME),
            int(dimensions[TEXT_VECTOR_NAME]),
            TEXT_VECTOR_NAME,
            sample_id,
        ),
        IMAGE_VECTOR_NAME: validate_vector(
            row.get(IMAGE_VECTOR_NAME),
            int(dimensions[IMAGE_VECTOR_NAME]),
            IMAGE_VECTOR_NAME,
            sample_id,
        ),
    }
    if query_mode == "text_image_object":
        vectors[OBJECT_VECTOR_NAME] = validate_object_vectors(
            row.get(OBJECT_VECTOR_NAME),
            int(dimensions[OBJECT_DIMENSION_NAME]),
            sample_id,
        )
    return vectors


def citations_for_train_row(row: Mapping[str, Any]) -> list[dict[str, str]]:
    citations = row.get("relevant_articles")
    if not isinstance(citations, list) or not citations:
        sample_id = row.get("sample_id") or row.get("id") or "<unknown>"
        raise ValueError(f"{sample_id}: train feature row must include relevant_articles")
    normalized: list[dict[str, str]] = []
    for citation in citations:
        if not isinstance(citation, Mapping):
            raise ValueError(f"{sample_id}: relevant_articles must contain objects")
        law_id = str(citation.get("law_id") or "").strip()
        article_id = str(citation.get("article_id") or "").strip()
        if not law_id or not article_id:
            raise ValueError(f"{sample_id}: relevant_articles require law_id and article_id")
        normalized.append({"law_id": law_id, "article_id": article_id})
    return normalized


def payload_from_feature_row(row: Mapping[str, Any]) -> dict[str, Any]:
    sample_id = sample_id_from_feature_row(row)
    payload = {
        "sample_id": sample_id,
        "image_id": row.get("image_id"),
        "image_path": row.get("image_path"),
        "question": row.get("question"),
        "question_type": row.get("question_type"),
        "choices": row.get("choices", {}),
        "answer": row.get("answer"),
        "relevant_articles": citations_for_train_row(row),
        "split": row.get("split", "train"),
    }
    missing = [key for key in ("image_id", "image_path", "question") if not payload.get(key)]
    if missing:
        raise ValueError(f"{sample_id}: payload missing required field(s): {missing}")
    return payload


def point_from_feature_row(
    row: Mapping[str, Any],
    dimensions: Mapping[str, int],
) -> dict[str, Any]:
    payload = payload_from_feature_row(row)
    sample_id = payload["sample_id"]
    return {
        "id": point_id_from_sample_id(sample_id),
        "vector": {
            TEXT_VECTOR_NAME: validate_vector(
                row.get(TEXT_VECTOR_NAME),
                int(dimensions[TEXT_VECTOR_NAME]),
                TEXT_VECTOR_NAME,
                sample_id,
            ),
            IMAGE_VECTOR_NAME: validate_vector(
                row.get(IMAGE_VECTOR_NAME),
                int(dimensions[IMAGE_VECTOR_NAME]),
                IMAGE_VECTOR_NAME,
                sample_id,
            ),
            OBJECT_VECTOR_NAME: validate_object_vectors(
                row.get(OBJECT_VECTOR_NAME),
                int(dimensions[OBJECT_DIMENSION_NAME]),
                sample_id,
            ),
        },
        "payload": payload,
    }


class QdrantLowCostVectorStore:
    def __init__(
        self,
        url: str,
        collection_name: str,
        batch_size: int = 64,
        api_key: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self.batch_size = int(batch_size)
        self.uses_multivector = True
        self.client = QdrantClient(url=url, api_key=api_key)

    def recreate_collection(self, dimensions: Mapping[str, int]) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=build_named_vector_config(dimensions),
            on_disk_payload=True,
        )

    def upsert_points(self, points: list[dict[str, Any]]) -> None:
        for start in range(0, len(points), self.batch_size):
            batch = points[start : start + self.batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point["id"],
                        vector=point["vector"],
                        payload=point["payload"],
                    )
                    for point in batch
                ],
            )

    def query_task1(
        self,
        query_vectors: Mapping[str, Any],
        limits: Mapping[str, int | None],
        query_mode: str,
    ) -> list[dict[str, Any]]:
        if query_mode == "text_image_object":
            prefetch = [
                models.Prefetch(
                    query=query_vectors[TEXT_VECTOR_NAME],
                    using=TEXT_VECTOR_NAME,
                    limit=int(limits["text_limit"] or TASK1_DEFAULT_LIMITS["text_limit"]),
                ),
                models.Prefetch(
                    query=query_vectors[IMAGE_VECTOR_NAME],
                    using=IMAGE_VECTOR_NAME,
                    limit=int(limits["image_limit"] or TASK1_DEFAULT_LIMITS["image_limit"]),
                ),
            ]
            query = query_vectors[OBJECT_VECTOR_NAME]
            using = OBJECT_VECTOR_NAME
            limit = int(limits["object_limit"] or TASK1_DEFAULT_LIMITS["object_limit"])
        elif query_mode == "text_image":
            prefetch = [
                models.Prefetch(
                    query=query_vectors[TEXT_VECTOR_NAME],
                    using=TEXT_VECTOR_NAME,
                    limit=int(limits["text_limit"] or TASK1_DEFAULT_LIMITS["text_limit"]),
                )
            ]
            query = query_vectors[IMAGE_VECTOR_NAME]
            using = IMAGE_VECTOR_NAME
            limit = int(limits["image_limit"] or TASK1_DEFAULT_LIMITS["image_limit"])
        else:
            raise ValueError(f"Unsupported low-cost Task 1 query_mode={query_mode!r}")

        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=query,
            using=using,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        return [
            {
                "score": float(getattr(point, "score", 0.0) or 0.0),
                "payload": dict(getattr(point, "payload", {}) or {}),
            }
            for point in points
        ]


def make_lowcost_vector_store(config: Mapping[str, Any]) -> QdrantLowCostVectorStore:
    qdrant_config = config.get("qdrant", {})
    retrieval_config = lowcost_retrieval_config(config)
    return QdrantLowCostVectorStore(
        url=str(qdrant_config.get("url", "http://localhost:6333")),
        collection_name=lowcost_collection_name(config),
        batch_size=int(retrieval_config.get("batch_size", qdrant_config.get("batch_size", 64))),
        api_key=qdrant_config.get("api_key"),
    )


def feature_manifest_path(features_path: str | Path) -> Path:
    path = Path(features_path)
    return path.with_suffix(".manifest.json")


def load_feature_rows(features_path: str | Path) -> list[dict[str, Any]]:
    path = Path(features_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Low-cost feature file not found: {path}. "
            "Run `python -m src.lowcost_features --set-name train` first."
        )
    rows = read_jsonl(str(path))
    if not rows:
        raise ValueError(f"Low-cost feature file is empty: {path}")
    return rows


def load_feature_manifest(features_path: str | Path) -> dict[str, Any]:
    path = feature_manifest_path(features_path)
    if not path.exists():
        # Backward-compatible fallback for callers that pass only an output dir.
        fallback = manifest_output_path(Path(features_path).parent, "train")
        path = fallback if fallback.exists() else path
    if not path.exists():
        raise FileNotFoundError(
            f"Low-cost feature manifest not found: {path}. "
            "Feature dimensions and source metadata are required before indexing."
        )
    return read_json(str(path))


def assert_feature_set_name(manifest: Mapping[str, Any], expected_set_name: str) -> None:
    actual_set_name = manifest.get("set_name")
    if actual_set_name and actual_set_name != expected_set_name:
        raise ValueError(
            f"Feature manifest set_name mismatch: expected {expected_set_name!r}, "
            f"found {actual_set_name!r}"
        )


def index_manifest_path(config: Mapping[str, Any], features_path: str | Path) -> Path:
    retrieval_config = lowcost_retrieval_config(config)
    if retrieval_config.get("index_manifest_path"):
        return Path(str(retrieval_config["index_manifest_path"]))
    return Path(features_path).parent / "index_manifest.json"


def write_index_manifest(
    config: Mapping[str, Any],
    features_path: str | Path,
    feature_manifest: Mapping[str, Any],
    dimensions: Mapping[str, int],
    indexed_count: int,
    collection_name: str,
    uses_multivector: bool,
) -> Path:
    path = index_manifest_path(config, features_path)
    manifest = {
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "collection": collection_name,
        "indexed_examples": indexed_count,
        "vectors": LOWCOST_VECTOR_NAMES,
        "uses_multivector": uses_multivector,
        "features_path": str(features_path),
        "feature_manifest_hash": stable_json_hash(feature_manifest),
        "dimensions": dict(dimensions),
    }
    write_json(manifest, str(path))
    return path


def index_lowcost_train_examples(
    config: Mapping[str, Any],
    features_path: str | Path,
    vector_store: LowCostVectorStore | None = None,
) -> dict[str, Any]:
    rows = load_feature_rows(features_path)
    manifest = load_feature_manifest(features_path)
    if manifest.get("set_name") != "train":
        raise ValueError(
            f"Low-cost train index requires train features; manifest has "
            f"set_name={manifest.get('set_name')!r}"
        )
    dimensions = resolve_feature_dimensions(config, manifest, rows)
    points = [point_from_feature_row(row, dimensions) for row in rows]
    vector_store = vector_store or make_lowcost_vector_store(config)
    vector_store.recreate_collection(dimensions)
    vector_store.upsert_points(points)
    manifest_path = write_index_manifest(
        config=config,
        features_path=features_path,
        feature_manifest=manifest,
        dimensions=dimensions,
        indexed_count=len(points),
        collection_name=vector_store.collection_name,
        uses_multivector=vector_store.uses_multivector,
    )
    return {
        "indexed_examples": len(points),
        "collection": vector_store.collection_name,
        "vectors": LOWCOST_VECTOR_NAMES,
        "uses_multivector": vector_store.uses_multivector,
        "index_manifest_path": str(manifest_path),
    }


def normalize_output_citation(citation: Any, sample_id: str) -> dict[str, str]:
    if not isinstance(citation, Mapping):
        raise ValueError(f"{sample_id}: relevant_articles must contain objects")
    law_id = str(citation.get("law_id") or "").strip()
    article_id = str(citation.get("article_id") or "").strip()
    if not law_id or not article_id:
        raise ValueError(f"{sample_id}: relevant_articles require law_id and article_id")
    return {"law_id": law_id, "article_id": article_id}


def retrieved_payload(result: Mapping[str, Any] | Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        payload = result.get("payload")
        if isinstance(payload, Mapping):
            return payload
        return result
    payload = getattr(result, "payload", None)
    return payload if isinstance(payload, Mapping) else {}


def union_relevant_articles_from_examples(
    retrieved_examples: Sequence[Mapping[str, Any] | Any],
    max_articles: int | None = None,
) -> list[dict[str, str]]:
    citations: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for result in retrieved_examples:
        payload = retrieved_payload(result)
        sample_id = str(payload.get("sample_id") or payload.get("id") or "<retrieved>")
        raw_citations = payload.get("relevant_articles")
        if not isinstance(raw_citations, list):
            raise ValueError(f"{sample_id}: retrieved example is missing relevant_articles")
        for citation in raw_citations:
            item = normalize_output_citation(citation, sample_id)
            key = (item["law_id"], item["article_id"])
            if key in seen:
                continue
            citations.append(item)
            seen.add(key)
            if max_articles is not None and len(citations) >= max_articles:
                return citations
    return citations


def task1_prediction_row(
    query_row: Mapping[str, Any],
    retrieved_examples: Sequence[Mapping[str, Any] | Any],
    max_articles: int | None = None,
) -> dict[str, Any]:
    sample_id = sample_id_from_feature_row(query_row)
    image_id = str(query_row.get("image_id") or "").strip()
    question = str(query_row.get("question") or "").strip()
    if not image_id:
        raise ValueError(f"{sample_id}: Task 1 feature row is missing image_id")
    if not question:
        raise ValueError(f"{sample_id}: Task 1 feature row is missing question")

    citations = union_relevant_articles_from_examples(retrieved_examples, max_articles)
    if not citations:
        raise ValueError(f"{sample_id}: retrieved examples produced no relevant_articles")
    return {
        "id": sample_id,
        "image_id": image_id,
        "question": question,
        "relevant_articles": citations,
    }


def task1_evidence_from_citations(
    citations: Sequence[Mapping[str, Any]],
    article_index: Mapping[str, dict],
    retrieved_examples: Sequence[Mapping[str, Any] | Any] | None = None,
    base_score: float = 1.0,
) -> tuple[list[Evidence], list[dict[str, Any]]]:
    """Resolve Task 1 citations into prompt-ready LawDB evidence."""
    evidence: list[Evidence] = []
    diagnostics: list[dict[str, Any]] = []
    sources = [
        {
            "sample_id": retrieved_payload(item).get("sample_id")
            or retrieved_payload(item).get("id"),
            "image_id": retrieved_payload(item).get("image_id"),
            "score": item.get("score") if isinstance(item, Mapping) else getattr(item, "score", None),
        }
        for item in (retrieved_examples or [])
    ]
    seen: set[str] = set()

    for citation in citations:
        sample_id = "lowcost_task1"
        item = normalize_output_citation(citation, sample_id)
        try:
            article = resolve_law_reference(item, article_index)
        except KeyError as exc:
            diagnostics.append(
                {
                    "type": "unknown_task1_citation",
                    "law_id": item["law_id"],
                    "article_id": item["article_id"],
                    "reason": str(exc),
                }
            )
            continue
        uid = str(article["uid"])
        if uid in seen:
            continue
        seen.add(uid)
        rank = len(evidence) + 1
        evidence.append(
            Evidence(
                law_id=str(article["law_id"]),
                article_id=str(article["article_id"]),
                title=str(article.get("title") or ""),
                content=str(article["content"]),
                score=max(0.0, float(base_score) - (rank - 1) * 0.01),
                rank=rank,
                retrieval_method=RetrievalMethod.EXAMPLE,
                metadata={
                    "schema_version": TASK1_EVIDENCE_SCHEMA_VERSION,
                    "source": "lowcost_task1_retrieved_examples",
                    "law_title": article.get("law_title"),
                    "images": article.get("images", []),
                    "tables": article.get("tables", []),
                    "retrieved_examples": sources,
                },
            )
        )
    return evidence, diagnostics


def retrieve_task1_citation_evidence(
    query_row: Mapping[str, Any],
    config: Mapping[str, Any],
    article_index: Mapping[str, dict],
    vector_store: LowCostVectorStore | None = None,
) -> tuple[list[Evidence], list[dict[str, Any]]]:
    """Run low-cost Task 1 retrieval and return LawDB evidence for a query row."""
    query_mode = lowcost_task1_query_mode(config)
    dimensions = dimensions_from_config(config) or vector_dimensions_from_row(query_row)
    missing = [
        key
        for key in (TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, OBJECT_DIMENSION_NAME)
        if not dimensions.get(key)
    ]
    if missing:
        raise ValueError(f"Task 1 evidence retrieval is missing dimensions: {missing}")

    limits = lowcost_task1_limits(config)
    vector_store = vector_store or make_lowcost_vector_store(config)
    query_vectors = query_vectors_from_feature_row(query_row, dimensions, query_mode)
    retrieved_examples = vector_store.query_task1(query_vectors, limits, query_mode)
    citations = union_relevant_articles_from_examples(
        retrieved_examples,
        max_articles=limits["max_articles"],
    )
    evidence, diagnostics = task1_evidence_from_citations(
        citations,
        article_index=article_index,
        retrieved_examples=retrieved_examples,
    )
    diagnostics.append(
        {
            "type": "lowcost_task1_retrieval",
            "collection": vector_store.collection_name,
            "query_mode": query_mode,
            "limits": limits,
            "retrieved_example_count": len(retrieved_examples),
            "citation_count": len(citations),
            "evidence_count": len(evidence),
        }
    )
    return evidence, diagnostics


def run_lowcost_task1_predictions(
    config: Mapping[str, Any],
    set_name: str,
    features_path: str | Path,
    output_path: str | Path,
    limit: int | None = None,
    vector_store: LowCostVectorStore | None = None,
) -> dict[str, Any]:
    if set_name not in {"public_test", "private_test", "val"}:
        raise ValueError("set_name must be public_test, private_test, or val")
    rows = load_feature_rows(features_path)
    manifest = load_feature_manifest(features_path)
    assert_feature_set_name(manifest, set_name)
    rows = rows[:limit] if limit is not None else rows
    if not rows:
        raise ValueError("No feature rows selected for low-cost Task 1 prediction")

    dimensions = resolve_feature_dimensions(config, manifest, rows)
    query_mode = lowcost_task1_query_mode(config)
    limits = lowcost_task1_limits(config)
    vector_store = vector_store or make_lowcost_vector_store(config)

    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        query_vectors = query_vectors_from_feature_row(row, dimensions, query_mode)
        retrieved_examples = vector_store.query_task1(query_vectors, limits, query_mode)
        prediction_rows.append(
            task1_prediction_row(row, retrieved_examples, limits["max_articles"])
        )

    write_jsonl(prediction_rows, str(output_path))
    return {
        "set_name": set_name,
        "output_path": str(output_path),
        "prediction_count": len(prediction_rows),
        "collection": vector_store.collection_name,
        "query_mode": query_mode,
        "limits": limits,
    }


def rows_by_sample_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {sample_id_from_feature_row(row): row for row in rows}


def evaluate_task1_predictions(
    gold_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predictions = rows_by_sample_id(prediction_rows)
    per_sample: list[dict[str, Any]] = []
    for gold in gold_rows:
        sample_id = sample_id_from_feature_row(gold)
        prediction = predictions.get(sample_id, {})
        scores = score_retrieval_sample(
            gold.get("relevant_articles") or [],
            prediction.get("relevant_articles") or [],
        )
        per_sample.append({"id": sample_id, **scores})

    return {
        "sample_count": len(gold_rows),
        "prediction_count": len(prediction_rows),
        "precision": mean(float(row["precision"]) for row in per_sample),
        "recall": mean(float(row["recall"]) for row in per_sample),
        "f2": mean(float(row["f2"]) for row in per_sample),
        "missing_prediction_ids": [
            sample_id_from_feature_row(gold)
            for gold in gold_rows
            if sample_id_from_feature_row(gold) not in predictions
        ],
        "per_sample": per_sample,
    }


def summarize_task1_ablation(
    gold_rows: Sequence[Mapping[str, Any]],
    predictions_by_setting: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    settings = []
    for name, rows in predictions_by_setting.items():
        metrics = evaluate_task1_predictions(gold_rows, rows)
        settings.append({"setting": name, **metrics})
    settings.sort(
        key=lambda row: (
            float(row["f2"]),
            float(row["recall"]),
            float(row["precision"]),
            str(row["setting"]),
        ),
        reverse=True,
    )
    return {
        "settings": settings,
        "best_setting": settings[0]["setting"] if settings else None,
    }


def run_task1_ablation(gold_path: str | Path, prediction_glob: str) -> dict[str, Any]:
    gold_rows = read_jsonl(str(gold_path))
    prediction_paths = sorted(glob.glob(prediction_glob))
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction files match: {prediction_glob}")
    predictions_by_setting = {
        Path(path).stem: read_jsonl(path)
        for path in prediction_paths
    }
    return summarize_task1_ablation(gold_rows, predictions_by_setting)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-cost train-example retrieval for VLSP Task 1."
    )
    parser.add_argument("--config", default="configs/experiments/lowcost_retrieval.yaml")
    parser.add_argument("--mode", choices=["index", "task1", "ablate"], default="index")
    parser.add_argument("--features")
    parser.add_argument("--set-name", default="public_test")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--gold")
    parser.add_argument("--prediction-glob")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.mode == "index":
        if not args.features:
            raise SystemExit("ERROR: --features is required for --mode index")
        try:
            summary = index_lowcost_train_examples(config, args.features)
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        print(f"indexed_examples={summary['indexed_examples']}")
        print(f"collection={summary['collection']}")
        print("vectors=" + ",".join(summary["vectors"]))
        if summary["uses_multivector"]:
            print("object_vector_mode=multivector_max_sim")
        else:
            print("object_vector_mode=fallback_single_vector")
    elif args.mode == "task1":
        if not args.features:
            raise SystemExit("ERROR: --features is required for --mode task1")
        if not args.output:
            raise SystemExit("ERROR: --output is required for --mode task1")
        try:
            summary = run_lowcost_task1_predictions(
                config=config,
                set_name=args.set_name,
                features_path=args.features,
                output_path=args.output,
                limit=args.limit,
            )
        except (FileNotFoundError, ValueError, RuntimeError) as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        print(f"predicted_examples={summary['prediction_count']}")
        print(f"collection={summary['collection']}")
        print(f"query_mode={summary['query_mode']}")
        print(
            "limits="
            f"text:{summary['limits']['text_limit']},"
            f"image:{summary['limits']['image_limit']},"
            f"object:{summary['limits']['object_limit']},"
            f"max_articles:{summary['limits']['max_articles']}"
        )
        print(f"output={summary['output_path']}")
    elif args.mode == "ablate":
        if not args.gold or not args.prediction_glob:
            raise SystemExit("ERROR: --gold and --prediction-glob are required for --mode ablate")
        try:
            summary = run_task1_ablation(args.gold, args.prediction_glob)
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"ERROR: {exc}") from None
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
