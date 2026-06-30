from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from src.lowcost_features import manifest_output_path
from src.utils import load_config, read_json, read_jsonl, stable_json_hash, utc_now_iso, write_json


TEXT_VECTOR_NAME = "text_vector"
IMAGE_VECTOR_NAME = "image_general_feature_vector"
OBJECT_VECTOR_NAME = "image_object_feature_list_vector"
OBJECT_DIMENSION_NAME = "image_object_feature_vector"
LOWCOST_VECTOR_NAMES = [TEXT_VECTOR_NAME, IMAGE_VECTOR_NAME, OBJECT_VECTOR_NAME]
DEFAULT_COLLECTION_NAME = "traffic_train_examples_lowcost"
INDEX_MANIFEST_SCHEMA_VERSION = "lowcost-qdrant-index-v1"


class LowCostVectorStore(Protocol):
    collection_name: str
    uses_multivector: bool

    def recreate_collection(self, dimensions: Mapping[str, int]) -> None:
        ...

    def upsert_points(self, points: list[dict[str, Any]]) -> None:
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
    sample_id = str(row.get("sample_id") or row.get("id") or "").strip()
    if not sample_id:
        raise ValueError("Feature row is missing sample_id/id")
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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index low-cost train-example features into Qdrant named vectors."
    )
    parser.add_argument("--config", default="configs/experiments/lowcost_retrieval.yaml")
    parser.add_argument("--mode", choices=["index"], default="index")
    parser.add_argument("--features", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.mode == "index":
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


if __name__ == "__main__":
    main()
