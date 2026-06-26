import hashlib
import json
import random
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def write_jsonl(items, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_json_hash(data: Any) -> str:
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def files_data_hash(paths: list[str | Path]) -> str:
    """Hash file identities and contents without storing raw bytes in metadata."""
    return stable_json_hash(
        [
            {
                "path": str(Path(path)),
                "sha256": file_sha256(path),
            }
            for path in paths
        ]
    )


def l2_normalize_vectors(vectors: list[list[float]] | np.ndarray) -> list[list[float]]:
    """Normalize a batch of vectors, keeping zero vectors as zero vectors."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.size == 0:
        return []
    if array.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.divide(
        array,
        norms,
        out=np.zeros_like(array),
        where=norms > 0,
    )
    return normalized.astype("float32").tolist()


def as_embedding_matrix(vectors: list[list[float]] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Embeddings must be a 2D matrix")
    if matrix.shape[0] == 0:
        raise ValueError("Embedding cache cannot store an empty matrix")
    if matrix.shape[1] == 0:
        raise ValueError("Embedding dimension must be positive")
    return matrix


def safe_cache_key(value: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    if not compact:
        return digest
    return f"{compact[:80]}-{digest}"


class EmbeddingCache:
    """Small local cache for embedding matrices plus compatibility metadata."""

    REQUIRED_EXPECTED_FIELDS = ("model_name", "modality", "data_hash", "dimension")

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)

    def _paths(self, cache_key: str) -> tuple[Path, Path]:
        safe_key = safe_cache_key(cache_key)
        return (
            self.cache_dir / f"{safe_key}.npy",
            self.cache_dir / f"{safe_key}.manifest.json",
        )

    def exists(self, cache_key: str) -> bool:
        vector_path, manifest_path = self._paths(cache_key)
        return vector_path.exists() and manifest_path.exists()

    def read_manifest(self, cache_key: str) -> dict:
        _, manifest_path = self._paths(cache_key)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Embedding cache manifest not found: {manifest_path}")
        return read_json(str(manifest_path))

    def write(
        self,
        cache_key: str,
        embeddings: list[list[float]] | np.ndarray,
        metadata: Mapping[str, Any],
    ) -> dict:
        matrix = as_embedding_matrix(embeddings)
        vector_path, manifest_path = self._paths(cache_key)
        vector_path.parent.mkdir(parents=True, exist_ok=True)

        manifest = dict(metadata)
        manifest.update(
            {
                "created_at": manifest.get("created_at", utc_now_iso()),
                "dimension": int(matrix.shape[1]),
                "count": int(matrix.shape[0]),
                "vectors_file": vector_path.name,
            }
        )
        for field in ("model_name", "modality", "data_hash", "dimension", "created_at"):
            if field not in manifest or manifest[field] in (None, ""):
                raise ValueError(f"Embedding cache metadata missing required field: {field}")

        np.save(vector_path, matrix.astype("float32"))
        write_json(manifest, str(manifest_path))
        return manifest

    def read(
        self,
        cache_key: str,
        expected_metadata: Mapping[str, Any],
    ) -> list[list[float]]:
        for field in self.REQUIRED_EXPECTED_FIELDS:
            if field not in expected_metadata:
                raise ValueError(f"Expected cache metadata must include {field!r}")

        vector_path, manifest_path = self._paths(cache_key)
        if not vector_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(
                f"Embedding cache files not found for {cache_key!r} in {self.cache_dir}"
            )

        manifest = read_json(str(manifest_path))
        for field in self.REQUIRED_EXPECTED_FIELDS:
            expected = expected_metadata[field]
            actual = manifest.get(field)
            if actual != expected:
                raise ValueError(
                    "Embedding cache metadata mismatch for "
                    f"{field}: expected {expected!r}, found {actual!r}"
                )

        vectors = np.load(vector_path)
        if vectors.ndim != 2:
            raise ValueError(f"Cached embeddings must be 2D, got shape {vectors.shape}")
        if vectors.shape[1] != manifest["dimension"]:
            raise ValueError(
                "Embedding cache vector dimension mismatch: "
                f"manifest={manifest['dimension']}, vectors={vectors.shape[1]}"
            )
        if vectors.shape[0] != manifest["count"]:
            raise ValueError(
                "Embedding cache vector count mismatch: "
                f"manifest={manifest['count']}, vectors={vectors.shape[0]}"
            )
        return vectors.astype("float32").tolist()
