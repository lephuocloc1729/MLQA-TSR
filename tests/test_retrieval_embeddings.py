from pathlib import Path

import numpy as np
import pytest

from src.retrieval import (
    SentenceTransformerEmbedder,
    embed_texts,
    embed_texts_cached,
    text_data_hash,
)
from src.utils import EmbeddingCache, l2_normalize_vectors


class FakeSentenceModel:
    def __init__(self) -> None:
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.asarray(
            [[float(len(text)), 1.0, 0.0] for text in texts],
            dtype=np.float32,
        )


class FakeTextEmbedder:
    model_name = "fake-text"

    def __init__(self) -> None:
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += 1
        return [[float(len(text)), 1.0] for text in texts]


def test_sentence_transformer_adapter_can_wrap_fake_model():
    model = FakeSentenceModel()
    embedder = object.__new__(SentenceTransformerEmbedder)
    embedder.model_name = "fake-sentence-transformer"
    embedder.model = model

    vectors = embedder.embed_texts(["abc", "de"])

    assert vectors == [[3.0, 1.0, 0.0], [2.0, 1.0, 0.0]]
    assert model.calls[0][1]["normalize_embeddings"] is True


def test_text_embedding_adapter_batches_and_normalizes():
    embedder = FakeTextEmbedder()

    vectors = embed_texts(["abc"], embedder=embedder, normalize=True)

    assert embedder.calls == 1
    assert vectors[0] == pytest.approx([3 / np.sqrt(10), 1 / np.sqrt(10)])


def test_vector_normalization_keeps_zero_vectors():
    vectors = l2_normalize_vectors([[3.0, 4.0], [0.0, 0.0]])

    assert vectors[0] == pytest.approx([0.6, 0.8])
    assert vectors[1] == [0.0, 0.0]


def test_text_embedding_cache_reuses_compatible_vectors(tmp_path: Path):
    embedder = FakeTextEmbedder()
    texts = ["đường cấm", "biển báo"]

    first = embed_texts_cached(
        texts,
        embedder=embedder,
        cache_dir=tmp_path,
        cache_key="query-text",
        model_name=embedder.model_name,
        data_hash=text_data_hash(texts),
    )
    second = embed_texts_cached(
        texts,
        embedder=embedder,
        cache_dir=tmp_path,
        cache_key="query-text",
        model_name=embedder.model_name,
        data_hash=text_data_hash(texts),
    )

    assert first == second
    assert embedder.calls == 1


def test_cache_metadata_mismatch_fails_clearly(tmp_path: Path):
    cache = EmbeddingCache(tmp_path)
    cache.write(
        "articles",
        [[1.0, 0.0]],
        metadata={
            "model_name": "model-a",
            "modality": "text",
            "data_hash": "hash-a",
        },
    )

    with pytest.raises(ValueError, match="model_name"):
        cache.read(
            "articles",
            {
                "model_name": "model-b",
                "modality": "text",
                "data_hash": "hash-a",
                "dimension": 2,
            },
        )

    with pytest.raises(ValueError, match="dimension"):
        cache.read(
            "articles",
            {
                "model_name": "model-a",
                "modality": "text",
                "data_hash": "hash-a",
                "dimension": 3,
            },
        )

    with pytest.raises(ValueError, match="data_hash"):
        cache.read(
            "articles",
            {
                "model_name": "model-a",
                "modality": "text",
                "data_hash": "hash-b",
                "dimension": 2,
            },
        )
