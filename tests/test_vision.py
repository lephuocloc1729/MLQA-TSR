from pathlib import Path

import pytest
from PIL import Image

from src.vision import ImageEmbeddingAdapter, embed_images_cached, load_rgb_image


class FakeImageBackend:
    def __init__(self) -> None:
        self.image_modes: list[str] = []
        self.image_sizes: list[tuple[int, int]] = []
        self.calls = 0

    def embed_images(self, images):
        self.calls += 1
        self.image_modes.extend(image.mode for image in images)
        self.image_sizes.extend(image.size for image in images)
        return [[float(image.size[0]), float(image.size[1]), 1.0] for image in images]


def save_tiny_image(path: Path, mode: str = "L", size: tuple[int, int] = (6, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new(mode, size, color=128).save(path)


def test_load_rgb_image_converts_non_rgb_image(tmp_path: Path):
    image_path = tmp_path / "gray.png"
    save_tiny_image(image_path, mode="L")

    image = load_rgb_image(image_path)

    assert image.mode == "RGB"
    assert image.size == (6, 3)


def test_image_embedding_adapter_batches_and_normalizes(tmp_path: Path):
    image_path = tmp_path / "tiny.png"
    second_image_path = tmp_path / "tiny-2.png"
    save_tiny_image(image_path, size=(6, 3))
    save_tiny_image(second_image_path, size=(2, 2))
    backend = FakeImageBackend()
    adapter = ImageEmbeddingAdapter(
        backend=backend,
        model_name="fake-image",
        max_size=4,
        normalize=True,
    )

    vectors = adapter.embed_images([image_path, second_image_path])

    assert backend.image_modes == ["RGB", "RGB"]
    assert backend.image_sizes == [(4, 2), (2, 2)]
    assert len(vectors) == 2
    assert pytest.approx(sum(value * value for value in vectors[0])) == 1.0


def test_image_embedding_cache_uses_metadata(tmp_path: Path):
    image_path = tmp_path / "tiny.png"
    save_tiny_image(image_path)
    backend = FakeImageBackend()
    adapter = ImageEmbeddingAdapter(backend, model_name="fake-image")
    cache_dir = tmp_path / "cache"

    first = embed_images_cached(
        [image_path],
        embedder=adapter,
        cache_dir=cache_dir,
        cache_key="tiny-image",
        data_hash="image-hash",
    )
    second = embed_images_cached(
        [image_path],
        embedder=adapter,
        cache_dir=cache_dir,
        cache_key="tiny-image",
        data_hash="image-hash",
    )

    assert first == second
    assert backend.calls == 1


def test_missing_image_path_error_is_helpful(tmp_path: Path):
    missing_path = tmp_path / "missing.jpg"

    with pytest.raises(FileNotFoundError, match="Image file not found"):
        load_rgb_image(missing_path)
