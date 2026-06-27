from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, UnidentifiedImageError

from src.utils import (
    EmbeddingCache,
    files_data_hash,
    l2_normalize_vectors,
    load_config,
)


class ImageBackend(Protocol):
    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        ...


def load_rgb_image(path: str | Path) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    try:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Could not read image file: {image_path}") from exc


def resize_image(image: Image.Image, max_size: int | None) -> Image.Image:
    if not max_size or max_size <= 0:
        return image

    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_size:
        return image

    scale = max_size / longest_side
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


class TransformersImageBackend:
    """Lazy whole-image embedding backend; tests should inject a fake backend."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.torch = torch

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                features = _extract_feature_tensor(
                    self.model.get_image_features(**inputs)
                )
            else:
                features = _extract_feature_tensor(self.model(**inputs))
        return features.detach().cpu().float().numpy().tolist()


def _extract_feature_tensor(output: Any) -> Any:
    """Return a tensor from common image model output shapes."""
    if hasattr(output, "detach"):
        return output

    if isinstance(output, dict):
        for key in ("image_embeds", "pooler_output"):
            value = output.get(key)
            if value is not None:
                return _extract_feature_tensor(value)
        value = output.get("last_hidden_state")
        if value is not None:
            return value[:, 0]

    for attr in ("image_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if value is not None:
            return _extract_feature_tensor(value)

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        return last_hidden_state[:, 0]

    if isinstance(output, (tuple, list)):
        if len(output) > 1 and output[1] is not None:
            return _extract_feature_tensor(output[1])
        if output:
            first = output[0]
            if hasattr(first, "ndim") and first.ndim >= 3:
                return first[:, 0]
            return _extract_feature_tensor(first)

    raise RuntimeError("Image model output does not expose image embeddings")


class ImageEmbeddingAdapter:
    """Model-agnostic whole-image embedding interface for retrieval experiments."""

    def __init__(
        self,
        backend: ImageBackend,
        model_name: str,
        max_size: int | None = 768,
        normalize: bool = True,
    ) -> None:
        self.backend = backend
        self.model_name = model_name
        self.max_size = max_size
        self.normalize = normalize

    def embed_images(self, paths: list[str | Path]) -> list[list[float]]:
        if not paths:
            return []
        images = [resize_image(load_rgb_image(path), self.max_size) for path in paths]
        vectors = self.backend.embed_images(images)
        return l2_normalize_vectors(vectors) if self.normalize else vectors


def image_embedding_config(config: dict) -> dict[str, Any]:
    return config.get("embeddings", {}).get("image", {})


def make_image_embedder(
    config: dict,
    backend: ImageBackend | None = None,
) -> ImageEmbeddingAdapter:
    embeddings_config = config.get("embeddings", {})
    image_config = image_embedding_config(config)
    model_name = image_config.get("model_name", "google/siglip-base-patch16-224")
    backend = backend or TransformersImageBackend(
        model_name=model_name,
        device=image_config.get("device"),
    )
    return ImageEmbeddingAdapter(
        backend=backend,
        model_name=model_name,
        max_size=image_config.get("max_size", 768),
        normalize=embeddings_config.get("normalize", True),
    )


def embed_images_cached(
    paths: list[str | Path],
    embedder: ImageEmbeddingAdapter,
    cache_dir: str | Path,
    cache_key: str,
    data_hash: str | None = None,
) -> list[list[float]]:
    data_hash = data_hash or files_data_hash(paths)
    cache = EmbeddingCache(cache_dir)
    metadata = {
        "model_name": embedder.model_name,
        "modality": "image",
        "data_hash": data_hash,
        "normalized": embedder.normalize,
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

    embeddings = embedder.embed_images(paths)
    cache.write(cache_key, embeddings, metadata=metadata)
    return embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Whole-image embedding utilities for retrieval experiments."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print image embedding settings without loading model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.show_config:
        print(
            json.dumps(
                {
                    "cache_dir": config.get("embeddings", {}).get(
                        "cache_dir",
                        "data/outputs/embeddings",
                    ),
                    "normalize": config.get("embeddings", {}).get("normalize", True),
                    "image": image_embedding_config(config),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
