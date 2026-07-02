from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Mapping, Protocol

from PIL import Image

from src.data_utils import (
    attach_train_image_path,
    load_split_samples,
    load_vlsp_test_samples,
)
from src.utils import file_sha256, load_config, read_json, read_jsonl, stable_json_hash, utc_now_iso, write_json
from src.vision import load_rgb_image, resize_image


DEFAULT_TEXT_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_IMAGE_MODEL = "nvidia/C-RADIOv2-B"
DEFAULT_OBJECT_MODEL = "google/owlv2-large-patch14-ensemble"
DEFAULT_OBJECT_LABELS = [
    "traffic sign",
    "rectangle sign",
    "triangle sign",
    "square sign",
    "circle sign",
    "octagon sign",
    "stop sign",
]
DEFAULT_OUTPUT_DIR = "data/outputs/lowcost_features"
VALID_SET_NAMES = {"train", "public_test", "private_test"}
MANIFEST_SCHEMA_VERSION = "lowcost-features-manifest-v1"
FEATURE_SCHEMA_VERSION = "lowcost-features-v1"


class TextFeatureBackend(Protocol):
    model_name: str

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class ImageFeatureBackend(Protocol):
    model_name: str

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        ...


class ObjectDetectorBackend(Protocol):
    model_name: str
    labels: list[str]

    def detect(self, image: Image.Image, threshold: float) -> dict[str, list[Any]]:
        ...


def patch_transformers_tied_weights_compatibility() -> None:
    """Keep Jina remote-code loading compatible with newer Transformers builds.

    Some `jina-embeddings-v3` remote-code revisions assign
    `all_tied_weights_keys` during model initialization. Recent Transformers
    versions may expose that attribute without a setter, which makes the load
    fail before any weights are usable. The patch is intentionally tiny and
    local to the low-cost feature path.
    """
    try:
        from transformers.modeling_utils import PreTrainedModel
    except Exception:
        return

    current = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    if isinstance(current, property) and current.fset is not None:
        return

    class _TiedWeightKeys(list):
        def keys(self) -> Any:
            return dict.fromkeys(self).keys()

    def _get(self: Any) -> _TiedWeightKeys:
        value = self.__dict__.get("_all_tied_weights_keys", [])
        return _TiedWeightKeys(value or [])

    def _set(self: Any, value: Any) -> None:
        self.__dict__["_all_tied_weights_keys"] = list(value or [])

    PreTrainedModel.all_tied_weights_keys = property(_get, _set)  # type: ignore[attr-defined]


def normalize_owlv2_labels(raw_labels: Any, label_names: Sequence[str]) -> list[str]:
    """Normalize old numeric OWLv2 labels and newer grounded string labels."""
    if raw_labels is None:
        return []
    if hasattr(raw_labels, "detach"):
        raw_labels = raw_labels.detach().cpu().numpy().tolist()
    if not isinstance(raw_labels, Sequence) or isinstance(raw_labels, (str, bytes)):
        raw_labels = [raw_labels]

    normalized: list[str] = []
    for label in raw_labels:
        if isinstance(label, str):
            normalized.append(label)
            continue
        try:
            index = int(label)
        except (TypeError, ValueError):
            normalized.append(str(label))
            continue
        if 0 <= index < len(label_names):
            normalized.append(str(label_names[index]))
        else:
            normalized.append(str(index))
    return normalized


class JinaTextFeatureBackend:
    """Jina embedding wrapper matching the low-cost reference project."""

    def __init__(self, model_name: str = DEFAULT_TEXT_MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel

        self.model_name = model_name
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        patch_transformers_tied_weights_compatibility()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        with self.torch.no_grad():
            outputs = self.model.encode(texts)
        return [list(map(float, vector)) for vector in outputs]


class CRadioImageFeatureBackend:
    """C-RADIOv2-B summary-feature wrapper."""

    max_image_height = 1536
    max_image_width = 1536

    def __init__(self, model_name: str = DEFAULT_IMAGE_MODEL, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, CLIPImageProcessor

        self.model_name = model_name
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(self.model, "to"):
            self.model = self.model.to(self.device)
        self.model.eval()

    def _target_size(self, image: Image.Image) -> dict[str, int]:
        width, height = image.size
        if width > self.max_image_width or height > self.max_image_height:
            if width >= height:
                new_width = self.max_image_width
                new_height = max(1, new_width * height // width)
            else:
                new_height = self.max_image_height
                new_width = max(1, new_height * width // height)
        else:
            new_width = width
            new_height = height

        if hasattr(self.model, "get_nearest_supported_resolution"):
            resolution = self.model.get_nearest_supported_resolution(
                height=new_height,
                width=new_width,
            )
            return {"height": int(resolution.height), "width": int(resolution.width)}
        return {"height": int(new_height), "width": int(new_width)}

    def embed_images(self, images: list[Image.Image]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for image in images:
            model_inputs = self.processor(
                images=[image],
                return_tensors="pt",
                do_resize=True,
                size=self._target_size(image),
            ).to(self.device)
            with self.torch.no_grad():
                outputs = self.model(model_inputs.pixel_values)
            summary = getattr(outputs, "summary", None)
            if summary is None:
                raise RuntimeError("C-RADIO model output is missing summary features")
            vectors.append(summary[0].detach().cpu().float().numpy().tolist())
        return vectors


class OwlV2ObjectDetectorBackend:
    """OWLv2 traffic-sign detector wrapper used before object-crop encoding."""

    def __init__(
        self,
        model_name: str = DEFAULT_OBJECT_MODEL,
        labels: list[str] | None = None,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.model_name = model_name
        self.labels = list(labels or DEFAULT_OBJECT_LABELS)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

    def detect(self, image: Image.Image, threshold: float) -> dict[str, list[Any]]:
        image_size = [image.size[::-1]]
        model_inputs = self.processor(
            images=[image],
            text=[self.labels],
            return_tensors="pt",
        ).to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**model_inputs)
        target_sizes = self.torch.Tensor(image_size).to(self.device)
        if hasattr(self.processor, "post_process_grounded_object_detection"):
            results = self.processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
                text_labels=[self.labels],
            )[0]
        else:
            results = self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
            )[0]
        return {
            "boxes": results["boxes"].detach().cpu().float().numpy().tolist(),
            "scores": results["scores"].detach().cpu().float().numpy().tolist(),
            "labels": normalize_owlv2_labels(results.get("labels"), self.labels),
        }


def lowcost_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("lowcost_features", {}))


def model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    lowcost = lowcost_config(config)
    return {
        "text_model": lowcost.get("text_model", DEFAULT_TEXT_MODEL),
        "image_model": lowcost.get("image_model", DEFAULT_IMAGE_MODEL),
        "object_model": lowcost.get("object_model", DEFAULT_OBJECT_MODEL),
        "object_labels": list(lowcost.get("object_labels", DEFAULT_OBJECT_LABELS)),
        "object_threshold": float(lowcost.get("object_threshold", 0.3)),
        "image_max_size": int(lowcost.get("image_max_size", 1536)),
        "device": lowcost.get("device"),
    }


def format_lowcost_text(sample: Mapping[str, Any]) -> str:
    question = str(sample.get("question") or "").strip()
    if not question:
        raise ValueError("Low-cost text feature input requires a non-empty question")

    question_type = str(sample.get("question_type") or "").strip()
    choices = sample.get("choices") or {}
    parts = [f"Question: {question}"]

    if question_type == "Yes/No":
        parts.extend(["Options:", "Đúng", "Sai"])
    elif isinstance(choices, Mapping) and choices:
        parts.append("Options:")
        for key in sorted(choices):
            parts.append(f"{str(key).strip().upper()}: {str(choices[key]).strip()}")
    return "\n".join(parts)


def feature_output_path(output_dir: str | Path, set_name: str) -> Path:
    return Path(output_dir) / f"{set_name}_features.jsonl"


def manifest_output_path(output_dir: str | Path, set_name: str) -> Path:
    return Path(output_dir) / f"{set_name}_features.manifest.json"


def git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _paths_hash(paths: list[str | Path]) -> str:
    payload = []
    for path in paths:
        item = Path(path)
        payload.append(
            {
                "path": str(item),
                "sha256": file_sha256(item) if item.exists() else None,
            }
        )
    return stable_json_hash(payload)


def source_paths_for_set(config: Mapping[str, Any], set_name: str) -> list[str]:
    data_config = config.get("data", {})
    if set_name == "train":
        split_path = Path(str(data_config.get("train_split_path", "")))
        path = data_config.get("train_split_path") if split_path.exists() else data_config.get("train_path")
        return [str(path)] if path else []
    if set_name == "public_test":
        return [
            str(data_config.get("public_test_task1_path", "")),
            str(data_config.get("public_test_task2_path", "")),
        ]
    if set_name == "private_test":
        return [
            str(data_config.get("private_test_task1_path", "")),
            str(data_config.get("private_test_task2_path", "")),
        ]
    raise ValueError("set_name must be train, public_test, or private_test")


def load_train_feature_samples(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    data_config = dict(config.get("data", {}))
    split_path = Path(str(data_config.get("train_split_path", "")))
    if split_path.exists():
        samples = load_split_samples(dict(config), "train")
    else:
        train_path = data_config.get("train_path")
        if not train_path:
            raise KeyError("Missing data.train_path in config")
        samples = read_json(str(train_path))
        if not isinstance(samples, list):
            raise ValueError("Training JSON must contain a list of samples")
    return [attach_train_image_path(sample, dict(config)) for sample in samples]


def _merge_samples_by_id(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for sample in samples:
        sample_id = str(sample.get("id") or "")
        if not sample_id:
            raise ValueError("Test sample is missing id")
        if sample_id not in merged:
            merged[sample_id] = dict(sample)
            order.append(sample_id)
        else:
            merged[sample_id].update({key: value for key, value in sample.items() if value not in (None, {}, [])})
    return [merged[sample_id] for sample_id in order]


def load_test_feature_samples(config: Mapping[str, Any], set_name: str) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for task in ("task1", "task2"):
        try:
            samples.extend(load_vlsp_test_samples(config, set_name=set_name, task=task))
        except FileNotFoundError:
            continue
    if not samples:
        raise FileNotFoundError(f"No VLSP samples found for {set_name}")
    return _merge_samples_by_id(samples)


def load_feature_samples(config: Mapping[str, Any], set_name: str) -> list[dict[str, Any]]:
    if set_name == "train":
        return load_train_feature_samples(config)
    if set_name in {"public_test", "private_test"}:
        return load_test_feature_samples(config, set_name)
    raise ValueError("set_name must be train, public_test, or private_test")


def build_manifest(
    config: Mapping[str, Any],
    set_name: str,
    samples: list[Mapping[str, Any]],
    dimensions: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    models = model_config(config)
    source_paths = [path for path in source_paths_for_set(config, set_name) if path]
    source_hashes = {
        path: file_sha256(path)
        for path in source_paths
        if Path(path).exists()
    }
    image_paths = [str(sample["image_path"]) for sample in samples]
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "created_at": utc_now_iso(),
        "git_commit": git_commit_hash(),
        "set_name": set_name,
        "sample_count": len(samples),
        "source_files": source_paths,
        "source_file_hash": stable_json_hash(source_hashes),
        "image_file_list_hash": _paths_hash(image_paths),
        "models": {
            "text": models["text_model"],
            "image": models["image_model"],
            "object_detector": models["object_model"],
            "object_encoder": models["image_model"],
        },
        "object_labels": models["object_labels"],
        "object_threshold": models["object_threshold"],
        "image_max_size": models["image_max_size"],
        "dimensions": dict(dimensions or {}),
    }


def manifest_compatibility_key(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "set_name": manifest.get("set_name"),
        "source_file_hash": manifest.get("source_file_hash"),
        "image_file_list_hash": manifest.get("image_file_list_hash"),
        "models": manifest.get("models"),
        "object_labels": manifest.get("object_labels"),
        "object_threshold": manifest.get("object_threshold"),
        "image_max_size": manifest.get("image_max_size"),
    }


def assert_manifest_compatible(existing: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if manifest_compatibility_key(existing) != manifest_compatibility_key(expected):
        raise ValueError(
            "Low-cost feature manifest mismatch; refusing to resume with "
            "different models, inputs, threshold, or image settings"
        )


def existing_feature_ids(path: str | Path) -> set[str]:
    path = Path(path)
    if not path.exists():
        return set()
    ids: set[str] = set()
    for row in read_jsonl(str(path)):
        sample_id = str(row.get("id") or row.get("sample_id") or "")
        if sample_id:
            ids.add(sample_id)
    return ids


def crop_objects(image: Image.Image, boxes: list[Any]) -> list[Image.Image]:
    crops: list[Image.Image] = []
    width, height = image.size
    for box in boxes:
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        x_min, y_min, x_max, y_max = [float(value) for value in box]
        x_min = max(0, min(width, x_min))
        x_max = max(0, min(width, x_max))
        y_min = max(0, min(height, y_min))
        y_max = max(0, min(height, y_max))
        if x_max <= x_min or y_max <= y_min:
            continue
        crops.append(image.crop((x_min, y_min, x_max, y_max)).convert("RGB"))
    return crops


def zero_vector(dimension: int) -> list[float]:
    if dimension <= 0:
        raise ValueError("zero vector dimension must be positive")
    return [0.0] * dimension


class LowCostFeatureExtractor:
    def __init__(
        self,
        text_backend: TextFeatureBackend,
        image_backend: ImageFeatureBackend,
        object_detector: ObjectDetectorBackend,
        object_threshold: float = 0.3,
        image_max_size: int = 1536,
    ) -> None:
        self.text_backend = text_backend
        self.image_backend = image_backend
        self.object_detector = object_detector
        self.object_threshold = object_threshold
        self.image_max_size = image_max_size

    @property
    def model_names(self) -> dict[str, str]:
        return {
            "text": self.text_backend.model_name,
            "image": self.image_backend.model_name,
            "object_detector": self.object_detector.model_name,
            "object_encoder": self.image_backend.model_name,
        }

    def extract_one(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        text_input = format_lowcost_text(sample)
        text_vector = self.text_backend.embed_texts([text_input])[0]
        image = resize_image(load_rgb_image(str(sample["image_path"])), self.image_max_size)
        image_vector = self.image_backend.embed_images([image])[0]
        detections = self.object_detector.detect(image, threshold=self.object_threshold)
        boxes = list(detections.get("boxes", []))
        scores = list(detections.get("scores", []))
        labels = list(detections.get("labels", []))
        crops = crop_objects(image, boxes)
        if crops:
            object_vectors = self.image_backend.embed_images(crops)
        else:
            object_vectors = [zero_vector(len(image_vector))]

        row = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "id": sample.get("id"),
            "sample_id": sample.get("id"),
            "image_id": sample.get("image_id"),
            "image_path": sample.get("image_path"),
            "question_type": sample.get("question_type"),
            "question": sample.get("question"),
            "choices": sample.get("choices", {}),
            "text_input": text_input,
            "text_vector": text_vector,
            "image_general_feature_vector": image_vector,
            "image_object_feature_list_vector": object_vectors,
            "object_boxes": boxes,
            "object_scores": scores,
            "object_labels": labels,
        }
        if "answer" in sample:
            row["answer"] = sample.get("answer")
        if "relevant_articles" in sample:
            row["relevant_articles"] = sample.get("relevant_articles") or []
        return row


def make_lowcost_extractor(config: Mapping[str, Any]) -> LowCostFeatureExtractor:
    models = model_config(config)
    text_backend = JinaTextFeatureBackend(models["text_model"], device=models["device"])
    image_backend = CRadioImageFeatureBackend(models["image_model"], device=models["device"])
    object_detector = OwlV2ObjectDetectorBackend(
        models["object_model"],
        labels=models["object_labels"],
        device=models["device"],
    )
    return LowCostFeatureExtractor(
        text_backend=text_backend,
        image_backend=image_backend,
        object_detector=object_detector,
        object_threshold=models["object_threshold"],
        image_max_size=models["image_max_size"],
    )


def row_dimensions(row: Mapping[str, Any]) -> dict[str, int]:
    object_vectors = row.get("image_object_feature_list_vector") or []
    first_object = object_vectors[0] if object_vectors else []
    return {
        "text_vector": len(row.get("text_vector") or []),
        "image_general_feature_vector": len(row.get("image_general_feature_vector") or []),
        "image_object_feature_vector": len(first_object),
    }


def run_feature_cache(
    config: Mapping[str, Any],
    set_name: str,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    resume: bool = False,
    extractor: LowCostFeatureExtractor | None = None,
) -> dict[str, Any]:
    if set_name not in VALID_SET_NAMES:
        raise ValueError("set_name must be one of train, public_test, private_test")

    samples = load_feature_samples(config, set_name)
    if limit is not None:
        samples = samples[:limit]

    output_path = feature_output_path(output_dir, set_name)
    manifest_path = manifest_output_path(output_dir, set_name)
    expected_manifest = build_manifest(config, set_name, samples)
    done_ids: set[str] = set()
    if resume and output_path.exists():
        if not manifest_path.exists():
            raise FileNotFoundError(f"Cannot resume without manifest: {manifest_path}")
        existing_manifest = read_json(str(manifest_path))
        assert_manifest_compatible(existing_manifest, expected_manifest)
        done_ids = existing_feature_ids(output_path)
    elif output_path.exists() and not resume:
        output_path.unlink()

    extractor = extractor or make_lowcost_extractor(config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    rows_skipped = 0
    first_dimensions: dict[str, int] | None = None
    with output_path.open("a", encoding="utf-8") as handle:
        for sample in samples:
            sample_id = str(sample.get("id") or "")
            if resume and sample_id in done_ids:
                rows_skipped += 1
                continue
            row = extractor.extract_one(sample)
            if first_dimensions is None:
                first_dimensions = row_dimensions(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows_written += 1

    if first_dimensions is None and manifest_path.exists():
        existing_manifest = read_json(str(manifest_path))
        first_dimensions = dict(existing_manifest.get("dimensions", {}))
    manifest = build_manifest(config, set_name, samples, dimensions=first_dimensions or {})
    manifest.update(
        {
            "output_path": str(output_path),
            "rows_written": rows_written,
            "rows_skipped": rows_skipped,
            "resume": resume,
        }
    )
    write_json(manifest, str(manifest_path))
    return {
        "set_name": set_name,
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "sample_count": len(samples),
        "rows_written": rows_written,
        "rows_skipped": rows_skipped,
        "resume": resume,
    }


def set_names_from_cli(value: str) -> list[str]:
    if value == "all":
        return ["train", "public_test", "private_test"]
    if value not in VALID_SET_NAMES:
        raise ValueError("--set-name must be train, public_test, private_test, or all")
    return [value]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache low-cost text, whole-image, and object-crop features."
    )
    parser.add_argument("--config", default="configs/experiments/lowcost_retrieval.yaml")
    parser.add_argument(
        "--set-name",
        choices=["train", "public_test", "private_test", "all"],
        required=True,
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    summaries = [
        run_feature_cache(
            config,
            set_name=set_name,
            output_dir=args.output_dir,
            limit=args.limit,
            resume=args.resume,
        )
        for set_name in set_names_from_cli(args.set_name)
    ]
    print(json.dumps({"runs": summaries}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
