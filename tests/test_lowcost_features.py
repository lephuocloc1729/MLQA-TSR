import json
from pathlib import Path

import pytest
from PIL import Image

from src.lowcost_features import (
    LowCostFeatureExtractor,
    assert_manifest_compatible,
    build_manifest,
    format_lowcost_text,
    load_feature_samples,
    normalize_owlv2_labels,
    patch_transformers_tied_weights_compatibility,
    run_feature_cache,
)
from src.utils import read_json, write_json, write_jsonl


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeTextBackend:
    model_name = "fake-jina"

    def embed_texts(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class FakeImageBackend:
    model_name = "fake-c-radio"

    def embed_images(self, images):
        return [[float(image.width), float(image.height), 1.0] for image in images]


class FakeEmptyDetector:
    model_name = "fake-owlv2"
    labels = ["traffic sign"]

    def detect(self, image, threshold):
        return {"boxes": [], "scores": [], "labels": []}


class FakeOneBoxDetector:
    model_name = "fake-owlv2"
    labels = ["traffic sign"]

    def detect(self, image, threshold):
        return {
            "boxes": [[0, 0, image.width, image.height]],
            "scores": [0.91],
            "labels": ["traffic sign"],
        }


def tiny_image(path: Path, size=(4, 3)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(0, 128, 255)).save(path)


def fake_extractor(detector=None) -> LowCostFeatureExtractor:
    return LowCostFeatureExtractor(
        text_backend=FakeTextBackend(),
        image_backend=FakeImageBackend(),
        object_detector=detector or FakeEmptyDetector(),
        object_threshold=0.3,
        image_max_size=1536,
    )


def sample(sample_id: str, image_path: Path, **overrides) -> dict:
    row = {
        "id": sample_id,
        "image_id": image_path.stem,
        "image_path": str(image_path),
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    row.update(overrides)
    return row


def base_config(tmp_path: Path, rows=None) -> dict:
    image_dir = tmp_path / "train_images"
    image_path = image_dir / "img_1.jpg"
    tiny_image(image_path)
    rows = rows or [sample("train_1", image_path)]
    train_split_path = tmp_path / "train_split.jsonl"
    write_jsonl(rows, train_split_path)
    return {
        "project": {"name": "traffic-legal-vlm", "seed": 42},
        "data": {
            "train_split_path": str(train_split_path),
            "train_path": str(tmp_path / "raw_train.json"),
            "train_image_dir": str(image_dir),
            "public_test_task1_path": str(tmp_path / "public_task1.json"),
            "public_test_task2_path": str(tmp_path / "public_task2.json"),
            "public_test_image_dir": str(tmp_path / "public_images"),
            "private_test_task1_path": str(tmp_path / "private_task1.json"),
            "private_test_task2_path": str(tmp_path / "private_task2.json"),
            "private_test_image_dir": str(tmp_path / "private_images"),
        },
        "lowcost_features": {
            "text_model": "fake-jina",
            "image_model": "fake-c-radio",
            "object_model": "fake-owlv2",
            "object_threshold": 0.3,
            "image_max_size": 1536,
            "object_labels": ["traffic sign"],
        },
    }


def read_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_lowcost_text_formatting_multiple_choice_and_yes_no():
    mc_text = format_lowcost_text(
        {
            "question": "Chọn biển báo đúng?",
            "question_type": "Multiple choice",
            "choices": {"B": "Hai", "A": "Một"},
        }
    )
    yes_no_text = format_lowcost_text(
        {"question": "Xe máy được đi thẳng, đúng hay sai?", "question_type": "Yes/No"}
    )

    assert mc_text == "Question: Chọn biển báo đúng?\nOptions:\nA: Một\nB: Hai"
    assert yes_no_text == "Question: Xe máy được đi thẳng, đúng hay sai?\nOptions:\nĐúng\nSai"


def test_fake_feature_extraction_row_shape_and_train_gold_preserved(tmp_path):
    cfg = base_config(tmp_path)
    output_dir = tmp_path / "features"

    summary = run_feature_cache(
        cfg,
        "train",
        output_dir=output_dir,
        limit=1,
        extractor=fake_extractor(detector=FakeOneBoxDetector()),
    )

    rows = read_rows(output_dir / "train_features.jsonl")
    manifest = read_json(output_dir / "train_features.manifest.json")
    assert summary["rows_written"] == 1
    assert rows[0]["id"] == "train_1"
    assert rows[0]["sample_id"] == "train_1"
    assert rows[0]["answer"] == "B"
    assert rows[0]["relevant_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert rows[0]["text_vector"] == [len(rows[0]["text_input"]), 1.0]
    assert rows[0]["image_general_feature_vector"] == [4.0, 3.0, 1.0]
    assert rows[0]["image_object_feature_list_vector"] == [[4.0, 3.0, 1.0]]
    assert rows[0]["object_labels"] == ["traffic sign"]
    assert manifest["models"]["text"] == "fake-jina"
    assert manifest["dimensions"]["image_object_feature_vector"] == 3


def test_empty_object_detection_uses_zero_vector(tmp_path):
    cfg = base_config(tmp_path)
    output_dir = tmp_path / "features"

    run_feature_cache(cfg, "train", output_dir=output_dir, limit=1, extractor=fake_extractor())

    row = read_rows(output_dir / "train_features.jsonl")[0]
    assert row["object_boxes"] == []
    assert row["object_scores"] == []
    assert row["object_labels"] == []
    assert row["image_object_feature_list_vector"] == [[0.0, 0.0, 0.0]]


def test_test_feature_loading_does_not_require_gold_labels(tmp_path):
    public_dir = tmp_path / "public_images"
    image_path = public_dir / "public_1.jpg"
    tiny_image(image_path)
    cfg = base_config(tmp_path)
    write_json(
        [{"id": "public_test_1", "image_id": "public_1", "question": "Đây là biển gì?"}],
        cfg["data"]["public_test_task1_path"],
    )
    write_json(
        [
            {
                "id": "public_test_51",
                "image_id": "public_1",
                "question": "Đúng hay sai?",
                "question_type": "Yes/No",
                "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
            }
        ],
        cfg["data"]["public_test_task2_path"],
    )

    samples = load_feature_samples(cfg, "public_test")

    assert [item["id"] for item in samples] == ["public_test_1", "public_test_51"]
    assert all("answer" not in item for item in samples)
    assert all("relevant_articles" not in item for item in samples)


def test_resume_skips_existing_rows(tmp_path):
    image_dir = tmp_path / "train_images"
    image_1 = image_dir / "img_1.jpg"
    image_2 = image_dir / "img_2.jpg"
    tiny_image(image_1)
    tiny_image(image_2)
    cfg = base_config(
        tmp_path,
        rows=[sample("train_1", image_1), sample("train_2", image_2)],
    )
    output_dir = tmp_path / "features"

    first = run_feature_cache(cfg, "train", output_dir=output_dir, extractor=fake_extractor())
    second = run_feature_cache(cfg, "train", output_dir=output_dir, resume=True, extractor=fake_extractor())

    rows = read_rows(output_dir / "train_features.jsonl")
    assert first["rows_written"] == 2
    assert second["rows_skipped"] == 2
    assert second["rows_written"] == 0
    assert [row["id"] for row in rows] == ["train_1", "train_2"]


def test_manifest_mismatch_raises_clear_error(tmp_path):
    cfg = base_config(tmp_path)
    output_dir = tmp_path / "features"
    run_feature_cache(cfg, "train", output_dir=output_dir, limit=1, extractor=fake_extractor())

    changed_cfg = base_config(tmp_path)
    changed_cfg["lowcost_features"]["object_threshold"] = 0.9

    with pytest.raises(ValueError, match="manifest mismatch"):
        run_feature_cache(
            changed_cfg,
            "train",
            output_dir=output_dir,
            limit=1,
            resume=True,
            extractor=fake_extractor(),
        )


def test_manifest_compatibility_helper_detects_model_changes(tmp_path):
    cfg = base_config(tmp_path)
    samples = load_feature_samples(cfg, "train")
    manifest = build_manifest(cfg, "train", samples)
    changed = dict(manifest)
    changed["models"] = {**manifest["models"], "text": "other"}

    with pytest.raises(ValueError, match="manifest mismatch"):
        assert_manifest_compatible(changed, manifest)


def test_owlv2_label_normalization_supports_numeric_and_string_labels():
    labels = normalize_owlv2_labels([0, "traffic sign", 99], ["traffic sign", "circle sign"])

    assert labels == ["traffic sign", "traffic sign", "99"]


def test_transformers_tied_weights_compatibility_patch_allows_assignment():
    pytest.importorskip("transformers")
    from transformers.modeling_utils import PreTrainedModel

    original = getattr(PreTrainedModel, "all_tied_weights_keys", None)
    try:
        patch_transformers_tied_weights_compatibility()
        instance = object.__new__(PreTrainedModel)
        instance.all_tied_weights_keys = ["a", "b"]

        assert instance.all_tied_weights_keys == ["a", "b"]
        assert set(instance.all_tied_weights_keys.keys()) == {"a", "b"}
        assert instance.all_tied_weights_keys.items() == [("a", "a"), ("b", "b")]
    finally:
        if original is None:
            try:
                delattr(PreTrainedModel, "all_tied_weights_keys")
            except AttributeError:
                pass
        else:
            PreTrainedModel.all_tied_weights_keys = original
