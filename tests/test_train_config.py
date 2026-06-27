import json
from pathlib import Path

import pytest
import yaml

from src.train_qlora import (
    build_metadata,
    load_and_validate_plan,
    main,
    validate_resume_checkpoint,
    validate_training_config,
    write_checkpoint_metadata,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def sft_row(sample_id: str) -> dict:
    return {
        "id": sample_id,
        "image_id": f"img_{sample_id}",
        "image_path": f"data/raw/train_data/train_images/img_{sample_id}.jpg",
        "messages": [
            {"role": "user", "content": "Question"},
            {
                "role": "assistant",
                "content": (
                    '{"answer":"A","citations":[{"law_id":"LAW","article_id":"1"}],'
                    '"explanation":"ok","confidence":1.0,"abstained":false}'
                ),
            },
        ],
    }


def qlora_config(tmp_path: Path, **training_overrides) -> tuple[Path, dict]:
    train_path = tmp_path / "sft_train.jsonl"
    val_path = tmp_path / "sft_val.jsonl"
    manifest_path = tmp_path / "split_manifest.json"
    write_jsonl(train_path, [sft_row("train_1"), sft_row("train_2")])
    write_jsonl(val_path, [sft_row("val_1")])
    manifest_path.write_text('{"split_hash":"split-test-hash"}', encoding="utf-8")

    training = {
        "method": "qlora",
        "load_in_4bit": True,
        "lora_rank": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "v_proj"],
        "freeze_vision_encoder": True,
        "learning_rate": 0.0001,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "max_samples": 20,
        "seed": 42,
    }
    training.update(training_overrides)
    config = {
        "model": {"name": "Qwen/Qwen2.5-VL-3B-Instruct"},
        "training": training,
        "data": {
            "train_path": str(train_path),
            "val_path": str(val_path),
            "split_manifest_path": str(manifest_path),
        },
        "output": {"checkpoint_dir": str(tmp_path / "adapter")},
    }
    config_path = tmp_path / "qlora.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path, config


def test_qlora_config_loading_builds_plan(tmp_path):
    config_path, _ = qlora_config(tmp_path)

    _, plan = load_and_validate_plan(str(config_path), max_samples_override=1)

    assert plan.base_model == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert plan.lora_rank == 8
    assert plan.lora_alpha == 16
    assert plan.max_samples == 1
    assert plan.train_count == 2
    assert plan.val_count == 1
    assert plan.effective_train_count == 1
    assert plan.target_modules == ["q_proj", "v_proj"]
    assert plan.estimated_vram_gb is not None


def test_dry_run_output_with_fake_filesystem_paths(tmp_path, capsys):
    config_path, _ = qlora_config(tmp_path)

    main(["--config", str(config_path), "--dry-run", "--max-samples", "1"])

    output = json.loads(capsys.readouterr().out)
    assert output["mode"] == "dry-run"
    assert output["plan"]["effective_train_count"] == 1
    assert output["plan"]["checkpoint_dir"] == str(tmp_path / "adapter")
    assert "runtime_warnings" in output["plan"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lora_rank", 0, "training.lora_rank"),
        ("lora_alpha", 0, "training.lora_alpha"),
        ("lora_dropout", 1.0, "training.lora_dropout"),
    ],
)
def test_invalid_lora_settings_fail_clearly(tmp_path, field, value, message):
    config_path, config = qlora_config(tmp_path)
    config["training"][field] = value

    with pytest.raises(ValueError, match=message):
        validate_training_config(config, str(config_path))


def test_missing_sft_file_fails_clearly(tmp_path):
    config_path, config = qlora_config(tmp_path)
    Path(config["data"]["train_path"]).unlink()

    with pytest.raises(FileNotFoundError, match="build-sft"):
        validate_training_config(config, str(config_path))


def test_checkpoint_metadata_writer(tmp_path):
    config_path, config = qlora_config(tmp_path)
    _, plan = load_and_validate_plan(str(config_path), max_samples_override=1)
    metadata = build_metadata(
        config,
        plan,
        parameter_summary={
            "total": 100,
            "trainable": 4,
            "trainable_percent": 4.0,
        },
        status="smoke-complete",
    )

    path = write_checkpoint_metadata(tmp_path / "adapter", metadata)
    saved = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == "adapter_metadata.json"
    assert saved["status"] == "smoke-complete"
    assert saved["base_model"] == plan.base_model
    assert saved["dataset"]["dataset_hash"]
    assert saved["dataset"]["split_hash"] == "split-test-hash"
    assert saved["lora"]["rank"] == 8
    assert saved["parameters"]["trainable"] == 4


def test_missing_resume_checkpoint_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError, match="Resume checkpoint does not exist"):
        validate_resume_checkpoint(str(tmp_path / "missing-checkpoint"))
