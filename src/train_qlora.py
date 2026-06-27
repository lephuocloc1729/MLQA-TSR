from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import torch

from src.collator import SFTDataCollator
from src.utils import file_sha256, load_config, read_json, read_jsonl, write_json


DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
METADATA_FILENAME = "adapter_metadata.json"


@dataclass(frozen=True)
class QLoRAPlan:
    config_path: str
    base_model: str
    train_path: str
    val_path: str
    checkpoint_dir: str
    seed: int
    method: str
    load_in_4bit: bool
    freeze_vision_encoder: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    target_modules: list[str]
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_samples: int | None
    train_count: int
    val_count: int
    effective_train_count: int
    device: str
    dtype: str
    estimated_vram_gb: float | None
    can_train_on_this_machine: bool
    runtime_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_path": self.config_path,
            "base_model": self.base_model,
            "train_path": self.train_path,
            "val_path": self.val_path,
            "checkpoint_dir": self.checkpoint_dir,
            "seed": self.seed,
            "method": self.method,
            "load_in_4bit": self.load_in_4bit,
            "freeze_vision_encoder": self.freeze_vision_encoder,
            "lora": {
                "rank": self.lora_rank,
                "alpha": self.lora_alpha,
                "dropout": self.lora_dropout,
                "target_modules": self.target_modules,
            },
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_samples": self.max_samples,
            "train_count": self.train_count,
            "val_count": self.val_count,
            "effective_train_count": self.effective_train_count,
            "device": self.device,
            "dtype": self.dtype,
            "estimated_vram_gb": self.estimated_vram_gb,
            "can_train_on_this_machine": self.can_train_on_this_machine,
            "runtime_warnings": self.runtime_warnings,
        }


class SFTJsonlDataset(torch.utils.data.Dataset):
    def __init__(self, path: str | Path, max_samples: int | None = None) -> None:
        rows = read_jsonl(str(path))
        self.rows = rows[:max_samples] if max_samples else rows
        if not self.rows:
            raise ValueError(f"SFT dataset is empty: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def positive_float(value: Any, field_name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be positive") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def validate_dropout(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.lora_dropout must be a number in [0, 1)") from exc
    if not 0 <= parsed < 1:
        raise ValueError("training.lora_dropout must be in [0, 1)")
    return parsed


def validate_target_modules(value: Any) -> list[str]:
    if value is None:
        return list(DEFAULT_TARGET_MODULES)
    if not isinstance(value, list) or not value:
        raise ValueError("training.target_modules must be a non-empty list")
    modules = [str(item).strip() for item in value]
    if any(not item for item in modules):
        raise ValueError("training.target_modules must not contain empty values")
    return modules


def count_jsonl_rows(path: str | Path) -> int:
    with Path(path).open("r", encoding="utf-8") as file:
        return sum(1 for line in file if line.strip())


def require_file(path: str | Path, field_name: str) -> str:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"{field_name} not found at {resolved}. "
            "Run `python -m src.data_utils --mode build-sft` first."
        )
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a file: {resolved}")
    return str(resolved)


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
    commit = result.stdout.strip()
    return commit or None


def split_hash_from_manifest(config: Mapping[str, Any]) -> str | None:
    data_config = config.get("data", {})
    data_manifest_path = (
        data_config.get("split_manifest_path")
        if isinstance(data_config, Mapping)
        else None
    )
    for candidate in (
        config.get("split_manifest_path"),
        data_manifest_path,
        "data/processed/split_manifest.json",
    ):
        if not candidate:
            continue
        path = Path(str(candidate))
        if path.exists():
            manifest = read_json(str(path))
            return manifest.get("split_hash")
    return None


def dataset_hash(train_path: str | Path, val_path: str | Path) -> str:
    payload = {
        "train_path": str(train_path),
        "train_sha256": file_sha256(train_path),
        "val_path": str(val_path),
        "val_sha256": file_sha256(val_path),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha256(encoded).hexdigest()


def runtime_capability(load_in_4bit: bool) -> tuple[bool, str, list[str]]:
    warnings: list[str] = []
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"

    if not load_in_4bit:
        return cuda_available, device, warnings

    if platform.system() != "Linux":
        warnings.append("4-bit QLoRA requires Linux for bitsandbytes in this project.")
    if not cuda_available:
        warnings.append("4-bit QLoRA requires a CUDA GPU.")
    if importlib.util.find_spec("bitsandbytes") is None:
        warnings.append("bitsandbytes is not installed in this environment.")

    return not warnings, device, warnings


def parse_model_size_billion(model_name: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*B", model_name, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def estimate_vram_gb(model_name: str, load_in_4bit: bool) -> float | None:
    params_b = parse_model_size_billion(model_name)
    if params_b is None:
        return None
    bytes_per_param = 0.5 if load_in_4bit else 2.0
    base_gb = params_b * bytes_per_param
    return round(base_gb * 1.8 + 2.0, 2)


def resolve_dtype(training_config: Mapping[str, Any], device: str) -> str:
    configured = training_config.get("dtype")
    if configured:
        return str(configured)
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return "bfloat16"
    if device == "cuda":
        return "float16"
    return "float32"


def validate_training_config(
    config: Mapping[str, Any],
    config_path: str,
    max_samples_override: int | None = None,
) -> QLoRAPlan:
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    output_config = config.get("output", {})

    if not isinstance(model_config, Mapping) or not model_config.get("name"):
        raise ValueError("configs/qlora.yaml must define model.name")
    if not isinstance(training_config, Mapping):
        raise ValueError("configs/qlora.yaml must define a training section")
    if training_config.get("method", "qlora") != "qlora":
        raise ValueError("training.method must be 'qlora'")

    train_path = require_file(data_config.get("train_path", ""), "data.train_path")
    val_path = require_file(data_config.get("val_path", ""), "data.val_path")

    lora_rank = positive_int(training_config.get("lora_rank"), "training.lora_rank")
    lora_alpha = positive_int(training_config.get("lora_alpha"), "training.lora_alpha")
    lora_dropout = validate_dropout(training_config.get("lora_dropout", 0.0))
    learning_rate = positive_float(
        training_config.get("learning_rate"),
        "training.learning_rate",
    )
    num_train_epochs = positive_float(
        training_config.get("num_train_epochs"),
        "training.num_train_epochs",
    )
    batch_size = positive_int(
        training_config.get("per_device_train_batch_size"),
        "training.per_device_train_batch_size",
    )
    grad_accum = positive_int(
        training_config.get("gradient_accumulation_steps"),
        "training.gradient_accumulation_steps",
    )
    configured_max_samples = training_config.get("max_samples")
    max_samples = (
        positive_int(max_samples_override, "--max-samples")
        if max_samples_override is not None
        else positive_int(configured_max_samples, "training.max_samples")
        if configured_max_samples is not None
        else None
    )
    seed = positive_int(training_config.get("seed", 42), "training.seed")
    target_modules = validate_target_modules(training_config.get("target_modules"))
    load_in_4bit = bool(training_config.get("load_in_4bit", True))
    freeze_vision_encoder = bool(training_config.get("freeze_vision_encoder", True))
    can_train, device, warnings = runtime_capability(load_in_4bit)
    dtype = resolve_dtype(training_config, device)

    train_count = count_jsonl_rows(train_path)
    val_count = count_jsonl_rows(val_path)
    effective_train_count = min(train_count, max_samples) if max_samples else train_count
    if train_count == 0:
        raise ValueError(f"data.train_path is empty: {train_path}")
    if val_count == 0:
        raise ValueError(f"data.val_path is empty: {val_path}")

    checkpoint_dir = str(output_config.get("checkpoint_dir", "checkpoints/qlora_adapter"))
    return QLoRAPlan(
        config_path=config_path,
        base_model=str(model_config["name"]),
        train_path=train_path,
        val_path=val_path,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        method="qlora",
        load_in_4bit=load_in_4bit,
        freeze_vision_encoder=freeze_vision_encoder,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        max_samples=max_samples,
        train_count=train_count,
        val_count=val_count,
        effective_train_count=effective_train_count,
        device=device,
        dtype=dtype,
        estimated_vram_gb=estimate_vram_gb(str(model_config["name"]), load_in_4bit),
        can_train_on_this_machine=can_train,
        runtime_warnings=warnings,
    )


def torch_dtype_from_name(dtype: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype}") from exc


def load_transformers_model_class():
    import transformers

    for class_name in (
        "AutoModelForImageTextToText",
        "AutoModelForVision2Seq",
        "AutoModelForCausalLM",
    ):
        model_class = getattr(transformers, class_name, None)
        if model_class is not None:
            return model_class
    raise ImportError(
        "Could not find a compatible AutoModel class in transformers. "
        "Upgrade transformers or add a model-specific loader."
    )


def freeze_vision_encoder(model: Any) -> int:
    frozen = 0
    keywords = ("vision", "visual", "image_tower", "vision_tower")
    for name, parameter in model.named_parameters():
        if any(keyword in name.lower() for keyword in keywords):
            parameter.requires_grad = False
            frozen += parameter.numel()
    return frozen


def parameter_counts(model: Any) -> dict[str, int | float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "trainable_percent": round((trainable / total) * 100, 4) if total else 0.0,
    }


def build_metadata(
    config: Mapping[str, Any],
    plan: QLoRAPlan,
    parameter_summary: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> dict[str, Any]:
    return {
        "schema_version": "qlora-adapter-metadata-v1",
        "created_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "status": status,
        "base_model": plan.base_model,
        "checkpoint_dir": plan.checkpoint_dir,
        "commit_hash": git_commit_hash(),
        "dataset": {
            "train_path": plan.train_path,
            "val_path": plan.val_path,
            "train_count": plan.train_count,
            "val_count": plan.val_count,
            "effective_train_count": plan.effective_train_count,
            "dataset_hash": dataset_hash(plan.train_path, plan.val_path),
            "split_hash": split_hash_from_manifest(config),
        },
        "training": {
            "seed": plan.seed,
            "load_in_4bit": plan.load_in_4bit,
            "freeze_vision_encoder": plan.freeze_vision_encoder,
            "dtype": plan.dtype,
            "device": plan.device,
            "learning_rate": plan.learning_rate,
            "num_train_epochs": plan.num_train_epochs,
            "per_device_train_batch_size": plan.per_device_train_batch_size,
            "gradient_accumulation_steps": plan.gradient_accumulation_steps,
            "max_samples": plan.max_samples,
            "estimated_vram_gb": plan.estimated_vram_gb,
        },
        "lora": {
            "rank": plan.lora_rank,
            "alpha": plan.lora_alpha,
            "dropout": plan.lora_dropout,
            "target_modules": plan.target_modules,
        },
        "parameters": dict(parameter_summary or {}),
        "runtime_warnings": plan.runtime_warnings,
    }


def write_checkpoint_metadata(
    output_dir: str | Path,
    metadata: Mapping[str, Any],
    filename: str = METADATA_FILENAME,
) -> Path:
    path = Path(output_dir) / filename
    write_json(dict(metadata), str(path))
    return path


def validate_resume_checkpoint(path: str | None) -> str | None:
    if not path:
        return None
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {checkpoint_path}")
    if not checkpoint_path.is_dir():
        raise ValueError(f"Resume checkpoint must be a directory: {checkpoint_path}")
    return str(checkpoint_path)


def run_training(
    config: Mapping[str, Any],
    plan: QLoRAPlan,
    resume_from_checkpoint: str | None = None,
) -> dict[str, Any]:
    if not plan.can_train_on_this_machine:
        details = "; ".join(plan.runtime_warnings) or "CUDA GPU is unavailable."
        raise RuntimeError(
            "QLoRA training is GPU-only for this project. "
            f"Cannot start training: {details}"
        )

    torch.manual_seed(plan.seed)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Trainer, TrainingArguments

    model_class = load_transformers_model_class()
    quantization_config = None
    if plan.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype_from_name(plan.dtype),
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(plan.base_model)
    model = model_class.from_pretrained(
        plan.base_model,
        quantization_config=quantization_config,
        torch_dtype=torch_dtype_from_name(plan.dtype),
        device_map="auto",
    )

    frozen_vision_params = (
        freeze_vision_encoder(model) if plan.freeze_vision_encoder else 0
    )
    if plan.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=plan.lora_rank,
        lora_alpha=plan.lora_alpha,
        lora_dropout=plan.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=plan.target_modules,
    )
    model = get_peft_model(model, peft_config)
    param_summary = parameter_counts(model)
    param_summary["frozen_vision_parameters"] = frozen_vision_params

    train_dataset = SFTJsonlDataset(plan.train_path, max_samples=plan.max_samples)
    data_collator = SFTDataCollator(processor=processor)
    training_args = TrainingArguments(
        output_dir=plan.checkpoint_dir,
        num_train_epochs=plan.num_train_epochs,
        per_device_train_batch_size=plan.per_device_train_batch_size,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        learning_rate=plan.learning_rate,
        logging_steps=int(config.get("training", {}).get("logging_steps", 1)),
        save_steps=int(config.get("training", {}).get("save_steps", 20)),
        save_total_limit=int(config.get("training", {}).get("save_total_limit", 1)),
        remove_unused_columns=False,
        report_to=[],
        seed=plan.seed,
        fp16=plan.dtype == "float16",
        bf16=plan.dtype == "bfloat16",
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(plan.checkpoint_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(plan.checkpoint_dir)

    metadata = build_metadata(
        config,
        plan,
        parameter_summary=param_summary,
        status="completed",
    )
    metadata_path = write_checkpoint_metadata(plan.checkpoint_dir, metadata)
    return {"metadata_path": str(metadata_path), "parameters": param_summary}


def load_and_validate_plan(
    config_path: str,
    max_samples_override: int | None = None,
) -> tuple[dict, QLoRAPlan]:
    config = load_config(config_path)
    plan = validate_training_config(
        config,
        config_path=config_path,
        max_samples_override=max_samples_override,
    )
    return config, plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA smoke trainer for traffic-legal-vlm.")
    parser.add_argument("--config", default="configs/qlora.yaml")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print a training plan without loading model weights.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override training.max_samples for smoke runs.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Resume from an existing Trainer checkpoint directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    resume_from_checkpoint = validate_resume_checkpoint(args.resume_from_checkpoint)
    config, plan = load_and_validate_plan(
        args.config,
        max_samples_override=args.max_samples,
    )

    payload = {
        "mode": "dry-run" if args.dry_run else "train",
        "plan": plan.to_dict(),
        "resume_from_checkpoint": resume_from_checkpoint,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))

    if args.dry_run:
        return

    result = run_training(
        config,
        plan,
        resume_from_checkpoint=resume_from_checkpoint,
    )
    print(json.dumps({"mode": "train", "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
