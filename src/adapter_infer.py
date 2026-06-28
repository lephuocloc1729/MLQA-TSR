from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Protocol

from src.collator import (
    format_sft_chat_template_texts,
    format_sft_text,
    load_rgb_image,
)
from src.data_utils import (
    build_sft_record,
    load_processed_law_article_index,
    load_split_samples,
)
from src.evaluate import normalize_gold_answer, normalize_prediction_answer
from src.schemas import Evidence, Prediction, Query
from src.utils import file_sha256, load_config, read_json, read_jsonl, utc_now_iso, write_jsonl
from src.vlm import parse_prediction


SCHEMA_VERSION = "adapter-diagnostic-v1"
DEFAULT_CONFIG_PATH = "configs/experiments/w4_adapter_diag.yaml"
DEFAULT_ADAPTER_PATH = "checkpoints/qlora_adapter"
DEFAULT_OUTPUT_PATH = "data/outputs/experiments/w4_adapter_diag.jsonl"
DEFAULT_MAX_NEW_TOKENS = 320


class AdapterGenerator(Protocol):
    def generate(self, record: Mapping[str, Any], max_new_tokens: int) -> str:
        ...


def require_adapter_metadata(adapter_path: str | Path) -> dict[str, Any]:
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"Adapter path not found: {adapter_dir}. "
            "Run `python -m src.train_qlora --config configs/qlora.yaml --max-samples 80` "
            "or pass --adapter to an existing local adapter directory."
        )
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter path must be a directory: {adapter_dir}")

    metadata_path = adapter_dir / "adapter_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Adapter metadata not found: {metadata_path}. "
            "W4-01 diagnostics require adapter_metadata.json for traceability."
        )
    metadata = read_json(str(metadata_path))
    if not isinstance(metadata, dict):
        raise ValueError(f"Adapter metadata must be a JSON object: {metadata_path}")
    return metadata


def adapter_metadata_summary(adapter_path: str | Path) -> dict[str, Any]:
    adapter_dir = Path(adapter_path)
    metadata_path = adapter_dir / "adapter_metadata.json"
    metadata = require_adapter_metadata(adapter_dir)
    dataset = metadata.get("dataset", {}) if isinstance(metadata.get("dataset"), dict) else {}
    training = metadata.get("training", {}) if isinstance(metadata.get("training"), dict) else {}
    lora = metadata.get("lora", {}) if isinstance(metadata.get("lora"), dict) else {}
    parameters = (
        metadata.get("parameters", {})
        if isinstance(metadata.get("parameters"), dict)
        else {}
    )
    return {
        "adapter_path": str(adapter_dir),
        "metadata_path": str(metadata_path),
        "metadata_sha256": file_sha256(metadata_path),
        "schema_version": metadata.get("schema_version"),
        "status": metadata.get("status"),
        "created_at": metadata.get("created_at"),
        "base_model": metadata.get("base_model"),
        "checkpoint_dir": metadata.get("checkpoint_dir"),
        "commit_hash": metadata.get("commit_hash"),
        "effective_train_count": dataset.get("effective_train_count"),
        "train_count": dataset.get("train_count"),
        "val_count": dataset.get("val_count"),
        "dataset_hash": dataset.get("dataset_hash"),
        "split_hash": dataset.get("split_hash"),
        "device": training.get("device"),
        "dtype": training.get("dtype"),
        "max_samples": training.get("max_samples"),
        "lora": {
            "rank": lora.get("rank"),
            "alpha": lora.get("alpha"),
            "dropout": lora.get("dropout"),
            "target_modules": lora.get("target_modules", []),
        },
        "parameters": {
            "total": parameters.get("total"),
            "trainable": parameters.get("trainable"),
            "trainable_percent": parameters.get("trainable_percent"),
        },
    }


def _supports_chat_template(processor: Any) -> bool:
    return callable(getattr(processor, "apply_chat_template", None))


class LocalAdapterGenerator:
    """GPU/local adapter generator. Imports model packages only when used."""

    def __init__(
        self,
        adapter_path: str | Path,
        metadata: Mapping[str, Any],
        dtype: str | None = None,
    ) -> None:
        import torch
        import transformers
        from peft import PeftModel

        adapter_dir = Path(adapter_path)
        base_model = str(metadata.get("base_model") or "").strip()
        if not base_model:
            raise ValueError("adapter_metadata.json must include base_model")

        processor_source = str(adapter_dir if (adapter_dir / "processor_config.json").exists() else base_model)
        self.processor = transformers.AutoProcessor.from_pretrained(
            processor_source,
            trust_remote_code=True,
        )

        dtype_name = dtype or str(
            (metadata.get("training") or {}).get("dtype") or "auto"
        )
        torch_dtype = self._resolve_torch_dtype(torch, dtype_name)
        model_class = getattr(
            transformers,
            "AutoModelForVision2Seq",
            transformers.AutoModelForCausalLM,
        )
        base = model_class.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(base, str(adapter_dir))
        self.model.eval()

    @staticmethod
    def _resolve_torch_dtype(torch_module: Any, dtype_name: str) -> Any:
        normalized = str(dtype_name).lower()
        if normalized in {"auto", "", "none"}:
            return "auto"
        if normalized == "bfloat16":
            return torch_module.bfloat16
        if normalized == "float16":
            return torch_module.float16
        if normalized == "float32":
            return torch_module.float32
        return "auto"

    def _prompt_text(self, record: Mapping[str, Any]) -> str:
        if _supports_chat_template(self.processor):
            _, prompt_text = format_sft_chat_template_texts(record, self.processor)
            return prompt_text
        _, prompt_text = format_sft_text(record, assistant_marker="\n\nASSISTANT:\n")
        return prompt_text

    def generate(self, record: Mapping[str, Any], max_new_tokens: int) -> str:
        import torch

        prompt_text = self._prompt_text(record)
        image_path = record.get("image_path")
        images = [load_rgb_image(str(image_path))] if image_path else None
        inputs = self.processor(
            text=[prompt_text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = {
                key: value.to(self.model.device) if hasattr(value, "to") else value
                for key, value in dict(inputs).items()
            }

        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        input_ids = inputs.get("input_ids")
        if input_ids is not None:
            generated = generated[:, input_ids.shape[1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _sample_to_sft_like_record(
    sample: Mapping[str, Any],
    article_index: Mapping[str, dict],
    split: str,
) -> dict[str, Any]:
    return build_sft_record(sample, article_index, split=split)


def load_adapter_inputs(
    config: Mapping[str, Any],
    split: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    split = split.strip().lower()
    if split == "sft_val":
        path = Path(
            str(
                config.get("adapter_diagnostic", {}).get("sft_val_path")
                or config.get("data", {}).get("sft_val_path")
                or config.get("sft", {}).get("val_output_path")
                or "data/processed/sft_val.jsonl"
            )
        )
        if not path.exists():
            raise FileNotFoundError(
                f"SFT validation file not found: {path}. "
                "Run `python -m src.data_utils --mode build-sft` first."
            )
        rows = read_jsonl(str(path))
    elif split in {"val", "validation"}:
        article_index = load_processed_law_article_index(dict(config))
        rows = [
            _sample_to_sft_like_record(sample, article_index, split="val")
            for sample in load_split_samples(dict(config), "val")
        ]
    else:
        raise ValueError("split must be one of: val, validation, sft_val")
    return rows[:limit] if limit is not None else rows


def evidence_models(record: Mapping[str, Any]) -> list[Evidence]:
    return [Evidence.model_validate(item) for item in record.get("evidence", [])]


def query_model(record: Mapping[str, Any]) -> Query:
    target = record.get("target", {}) if isinstance(record.get("target"), dict) else {}
    return Query.model_validate(
        {
            "id": record.get("id"),
            "image_id": record.get("image_id"),
            "image_path": record.get("image_path"),
            "question": _question_from_messages(record),
            "question_type": record.get("question_type"),
            "choices": _choices_from_record(record),
            "answer": target.get("answer"),
            "relevant_articles": target.get("citations", []),
        }
    )


def _question_from_messages(record: Mapping[str, Any]) -> str:
    messages = record.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, Mapping) and message.get("role") == "user":
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    return content
                if isinstance(content, list):
                    text_parts = [
                        str(item.get("text", ""))
                        for item in content
                        if isinstance(item, Mapping) and item.get("type") == "text"
                    ]
                    joined = "\n".join(part for part in text_parts if part.strip())
                    if joined:
                        return joined
    return "Adapter diagnostic prompt"


def _choices_from_record(record: Mapping[str, Any]) -> dict[str, str]:
    choices = record.get("choices")
    if isinstance(choices, dict):
        return {str(key): str(value) for key, value in choices.items()}
    target = record.get("target", {}) if isinstance(record.get("target"), dict) else {}
    answer = target.get("answer")
    question_type = record.get("question_type")
    if question_type == "Multiple choice" or answer in {"A", "B", "C", "D"}:
        return {
            "A": "A",
            "B": "B",
            "C": "C",
            "D": "D",
        }
    return {}


def looks_truncated(raw_response: str, error: Exception | None = None) -> bool:
    text = raw_response.strip()
    error_text = str(error or "").casefold()
    if any(
        marker in error_text
        for marker in ("unterminated", "unexpected end", "eof", "truncated")
    ):
        return True
    if text.startswith("```") and text.count("```") < 2:
        return True
    if text.startswith("{") and not text.endswith("}"):
        return True
    return False


def classify_parse_error(raw_response: str, error: Exception) -> dict[str, Any]:
    message = str(error)
    lowered = message.casefold()
    truncated = looks_truncated(raw_response, error)
    unsupported = "outside retrieved evidence" in lowered
    missing_citation = "requires at least one citation" in lowered
    invalid_answer = "prediction must be" in lowered or "answer must be" in lowered
    invalid_json = (
        "json" in lowered
        or "does not contain a json object" in lowered
        or "must contain exactly answer" in lowered
    )
    if truncated:
        status = "truncated"
    elif unsupported:
        status = "unsupported_citation"
    elif missing_citation:
        status = "missing_citation"
    elif invalid_answer:
        status = "invalid_answer"
    elif invalid_json:
        status = "invalid_json"
    else:
        status = "parse_error"
    return {
        "status": status,
        "success": False,
        "invalid_json": bool(invalid_json or truncated),
        "truncated_output": truncated,
        "unsupported_citation": unsupported,
        "missing_citation": missing_citation,
        "invalid_answer": invalid_answer,
        "error_type": type(error).__name__,
        "message": message,
    }


def parse_adapter_response(
    raw_response: str,
    record: Mapping[str, Any],
) -> tuple[Prediction | None, dict[str, Any]]:
    query = query_model(record)
    evidence = evidence_models(record)
    try:
        prediction = parse_prediction(raw_response, query=query, evidence=evidence)
    except Exception as exc:
        return None, classify_parse_error(raw_response, exc)
    return prediction, {
        "status": "success",
        "success": True,
        "invalid_json": False,
        "truncated_output": looks_truncated(raw_response),
        "unsupported_citation": False,
        "missing_citation": False,
        "invalid_answer": False,
    }


def target_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    target = record.get("target") if isinstance(record.get("target"), dict) else {}
    return {
        "answer": target.get("answer"),
        "citations": target.get("citations", []),
    }


def exact_match(target: Mapping[str, Any], prediction: Prediction | None) -> bool:
    if prediction is None or target.get("answer") is None:
        return False
    return normalize_prediction_answer(prediction.answer) == normalize_gold_answer(
        target["answer"]
    )


def diagnostic_row(
    record: Mapping[str, Any],
    raw_response: str,
    adapter: Mapping[str, Any],
    max_new_tokens: int,
    generation_ms: float,
) -> dict[str, Any]:
    parse_start = time.perf_counter()
    prediction, parse = parse_adapter_response(raw_response, record)
    parse_ms = (time.perf_counter() - parse_start) * 1000
    target = target_payload(record)
    query = query_model(record)
    evidence = evidence_models(record)
    prediction_payload = (
        prediction.model_dump(mode="json") if prediction is not None else {
            "id": query.id,
            "question_type": query.question_type.value if query.question_type else None,
            "answer": None,
            "citations": [],
            "explanation": parse.get("message", "Adapter output could not be parsed."),
            "confidence": 0.0,
            "abstained": True,
            "raw_response": raw_response,
            "error": {
                "type": parse.get("error_type"),
                "message": parse.get("message"),
            },
        }
    )
    predicted_articles = (
        [
            {"law_id": citation.law_id, "article_id": citation.article_id}
            for citation in prediction.citations
        ]
        if prediction is not None
        else []
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "id": record.get("id"),
        "split": record.get("split"),
        "image_id": record.get("image_id"),
        "image_path": record.get("image_path"),
        "query": query.model_dump(mode="json"),
        "target": target,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "raw_response": raw_response,
        "prediction": prediction_payload,
        "predicted_articles": predicted_articles,
        "parse": parse,
        "exact_match": exact_match(target, prediction),
        "adapter": dict(adapter),
        "generation": {
            "max_new_tokens": max_new_tokens,
        },
        "timings_ms": {
            "generation": generation_ms,
            "parse": parse_ms,
            "total": generation_ms + parse_ms,
        },
        "created_at": utc_now_iso(),
    }


def run_adapter_diagnostic(
    records: list[Mapping[str, Any]],
    generator: AdapterGenerator,
    adapter: Mapping[str, Any],
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        generation_start = time.perf_counter()
        raw_response = generator.generate(record, max_new_tokens=max_new_tokens)
        generation_ms = (time.perf_counter() - generation_start) * 1000
        rows.append(
            diagnostic_row(
                record,
                raw_response=raw_response,
                adapter=adapter,
                max_new_tokens=max_new_tokens,
                generation_ms=generation_ms,
            )
        )
    return rows


def summarize_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    parse_status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("parse", {}).get("status", "unknown"))
        parse_status_counts[status] = parse_status_counts.get(status, 0) + 1
    return {
        "schema_version": "adapter-diagnostic-summary-v1",
        "sample_count": len(rows),
        "exact_match_count": sum(1 for row in rows if row.get("exact_match")),
        "exact_match_accuracy": (
            sum(1 for row in rows if row.get("exact_match")) / len(rows)
            if rows
            else 0.0
        ),
        "parse_success_count": sum(
            1 for row in rows if row.get("parse", {}).get("success")
        ),
        "invalid_json_count": sum(
            1 for row in rows if row.get("parse", {}).get("invalid_json")
        ),
        "truncated_output_count": sum(
            1 for row in rows if row.get("parse", {}).get("truncated_output")
        ),
        "unsupported_citation_count": sum(
            1 for row in rows if row.get("parse", {}).get("unsupported_citation")
        ),
        "missing_citation_count": sum(
            1 for row in rows if row.get("parse", {}).get("missing_citation")
        ),
        "parse_status_counts": dict(sorted(parse_status_counts.items())),
    }


def positive_int(value: Any, field_name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer") from exc
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def resolve_config_value(
    cli_value: Any,
    config: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    if cli_value is not None:
        return cli_value
    adapter_config = (
        config.get("adapter_diagnostic", {})
        if isinstance(config.get("adapter_diagnostic"), Mapping)
        else {}
    )
    return adapter_config.get(key, default)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run diagnostic batch inference for a local QLoRA adapter."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--split", choices=["val", "validation", "sft_val"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    adapter_path = str(
        resolve_config_value(args.adapter, config, "adapter_path", DEFAULT_ADAPTER_PATH)
    )
    split = str(resolve_config_value(args.split, config, "split", "val"))
    output = Path(
        str(resolve_config_value(args.output, config, "output_path", DEFAULT_OUTPUT_PATH))
    )
    max_new_tokens = positive_int(
        resolve_config_value(
            args.max_new_tokens,
            config,
            "max_new_tokens",
            DEFAULT_MAX_NEW_TOKENS,
        ),
        "--max-new-tokens",
    )
    limit = (
        positive_int(args.limit, "--limit")
        if args.limit is not None
        else config.get("adapter_diagnostic", {}).get("limit")
    )
    limit = positive_int(limit, "adapter_diagnostic.limit") if limit else None

    adapter = adapter_metadata_summary(adapter_path)
    records = load_adapter_inputs(config, split=split, limit=limit)
    generator = LocalAdapterGenerator(
        adapter_path=adapter_path,
        metadata=require_adapter_metadata(adapter_path),
    )
    rows = run_adapter_diagnostic(
        records,
        generator=generator,
        adapter=adapter,
        max_new_tokens=max_new_tokens,
    )
    write_jsonl(rows, str(output))
    summary = summarize_rows(rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"Saved adapter diagnostics to {output}")


if __name__ == "__main__":
    main()
