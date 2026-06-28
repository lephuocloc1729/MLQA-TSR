import json
from pathlib import Path

import pytest

from src.adapter_infer import (
    adapter_metadata_summary,
    load_adapter_inputs,
    parse_args,
    require_adapter_metadata,
    run_adapter_diagnostic,
    summarize_rows,
)
from src.utils import write_jsonl


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeGenerator:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = []

    def generate(self, record, max_new_tokens: int) -> str:
        self.calls.append({"id": record["id"], "max_new_tokens": max_new_tokens})
        return self.responses.pop(0)


def metadata() -> dict:
    return {
        "schema_version": "qlora-adapter-metadata-v1",
        "status": "completed",
        "created_at": "2026-06-28T08:09:36.814182Z",
        "base_model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "checkpoint_dir": "checkpoints/qlora_adapter",
        "commit_hash": "abc123",
        "dataset": {
            "train_count": 421,
            "val_count": 109,
            "effective_train_count": 80,
            "dataset_hash": "dataset-hash",
            "split_hash": "split-hash",
        },
        "training": {
            "device": "cuda",
            "dtype": "bfloat16",
            "max_samples": 80,
        },
        "lora": {
            "rank": 8,
            "alpha": 16,
            "dropout": 0.05,
            "target_modules": ["q_proj"],
        },
        "parameters": {
            "total": 2052600832,
            "trainable": 18576384,
            "trainable_percent": 0.905,
        },
    }


def write_adapter(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_metadata.json").write_text(
        json.dumps(metadata(), ensure_ascii=False),
        encoding="utf-8",
    )
    return adapter_dir


def sft_record(**overrides) -> dict:
    record = {
        "id": "val_1",
        "split": "val",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question_type": "Multiple choice",
        "messages": [
            {"role": "user", "content": "Question and retrieved legal evidence"},
            {"role": "assistant", "content": '{"answer":"B"}'},
        ],
        "target": {
            "answer": "B",
            "citations": [{"law_id": LAW_ID, "article_id": "22"}],
        },
        "evidence": [
            {
                "law_id": LAW_ID,
                "article_id": "22",
                "title": "Điều 22",
                "content": "Nội dung pháp lý.",
                "score": 1.0,
                "rank": 1,
                "retrieval_method": "oracle",
            }
        ],
    }
    record.update(overrides)
    return record


def valid_response(article_id: str = "22", answer: str = "B") -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": [{"law_id": LAW_ID, "article_id": article_id}],
            "explanation": "Dựa trên điều luật được trích dẫn.",
            "confidence": 0.8,
            "abstained": False,
        },
        ensure_ascii=False,
    )


def test_adapter_metadata_summary_reads_traceable_fields(tmp_path):
    adapter_dir = write_adapter(tmp_path)

    summary = adapter_metadata_summary(adapter_dir)

    assert summary["metadata_path"].endswith("adapter_metadata.json")
    assert len(summary["metadata_sha256"]) == 64
    assert summary["base_model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert summary["effective_train_count"] == 80
    assert summary["lora"]["rank"] == 8


def test_fake_adapter_output_parses_to_valid_row(tmp_path):
    adapter = adapter_metadata_summary(write_adapter(tmp_path))
    rows = run_adapter_diagnostic(
        [sft_record()],
        generator=FakeGenerator([valid_response()]),
        adapter=adapter,
        max_new_tokens=320,
    )

    row = rows[0]
    assert row["schema_version"] == "adapter-diagnostic-v1"
    assert row["prediction"]["answer"] == "B"
    assert row["predicted_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert row["exact_match"] is True
    assert row["parse"]["status"] == "success"
    assert row["generation"]["max_new_tokens"] == 320
    assert row["adapter"]["effective_train_count"] == 80


def test_truncated_json_is_counted_separately(tmp_path):
    adapter = adapter_metadata_summary(write_adapter(tmp_path))
    rows = run_adapter_diagnostic(
        [sft_record()],
        generator=FakeGenerator(['{"answer":"B","citations":[']),
        adapter=adapter,
        max_new_tokens=160,
    )
    summary = summarize_rows(rows)

    assert rows[0]["parse"]["status"] == "truncated"
    assert rows[0]["parse"]["invalid_json"] is True
    assert rows[0]["parse"]["truncated_output"] is True
    assert rows[0]["exact_match"] is False
    assert summary["truncated_output_count"] == 1
    assert summary["invalid_json_count"] == 1


def test_unsupported_citation_is_marked_invalid(tmp_path):
    adapter = adapter_metadata_summary(write_adapter(tmp_path))
    rows = run_adapter_diagnostic(
        [sft_record()],
        generator=FakeGenerator([valid_response(article_id="999")]),
        adapter=adapter,
        max_new_tokens=320,
    )

    assert rows[0]["parse"]["status"] == "unsupported_citation"
    assert rows[0]["parse"]["unsupported_citation"] is True
    assert rows[0]["prediction"]["answer"] is None
    assert summarize_rows(rows)["unsupported_citation_count"] == 1


def test_missing_adapter_path_fails_helpfully(tmp_path):
    with pytest.raises(FileNotFoundError, match="Adapter path not found"):
        require_adapter_metadata(tmp_path / "missing_adapter")


def test_load_sft_val_split_without_checkpoint_or_gpu(tmp_path):
    path = tmp_path / "sft_val.jsonl"
    write_jsonl([sft_record()], path)

    rows = load_adapter_inputs(
        {"adapter_diagnostic": {"sft_val_path": str(path)}},
        split="sft_val",
        limit=1,
    )

    assert rows[0]["id"] == "val_1"
    assert rows[0]["target"]["answer"] == "B"


def test_cli_parser_supports_required_flags(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "adapter_infer",
            "--adapter",
            "checkpoints/qlora_adapter",
            "--split",
            "val",
            "--limit",
            "3",
            "--max-new-tokens",
            "320",
            "--output",
            "data/outputs/experiments/w4_adapter_diag.jsonl",
        ],
    )

    args = parse_args()

    assert args.adapter == "checkpoints/qlora_adapter"
    assert args.split == "val"
    assert args.limit == 3
    assert args.max_new_tokens == 320
    assert args.output == "data/outputs/experiments/w4_adapter_diag.jsonl"
