from pathlib import Path

import pytest

from src.evaluate import build_evaluation_artifact
from src.pipeline import (
    assert_locked_validation_split,
    build_benchmark_record,
    benchmark_output_path,
    locked_split_identity,
    model_run_metadata,
)
from src.utils import load_config


LAW_ID = "QCVN 41:2024/BGTVT"
EXPERIMENT_CONFIGS = [
    "configs/experiments/w2_b1_zero_shot.yaml",
    "configs/experiments/w2_b2_text_rag.yaml",
    "configs/experiments/w2_b3_fused_rag.yaml",
    "configs/experiments/w2_b4_few_shot_rag.yaml",
]
W3_EXPERIMENT_CONFIGS = [
    "configs/experiments/w3_b2_text_rag_real.yaml",
    "configs/experiments/w3_b5_structured_real.yaml",
]
W4_FINAL_VALIDATION_CONFIGS = [
    "configs/experiments/w4_retrieval_only.yaml",
    "configs/experiments/w4_text_rag_real.yaml",
    "configs/experiments/w4_structured_rag.yaml",
]
W5_QWEN_CONFIGS = [
    "configs/experiments/vlsp_task2_qwen25vl_7b.yaml",
    "configs/experiments/vlsp_task2_qwen25vl_3b.yaml",
]


def citation(article_id: str) -> dict:
    return {"law_id": LAW_ID, "article_id": article_id}


def sample(**overrides) -> dict:
    data = {
        "id": "val_tiny_1",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [citation("22")],
    }
    data.update(overrides)
    return data


def test_week2_experiment_configs_load_from_base_config():
    configs = [load_config(path) for path in EXPERIMENT_CONFIGS]

    assert [config["experiment"]["name"] for config in configs] == [
        "w2_b1_zero_shot",
        "w2_b2_text_rag",
        "w2_b3_fused_rag",
        "w2_b4_few_shot_rag",
    ]
    assert configs[1]["data"]["val_split_path"] == "data/processed/val_split.jsonl"
    assert configs[1]["retrieval"]["top_k"] == 5
    assert configs[3]["prompt"]["variant"] == "few_shot_rag"
    assert benchmark_output_path(configs[1]) == Path(
        "data/outputs/experiments/w2_b2_text_rag.jsonl"
    )


def test_week2_configs_use_same_locked_validation_split():
    configs = [load_config(path) for path in EXPERIMENT_CONFIGS]

    identity = assert_locked_validation_split(configs)

    assert identity["split"] == "val"
    assert identity["split_path"] == "data/processed/val_split.jsonl"
    assert identity["split_manifest_path"] == "data/processed/split_manifest.json"


def test_retrieval_final_config_freezes_locked_validation_fusion_settings():
    config = load_config("configs/experiments/retrieval_final.yaml")
    freeze = config["retrieval_freeze"]

    assert freeze["version"] == "retrieval-final-v1"
    assert freeze["locked_split"] == "data/processed/val_split.jsonl"
    assert config["data"]["val_split_path"] == freeze["locked_split"]
    assert config["experiment"]["name"] == "retrieval_final"
    assert config["experiment"]["retrieval_strategy"] == "fusion"
    assert config["experiment"]["example_retrieval_mode"] == "fusion"
    assert config["experiment"]["mock"] is True
    assert config["retrieval"]["top_k"] == 5
    assert config["retrieval"]["example_top_k"] == 3
    assert config["retrieval"]["fusion_allow_example_failure"] is False
    assert config["retrieval"]["text_weight"] == 0.7
    assert config["retrieval"]["image_weight"] == 0.3
    assert set(freeze["stretch_not_included"]) == {
        "OCR",
        "cropped-sign detection",
        "detector-driven sign retrieval",
    }
    assert locked_split_identity(config)["split_path"] == freeze["locked_split"]


def test_week3_real_experiment_configs_load_and_use_locked_split():
    configs = [load_config(path) for path in W3_EXPERIMENT_CONFIGS]

    identity = assert_locked_validation_split(configs)

    assert identity["split"] == "val"
    assert identity["split_path"] == "data/processed/val_split.jsonl"
    assert [config["experiment"]["name"] for config in configs] == [
        "w3_b2_text_rag_real",
        "w3_b5_structured_real",
    ]
    assert all(config["experiment"]["mock"] is False for config in configs)
    assert configs[0]["experiment"]["retrieval_strategy"] == "text"
    assert configs[1]["experiment"]["retrieval_strategy"] == "fusion"
    assert configs[1]["experiment"]["prompt_variant"] == "structured_legal_rag"
    assert configs[1]["retrieval"]["fusion_allow_example_failure"] is False
    assert configs[0]["model"]["max_new_tokens"] == 512


def test_week4_adapter_diagnostic_config_loads_safe_defaults():
    config = load_config("configs/experiments/w4_adapter_diag.yaml")

    assert config["experiment"]["name"] == "w4_adapter_diag"
    assert config["experiment"]["mock"] is False
    assert config["adapter_diagnostic"]["adapter_path"] == "checkpoints/qlora_adapter"
    assert config["adapter_diagnostic"]["split"] == "val"
    assert config["adapter_diagnostic"]["max_new_tokens"] == 320
    assert config["adapter_diagnostic"]["output_path"] == (
        "data/outputs/experiments/w4_adapter_diag.jsonl"
    )
    assert config["model"]["backend"] == "local_qlora_adapter"


def test_week4_final_validation_configs_load_and_use_locked_split():
    configs = [load_config(path) for path in W4_FINAL_VALIDATION_CONFIGS]

    identity = assert_locked_validation_split(configs)

    assert identity["split"] == "val"
    assert identity["split_path"] == "data/processed/val_split.jsonl"
    assert [config["experiment"]["name"] for config in configs] == [
        "w4_retrieval_only",
        "w4_text_rag_real",
        "w4_structured_rag",
    ]
    assert configs[0]["experiment"]["mock"] is True
    assert configs[1]["experiment"]["mock"] is False
    assert configs[2]["experiment"]["mock"] is False
    assert configs[0]["final_validation"]["role"] == (
        "retrieval_only_evidence_inspection"
    )
    assert configs[2]["final_validation"]["main_product_candidate"] is True
    assert configs[0]["experiment"]["retrieval_strategy"] == "fusion"
    assert configs[1]["experiment"]["retrieval_strategy"] == "text"
    assert configs[2]["experiment"]["retrieval_strategy"] == "fusion"
    assert configs[2]["experiment"]["prompt_variant"] == "structured_legal_rag"
    assert configs[2]["retrieval"]["fusion_allow_example_failure"] is False
    assert configs[2]["model"]["max_new_tokens"] == 512
    assert benchmark_output_path(configs[2]) == Path(
        "data/outputs/experiments/w4_structured_rag.jsonl"
    )


def test_week4_adapter_diagnostic_is_non_final_but_uses_locked_validation_paths():
    config = load_config("configs/experiments/w4_adapter_diag.yaml")

    assert locked_split_identity(config)["split_path"] == (
        "data/processed/val_split.jsonl"
    )
    assert config["experiment"]["mock"] is False
    assert config["experiment"]["retrieval_strategy"] == "oracle"
    assert config["adapter_diagnostic"]["status"] == "diagnostic_not_submission_ready"
    assert config["adapter_diagnostic"]["max_new_tokens"] == 320


def test_week5_qwen_real_vlm_configs_load_with_image_metadata():
    configs = [load_config(path) for path in W5_QWEN_CONFIGS]

    identity = assert_locked_validation_split(configs)

    assert identity["split"] == "val"
    assert identity["split_path"] == "data/processed/val_split.jsonl"
    assert [config["experiment"]["name"] for config in configs] == [
        "vlsp_task2_qwen25vl_7b",
        "vlsp_task2_qwen25vl_3b",
    ]
    assert all(config["experiment"]["mock"] is False for config in configs)
    assert all(config["model"]["include_image"] is True for config in configs)
    assert all(config["model"]["max_new_tokens"] == 512 for config in configs)
    assert all(config["model"]["temperature"] == 0.0 for config in configs)
    assert configs[0]["model"]["name"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert configs[1]["model"]["name"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert configs[0]["w5_real_vlm"]["preferred_model"] is True
    assert configs[1]["w5_real_vlm"]["fallback_reason_required"] is True
    assert configs[0]["experiment"]["prompt_variant"] == "structured_legal_rag"
    assert configs[0]["retrieval"]["fusion_allow_example_failure"] is True
    assert benchmark_output_path(configs[0]) == Path(
        "data/outputs/experiments/vlsp_task2_qwen25vl_7b.jsonl"
    )


def test_week5_qwen_model_metadata_records_serving_details(monkeypatch):
    monkeypatch.setenv("QWEN_VL_GPU_HOST", "rtx3090-box")
    config = load_config("configs/experiments/vlsp_task2_qwen25vl_7b.yaml")

    metadata = model_run_metadata(config)

    assert metadata["backend"] == "openai_compatible"
    assert metadata["name"] == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert metadata["include_image"] is True
    assert metadata["max_new_tokens"] == 512
    assert metadata["serving"] == "openai-compatible"
    assert metadata["gpu_host"] == "rtx3090-box"
    assert metadata["gpu_host_env"] == "QWEN_VL_GPU_HOST"
    assert metadata["dtype"] == "bfloat16"
    assert metadata["quantization"] == "none"


def test_split_drift_is_rejected_unless_explicitly_overridden():
    base = {
        "experiment": {"name": "a", "split": "val"},
        "data": {
            "val_split_path": "data/processed/val_split.jsonl",
            "split_manifest_path": "data/processed/split_manifest.json",
        },
    }
    drift = {
        "experiment": {"name": "b", "split": "val"},
        "data": {
            "val_split_path": "data/processed/other_val.jsonl",
            "split_manifest_path": "data/processed/split_manifest.json",
        },
    }

    with pytest.raises(ValueError, match="different validation splits"):
        assert_locked_validation_split([base, drift])

    drift["experiment"]["allow_split_override"] = True
    assert_locked_validation_split([base, drift])


def test_metrics_artifact_records_experiment_schema_and_invalid_predictions():
    config = load_config("configs/experiments/w2_b2_text_rag.yaml")
    records = [
        {
            "query": {
                **sample(),
                "relevant_articles": [citation("22")],
            },
            "predicted_articles": [citation("22")],
            "prediction": {
                "id": "val_tiny_1",
                "question_type": "Multiple choice",
                "answer": "E",
                "citations": [citation("22")],
                "explanation": "Invalid answer for accounting test.",
            },
            "timings_ms": {"retrieval": 2.0, "generation": 3.0},
        }
    ]

    artifact = build_evaluation_artifact(records, config=config)

    assert artifact["schema_version"] == "w2-ablation-metrics-v1"
    assert artifact["config_name"] == "w2_b2_text_rag"
    assert artifact["experiment"]["label"] == "B2_text_rag"
    assert artifact["mock"] is True
    assert artifact["retrieval"]["f2"] == 1.0
    assert artifact["qa"]["accuracy"] == 0.0
    assert artifact["invalid_prediction_count"] == 1
    assert artifact["latency_ms"]["mean"] == 5.0


def test_evaluator_prefers_retrieved_articles_for_retrieval_metrics():
    config = load_config("configs/experiments/w2_b2_text_rag.yaml")
    records = [
        {
            "query": {
                **sample(relevant_articles=[citation("22"), citation("41")]),
            },
            "predicted_articles": [citation("22"), citation("41")],
            "prediction": {
                "id": "val_tiny_1",
                "question_type": "Multiple choice",
                "answer": "B",
                "citations": [citation("22")],
                "explanation": "The answer cites only one article.",
            },
        }
    ]

    artifact = build_evaluation_artifact(records, config=config)

    assert artifact["retrieval"]["precision"] == 1.0
    assert artifact["retrieval"]["recall"] == 1.0
    assert artifact["retrieval"]["f2"] == 1.0


def test_zero_shot_mock_benchmark_record_has_no_retrieval_or_gold_leakage():
    config = load_config("configs/experiments/w2_b1_zero_shot.yaml")

    record = build_benchmark_record(sample(answer="D"), config)

    assert record["schema_version"] == "w2-ablation-v1"
    assert record["experiment"]["name"] == "w2_b1_zero_shot"
    assert record["mock"] is True
    assert record["evidence"] == []
    assert record["predicted_articles"] == []
    assert record["prediction"]["abstained"] is True
    assert record["prediction"]["answer"] == "A"
    assert record["query"]["answer"] == "D"
    assert locked_split_identity(config)["split_path"] == "data/processed/val_split.jsonl"
