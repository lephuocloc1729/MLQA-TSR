import json
from pathlib import Path
from types import SimpleNamespace

import src.pipeline as pipeline
from src.schemas import Evidence, PipelineResult, Prediction


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeVLM:
    def __init__(self) -> None:
        self.calls = []

    def build_messages(self, query, evidence, examples=None, variant=None):
        self.calls.append(
            {
                "query": query.id,
                "evidence_count": len(evidence),
                "example_count": len(examples or []),
                "variant": variant.value if hasattr(variant, "value") else variant,
            }
        )
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"fake prompt for {query.id}",
                    }
                ],
            }
        ]


class FakeRealVLM(FakeVLM):
    def __init__(self, prediction=None, error: Exception | None = None) -> None:
        super().__init__()
        self.prediction = prediction
        self.error = error

    def answer(self, query, evidence, examples=None, variant=None):
        self.build_messages(query, evidence, examples=examples, variant=variant)
        if self.error:
            raise self.error
        return self.prediction


def sample(**overrides) -> dict:
    data = {
        "id": "val_tiny_1",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    data.update(overrides)
    return data


def fake_evidence(article_id: str = "22") -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id=article_id,
        title=f"Điều {article_id}",
        content=f"Nội dung pháp lý của Điều {article_id}.",
        score=0.93,
        rank=1,
        retrieval_method="fusion",
    )


def fake_retrieve_for_sample(sample, config, runtime):
    return [fake_evidence()], [], []


def fake_retrieve_prompt_examples(sample, config, runtime):
    return [
        {
            "sample_id": "train_example_1",
            "image_id": "train_img_1",
            "question": "Câu hỏi mẫu?",
            "question_type": "Multiple choice",
            "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
            "answer": "A",
            "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
            "image_path": "data/raw/train_data/train_images/train_img_1.jpg",
            "split": "train",
            "score": 0.8,
            "retrieval_mode": "fusion",
        }
    ]


def config(**experiment_overrides) -> dict:
    experiment = {
        "name": "test_release_pipeline",
        "label": "Test release pipeline",
        "mock": True,
        "retrieval_strategy": "fusion",
        "prompt_variant": "few_shot_rag",
        "use_examples": True,
    }
    experiment.update(experiment_overrides)
    return {
        "project": {"name": "traffic-legal-vlm", "seed": 42},
        "experiment": experiment,
        "retrieval": {"top_k": 1, "example_top_k": 1},
        "prompt": {"variant": "few_shot_rag", "top_examples": 1},
        "model": {"name": "fake-vlm"},
        "data": {
            "val_split_path": "unused.jsonl",
            "train_split_path": "unused.jsonl",
        },
    }


def test_benchmark_record_uses_fake_retriever_and_fake_vlm(monkeypatch):
    fake_vlm = FakeVLM()
    fake_runtime = SimpleNamespace(vlm=fake_vlm)
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)
    monkeypatch.setattr(
        pipeline,
        "retrieve_prompt_examples",
        fake_retrieve_prompt_examples,
    )

    record = pipeline.build_benchmark_record(
        sample(),
        config(),
        runtime=fake_runtime,
    )

    assert record["schema_version"] == "w2-ablation-v1"
    assert record["experiment"]["name"] == "test_release_pipeline"
    assert record["experiment"]["prompt_variant"] == "few_shot_rag"
    assert record["mock"] is True
    assert record["predicted_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert record["prompt"]["example_count"] == 1
    assert fake_vlm.calls == [
        {
            "query": "val_tiny_1",
            "evidence_count": 1,
            "example_count": 1,
            "variant": "few_shot_rag",
        }
    ]


def test_generated_result_jsonl_validates_with_pipeline_result(monkeypatch, tmp_path):
    output_path = tmp_path / "predictions.jsonl"
    monkeypatch.setattr(pipeline, "load_benchmark_samples", lambda config, limit: [sample()])
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)
    monkeypatch.setattr(pipeline, "retrieve_prompt_examples", lambda *args: [])

    path = pipeline.run_benchmark(
        config(prompt_variant="text_rag", use_examples=False),
        limit=1,
        output_path=output_path,
    )

    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    row = rows[0]
    result = PipelineResult.model_validate(
        {
            "query": row["query"],
            "evidence": row["evidence"],
            "prediction": row["prediction"],
            "timings_ms": row["timings_ms"],
        }
    )

    assert result.query.id == "val_tiny_1"
    assert result.evidence[0].uid == f"{LAW_ID}#22"
    assert row["experiment"]["name"] == "test_release_pipeline"


def test_non_mock_benchmark_uses_real_vlm_runtime(monkeypatch):
    prediction = Prediction(
        id="val_tiny_1",
        question_type="Multiple choice",
        answer="B",
        citations=[{"law_id": LAW_ID, "article_id": "22"}],
        explanation="Dựa trên Điều 22.",
        confidence=0.8,
        abstained=False,
        raw_response='{"answer":"B"}',
    )
    fake_vlm = FakeRealVLM(prediction=prediction)
    fake_runtime = SimpleNamespace(vlm=fake_vlm)
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)
    monkeypatch.setattr(pipeline, "retrieve_prompt_examples", lambda *args: [])

    record = pipeline.build_benchmark_record(
        sample(),
        config(mock=False, retrieval_strategy="text", prompt_variant="text_rag"),
        runtime=fake_runtime,
    )

    result = PipelineResult.model_validate(
        {
            "query": record["query"],
            "evidence": record["evidence"],
            "prediction": record["prediction"],
            "timings_ms": record["timings_ms"],
        }
    )
    assert record["mock"] is False
    assert record["experiment"]["mock"] is False
    assert result.prediction.answer == "B"
    assert fake_vlm.calls[-1]["variant"] == "text_rag"


def test_non_mock_model_error_is_recorded_as_invalid_sample(monkeypatch):
    fake_vlm = FakeRealVLM(error=ValueError("bad model JSON"))
    fake_runtime = SimpleNamespace(vlm=fake_vlm)
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)
    monkeypatch.setattr(pipeline, "retrieve_prompt_examples", lambda *args: [])

    record = pipeline.build_benchmark_record(
        sample(),
        config(mock=False, retrieval_strategy="text", prompt_variant="text_rag"),
        runtime=fake_runtime,
    )

    assert record["mock"] is False
    assert record["prediction"]["answer"] is None
    assert record["prediction"]["error"]["type"] == "ValueError"
    assert record["diagnostics"][-1]["type"] == "model_error"
