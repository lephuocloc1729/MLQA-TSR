import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.pipeline as pipeline
from src.data_utils import load_vlsp_test_samples
from src.schemas import Evidence
from src.utils import write_json


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeVLM:
    backend = "none"
    model_name = "fake-vlm"
    temperature = 0.0
    max_new_tokens = 64
    include_image = False

    def __init__(self) -> None:
        self.calls = []

    def build_messages(self, query, evidence, examples=None, variant=None):
        self.calls.append(
            {
                "query": query.id,
                "evidence_count": len(evidence),
                "variant": variant.value if hasattr(variant, "value") else variant,
            }
        )
        return [{"role": "user", "content": [{"type": "text", "text": "fake"}]}]


def sample(sample_id: str, image_id: str, **overrides) -> dict:
    row = {
        "id": sample_id,
        "image_id": image_id,
        "question": "Đây là biển báo gì?",
    }
    row.update(overrides)
    return row


def task2_sample(sample_id: str, image_id: str, **overrides) -> dict:
    row = sample(
        sample_id,
        image_id,
        question="Biển báo này thuộc nhóm nào?",
        question_type="Multiple choice",
        choices={"A": "Biển cấm", "B": "Biển hiệu lệnh", "C": "Biển chỉ dẫn", "D": "Biển phụ"},
        relevant_articles=[{"law_id": LAW_ID, "article_id": "22"}],
        answer="A",
    )
    row.update(overrides)
    return row


def make_image(root: Path, image_id: str) -> None:
    path = root / f"{image_id}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake-jpeg")


def base_config(tmp_path: Path, public_rows=None, private_rows=None, task2_rows=None) -> dict:
    public_dir = tmp_path / "public_images"
    private_dir = tmp_path / "private images"
    public_task1 = tmp_path / "public_task1.json"
    public_task2 = tmp_path / "public_task2.json"
    private_task1 = tmp_path / "Task 1 Submission File" / "private_task1.json"
    private_task2 = tmp_path / "Task 2 Submission File" / "private_task2.json"

    public_rows = public_rows or [sample("public_test_1", "public_img_1")]
    task2_rows = task2_rows or [task2_sample("public_test_51", "public_img_2")]
    private_rows = private_rows or [sample("private_test_1", "private_img_1")]

    for row in public_rows:
        make_image(public_dir, row["image_id"])
    for row in task2_rows:
        make_image(public_dir, row["image_id"])
    for row in private_rows:
        make_image(private_dir, row["image_id"])

    write_json(public_rows, public_task1)
    write_json(task2_rows, public_task2)
    write_json(private_rows, private_task1)
    write_json([task2_sample("private_test_51", "private_img_2")], private_task2)
    make_image(private_dir, "private_img_2")

    return {
        "project": {"name": "traffic-legal-vlm", "seed": 42},
        "data": {
            "public_test_task1_path": str(public_task1),
            "public_test_task2_path": str(public_task2),
            "public_test_image_dir": str(public_dir),
            "private_test_task1_path": str(private_task1),
            "private_test_task2_path": str(private_task2),
            "private_test_image_dir": str(private_dir),
            "val_split_path": str(tmp_path / "unused_val.jsonl"),
            "train_split_path": str(tmp_path / "unused_train.jsonl"),
        },
        "experiment": {
            "name": "vlsp_test_unit",
            "label": "VLSP test unit",
            "mock": True,
            "retrieval_strategy": "text",
            "prompt_variant": "text_rag",
        },
        "retrieval": {"top_k": 1},
        "prompt": {"variant": "text_rag"},
        "model": {"backend": "none", "include_image": False},
    }


def fake_evidence(article_id: str = "22") -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id=article_id,
        title=f"Điều {article_id}",
        content=f"Nội dung Điều {article_id}.",
        score=0.9,
        rank=1,
        retrieval_method="text",
    )


def fake_retrieve_for_sample(sample, config, runtime):
    return [fake_evidence()], [], []


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_public_private_sample_loading_without_gold_labels(tmp_path):
    config = base_config(tmp_path)

    public_samples = load_vlsp_test_samples(config, "public_test", "task2")
    private_samples = load_vlsp_test_samples(config, "private_test", "task1")

    assert public_samples[0]["id"] == "public_test_51"
    assert Path(public_samples[0]["image_path"]).exists()
    assert "answer" not in public_samples[0]
    assert "relevant_articles" not in public_samples[0]
    assert private_samples[0]["id"] == "private_test_1"
    assert Path(private_samples[0]["image_path"]).exists()


def test_task1_runner_returns_citations_and_no_answer(monkeypatch, tmp_path):
    config = base_config(tmp_path)
    output_path = tmp_path / "task1_predictions.jsonl"
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)

    path = pipeline.run_vlsp_test(
        config,
        set_name="public_test",
        task="task1",
        limit=1,
        output_path=output_path,
        runtime=SimpleNamespace(vlm=FakeVLM()),
    )

    rows = read_jsonl(path)
    assert [row["id"] for row in rows] == ["public_test_1"]
    assert rows[0]["relevant_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert rows[0]["predicted_articles"] == [{"law_id": LAW_ID, "article_id": "22"}]
    assert "answer" not in rows[0]
    assert rows[0]["model"]["backend"] == "none"


def test_task2_mock_runner_returns_valid_answer_shape(monkeypatch, tmp_path):
    config = base_config(tmp_path)
    output_path = tmp_path / "task2_predictions.jsonl"
    fake_vlm = FakeVLM()
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)
    monkeypatch.setattr(pipeline, "retrieve_prompt_examples", lambda *args: [])

    path = pipeline.run_vlsp_test(
        config,
        set_name="public_test",
        task="task2",
        limit=1,
        output_path=output_path,
        runtime=SimpleNamespace(vlm=fake_vlm),
    )

    rows = read_jsonl(path)
    assert rows[0]["task"] == "task2"
    assert rows[0]["set_name"] == "public_test"
    assert rows[0]["mock"] is True
    assert rows[0]["prediction"]["answer"] == "A"
    assert rows[0]["prediction"]["citations"] == [{"law_id": LAW_ID, "article_id": "22", "title": "Điều 22", "quote": None}]
    assert rows[0]["parse"]["success"] is True
    assert fake_vlm.calls[-1]["query"] == "public_test_51"


def test_missing_image_path_error_message(tmp_path):
    config = base_config(tmp_path, public_rows=[sample("public_test_1", "missing_img")])
    Path(config["data"]["public_test_image_dir"], "missing_img.jpg").unlink()

    with pytest.raises(FileNotFoundError) as exc_info:
        load_vlsp_test_samples(config, "public_test", "task1")

    assert "VLSP test image not found for sample 'public_test_1'" in str(exc_info.value)


def test_missing_live_backend_credentials_fail_before_fake_answers(tmp_path):
    config = base_config(tmp_path)
    config["experiment"]["mock"] = False
    config["model"] = {
        "backend": "openai_compatible",
        "include_image": True,
        "api_key_env": "__TRAFFIC_LEGAL_VLM_MISSING_TEST_KEY__",
        "base_url_env": "__TRAFFIC_LEGAL_VLM_MISSING_TEST_URL__",
    }
    output_path = tmp_path / "should_not_exist.jsonl"

    with pytest.raises(RuntimeError) as exc_info:
        pipeline.run_vlsp_test(
            config,
            set_name="public_test",
            task="task2",
            limit=1,
            output_path=output_path,
        )

    assert "Missing VLM backend configuration" in str(exc_info.value)
    assert not output_path.exists()


def test_output_row_ids_match_input_order(monkeypatch, tmp_path):
    rows = [
        sample("public_test_2", "public_img_2"),
        sample("public_test_1", "public_img_1"),
    ]
    config = base_config(tmp_path, public_rows=rows)
    output_path = tmp_path / "ordered_task1_predictions.jsonl"
    monkeypatch.setattr(pipeline, "retrieve_for_sample", fake_retrieve_for_sample)

    path = pipeline.run_vlsp_test(
        config,
        set_name="public_test",
        task="task1",
        output_path=output_path,
        runtime=SimpleNamespace(vlm=FakeVLM()),
    )

    assert [row["id"] for row in read_jsonl(path)] == [
        "public_test_2",
        "public_test_1",
    ]
