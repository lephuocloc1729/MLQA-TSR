import json

import pytest
from PIL import Image

from src.evaluate import build_evaluation_artifact
from src.schemas import Evidence, Query
from src.vlm import LegalQAVLM, chat_completions_endpoint, create_vlm_client


LAW_ID = "QCVN 41:2024/BGTVT"


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.response


def tiny_image(path):
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(path)


def query(image_path=None) -> Query:
    return Query(
        id="val_1",
        image_id="img_1",
        image_path=str(image_path) if image_path else None,
        question="Biển báo này có ý nghĩa gì?",
        question_type="Multiple choice",
        choices={"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        answer="B",
    )


def evidence() -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id="22",
        title="Điều 22",
        content="Nội dung pháp lý của Điều 22.",
        score=0.9,
        rank=1,
        retrieval_method="text",
    )


def response(**overrides) -> str:
    payload = {
        "answer": "B",
        "citations": [{"law_id": LAW_ID, "article_id": "22"}],
        "explanation": "Dựa trên Điều 22.",
        "confidence": 0.75,
        "abstained": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_fake_client_success_with_image_part(tmp_path):
    image_path = tmp_path / "traffic.jpg"
    tiny_image(image_path)
    client = FakeClient(response())
    runtime = LegalQAVLM(
        {
            "model": {
                "backend": "openai_compatible",
                "name": "mock-vlm",
                "include_image": True,
                "temperature": 0.0,
            }
        },
        client=client,
    )

    prediction = runtime.answer(query(image_path), [evidence()])

    assert prediction.answer == "B"
    assert prediction.raw_response == response()
    call = client.calls[0]
    assert call["model_name"] == "mock-vlm"
    assert call["temperature"] == 0.0
    assert call["messages"][0]["content"][-1]["type"] == "image_url"
    assert call["messages"][0]["content"][-1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )


def test_fake_client_invalid_json_still_uses_parser_validation():
    runtime = LegalQAVLM({"model": {"name": "mock-vlm"}}, client=FakeClient("not json"))

    with pytest.raises(ValueError, match="JSON object"):
        runtime.answer(query(), [evidence()])


def test_missing_openai_compatible_credentials_error_is_helpful():
    with pytest.raises(RuntimeError, match="OPENAI_COMPATIBLE_API_KEY"):
        create_vlm_client(
            {
                "backend": "openai_compatible",
                "base_url": "http://localhost:8000/v1",
                "api_key_env": "OPENAI_COMPATIBLE_API_KEY",
            },
            environ={},
        )


def test_missing_openai_compatible_base_url_error_is_helpful():
    with pytest.raises(RuntimeError, match="OPENAI_COMPATIBLE_BASE_URL"):
        create_vlm_client(
            {
                "backend": "openai_compatible",
                "api_key_env": "KEY_ENV",
                "base_url_env": "OPENAI_COMPATIBLE_BASE_URL",
            },
            environ={"KEY_ENV": "secret"},
        )


def test_chat_completions_endpoint_normalization():
    assert (
        chat_completions_endpoint("http://localhost:8000/v1")
        == "http://localhost:8000/v1/chat/completions"
    )
    assert (
        chat_completions_endpoint("http://localhost:8000/v1/chat/completions")
        == "http://localhost:8000/v1/chat/completions"
    )


def test_metrics_artifact_marks_non_mock_run():
    records = [
        {
            "query": {
                "id": "val_1",
                "image_id": "img_1",
                "question": "Chọn đáp án đúng?",
                "question_type": "Multiple choice",
                "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
                "answer": "B",
                "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
            },
            "prediction": {
                "answer": "B",
                "citations": [{"law_id": LAW_ID, "article_id": "22"}],
                "explanation": "Dựa trên Điều 22.",
                "confidence": 0.75,
                "abstained": False,
                "raw_response": response(),
            },
            "predicted_articles": [{"law_id": LAW_ID, "article_id": "22"}],
            "experiment": {
                "name": "w3_b2_text_rag_real",
                "label": "W3_B2_text_rag_real",
                "mock": False,
                "retrieval_strategy": "text",
                "prompt_variant": "text_rag",
            },
        }
    ]

    artifact = build_evaluation_artifact(records)

    assert artifact["config_name"] == "w3_b2_text_rag_real"
    assert artifact["mock"] is False
    assert artifact["experiment"]["mock"] is False
    assert artifact["qa"]["accuracy"] == 1.0
