import json

import pytest
from pydantic import ValidationError

from src.prompts import PromptConfig, build_legal_qa_prompt
from src.schemas import Evidence, Prediction, Query
from src.vlm import LegalQAVLM, parse_prediction


LAW_ID = "QCVN 41:2024/BGTVT"


def multiple_choice_query(**overrides) -> Query:
    data = {
        "id": "train_1",
        "image_id": "train_1_3",
        "image_path": "data/raw/train_data/train_images/train_1_3.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
    }
    data.update(overrides)
    return Query.model_validate(data)


def yes_no_query(**overrides) -> Query:
    data = {
        "id": "train_2",
        "image_id": "train_2_1",
        "question": "Phương tiện này được phép đi thẳng, đúng hay sai?",
        "question_type": "Yes/No",
    }
    data.update(overrides)
    return Query.model_validate(data)


def evidence(article_id: str = "22") -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id=article_id,
        title="Ý nghĩa sử dụng biển báo",
        content="Nội dung điều luật về biển báo giao thông.",
        score=0.91,
        rank=1,
        retrieval_method="text",
    )


def raw_response(**overrides) -> str:
    payload = {
        "answer": "B",
        "citations": [{"law_id": LAW_ID, "article_id": "22"}],
        "explanation": "Biển báo phù hợp với Điều 22.",
        "confidence": 0.72,
        "abstained": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_prompt_contains_image_placeholder_question_choices_and_evidence():
    prompt = build_legal_qa_prompt(
        multiple_choice_query(),
        [evidence()],
        prompt_config=PromptConfig(image_placeholder="<IMAGE_0>", max_evidence_chars=80),
    )

    assert "<IMAGE_0>" in prompt
    assert "Biển báo này có ý nghĩa gì?" in prompt
    assert "A. Một" in prompt
    assert "D. Bốn" in prompt
    assert f"uid={LAW_ID}#22" in prompt
    assert "answer, citations, explanation, confidence, abstained" in prompt


def test_valid_json_parses_to_prediction():
    prediction = parse_prediction(raw_response(), multiple_choice_query(), [evidence()])

    assert isinstance(prediction, Prediction)
    assert prediction.answer == "B"
    assert prediction.citations[0].uid == f"{LAW_ID}#22"
    assert prediction.confidence == 0.72


def test_markdown_fenced_json_parses_correctly():
    raw = (
        "```json\n"
        + raw_response(
            answer="Đúng",
            citations=[{"law_id": LAW_ID, "article_id": "22"}],
            explanation="Điều 22 hỗ trợ kết luận đúng.",
        )
        + "\n```"
    )

    prediction = parse_prediction(raw, yes_no_query(), [evidence()])

    assert prediction.answer == "Đúng"


def test_hallucinated_citation_fails_pipeline_validation():
    with pytest.raises(ValidationError, match="outside retrieved evidence"):
        parse_prediction(
            raw_response(citations=[{"law_id": LAW_ID, "article_id": "B.27"}]),
            multiple_choice_query(),
            [evidence("22")],
        )


def test_invalid_multiple_choice_answer_fails():
    with pytest.raises(ValueError, match="A, B, C or D"):
        parse_prediction(raw_response(answer="E"), multiple_choice_query(), [evidence()])


def test_free_form_prompt_and_prediction_are_supported():
    query = Query(
        id="demo_1",
        image_id="upload_1",
        question="Tôi có được đỗ xe ở đây vào cuối tuần không?",
        question_type="Free-form",
    )
    raw = raw_response(answer="Không đủ căn cứ để kết luận.")

    prediction = parse_prediction(raw, query, [evidence()])
    prompt = build_legal_qa_prompt(query, [evidence()])

    assert prediction.answer == "Không đủ căn cứ để kết luận."
    assert "free-form" in prompt.lower()
    assert "answer in Vietnamese" in prompt
    assert "abstained=true" in prompt
    assert "official legal advice" in prompt


def test_abstained_prediction_can_omit_citations():
    prediction = parse_prediction(
        raw_response(
            answer="A",
            citations=[],
            explanation="Không đủ căn cứ pháp lý trong evidence đã truy xuất.",
            confidence=0.0,
            abstained=True,
        ),
        multiple_choice_query(),
        [evidence()],
    )

    assert prediction.abstained is True
    assert prediction.citations == []


class FakeClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = []

    def generate(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return self.response


def test_runtime_wrapper_is_configurable_and_mockable():
    client = FakeClient(raw_response())
    runtime = LegalQAVLM(
        {
            "model": {
                "name": "mock-vlm",
                "temperature": 0.0,
                "max_new_tokens": 128,
            },
            "prompt": {
                "image_placeholder": "<MOCK_IMAGE>",
                "max_evidence_chars": 64,
            },
        },
        client=client,
    )

    prediction = runtime.answer(multiple_choice_query(), [evidence()])

    assert prediction.answer == "B"
    assert client.calls[0]["model_name"] == "mock-vlm"
    assert "<MOCK_IMAGE>" in client.calls[0]["messages"][0]["content"][0]["text"]
