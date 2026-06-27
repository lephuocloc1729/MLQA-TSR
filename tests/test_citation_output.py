import json

import pytest
from pydantic import ValidationError

from src.prompts import PromptVariant, build_legal_qa_prompt, normalize_prompt_variant
from src.schemas import Evidence, Prediction, Query
from src.vlm import parse_prediction


LAW_ID = "QCVN 41:2024/BGTVT"


def multiple_choice_query(**overrides) -> Query:
    data = {
        "id": "val_1",
        "image_id": "img_1",
        "image_path": "data/raw/train_data/train_images/img_1.jpg",
        "question": "Biển báo cấm xe khách áp dụng trong khung giờ nào?",
        "question_type": "Multiple choice",
        "choices": {
            "A": "Cả ngày",
            "B": "Theo khung giờ ghi trên biển",
            "C": "Không cấm",
            "D": "Chỉ ban đêm",
        },
        "answer": "B",
    }
    data.update(overrides)
    return Query.model_validate(data)


def yes_no_query(**overrides) -> Query:
    data = {
        "id": "val_2",
        "image_id": "img_2",
        "question": "Xe máy được phép đi vào đường này, đúng hay sai?",
        "question_type": "Yes/No",
        "answer": "Sai",
    }
    data.update(overrides)
    return Query.model_validate(data)


def evidence(article_id: str = "22") -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id=article_id,
        title=f"Điều {article_id}",
        content=f"Nội dung pháp lý của Điều {article_id}.",
        score=0.91,
        rank=1,
        retrieval_method="text",
    )


def structured_response(**overrides) -> str:
    payload = {
        "answer": "B",
        "citations": [{"law_id": LAW_ID, "article_id": "22"}],
        "explanation": (
            "Observation: Câu hỏi đề cập biển cấm có khung giờ. "
            "Legal basis: Điều 22 nêu hiệu lực của biển phụ theo thời gian. "
            "Conclusion: Phương án B phù hợp với khung giờ ghi trên biển."
        ),
        "confidence": 0.76,
        "abstained": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


def test_structured_prompt_includes_observation_legal_basis_conclusion():
    prompt = build_legal_qa_prompt(
        multiple_choice_query(),
        [evidence()],
        variant=PromptVariant.STRUCTURED_LEGAL_RAG,
    )

    assert "Prompt variant:\nstructured_legal_rag" in prompt
    assert "Structured explanation requirements" in prompt
    assert "Observation:" in prompt
    assert "Legal basis:" in prompt
    assert "Conclusion:" in prompt
    assert "Do not expose hidden chain-of-thought" in prompt
    assert "answer, citations, explanation, confidence, abstained" in prompt
    assert f"uid={LAW_ID}#22" in prompt


def test_structured_prompt_for_yes_no_keeps_yes_no_answer_constraint():
    prompt = build_legal_qa_prompt(
        yes_no_query(),
        [evidence("32")],
        variant="structured_legal_rag",
    )

    assert "Question type:\nYes/No" in prompt
    assert "Đúng\nSai" in prompt
    assert "answer must be exactly one of Đúng or Sai" in prompt
    assert "Observation:" in prompt
    assert "Legal basis:" in prompt
    assert "Conclusion:" in prompt


def test_valid_structured_explanation_parses_to_prediction():
    prediction = parse_prediction(
        structured_response(),
        multiple_choice_query(),
        [evidence()],
    )

    assert isinstance(prediction, Prediction)
    assert prediction.answer == "B"
    assert "Observation:" in prediction.explanation
    assert "Legal basis:" in prediction.explanation
    assert "Conclusion:" in prediction.explanation
    assert prediction.citations[0].uid == f"{LAW_ID}#22"


def test_citation_outside_evidence_fails_validation():
    with pytest.raises(ValidationError, match="outside retrieved evidence"):
        parse_prediction(
            structured_response(
                citations=[{"law_id": LAW_ID, "article_id": "999"}],
            ),
            multiple_choice_query(),
            [evidence("22")],
        )


def test_structured_abstention_response_remains_valid():
    prediction = parse_prediction(
        structured_response(
            answer="B",
            citations=[],
            explanation=(
                "Observation: Hình ảnh không đủ rõ để nhận diện biển. "
                "Legal basis: Không có điều luật được truy xuất hỗ trợ kết luận. "
                "Conclusion: Cần abstain vì evidence không đủ."
            ),
            confidence=0.0,
            abstained=True,
        ),
        multiple_choice_query(),
        [evidence()],
    )

    assert prediction.abstained is True
    assert prediction.citations == []
    assert "Conclusion:" in prediction.explanation


def test_prompt_variant_config_normalization():
    assert (
        normalize_prompt_variant("structured_legal_rag")
        == PromptVariant.STRUCTURED_LEGAL_RAG
    )

    with pytest.raises(ValueError, match="structured_legal_rag"):
        normalize_prompt_variant("structured-law")
