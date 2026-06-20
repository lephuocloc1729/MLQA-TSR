import pytest
from pydantic import ValidationError

from src.schemas import (
    Citation,
    DetectedSign,
    Evidence,
    PipelineResult,
    Prediction,
    Query,
    QuestionType,
    RetrievalMethod,
)


LAW_ID = "QCVN 41:2024/BGTVT"


def multiple_choice_query(**overrides):
    data = {
        "id": "train_1",
        "image_id": "train_1_3",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
    }
    data.update(overrides)
    return Query.model_validate(data)


def test_query_matches_vlsp_training_schema():
    query = multiple_choice_query()

    assert query.question_type == QuestionType.MULTIPLE_CHOICE
    assert query.answer == "B"
    assert query.relevant_articles[0].uid == f"{LAW_ID}#22"


def test_task_1_query_can_omit_question_type_and_choices():
    query = Query(
        id="public_test_1",
        image_id="public_test_2_2",
        question="Đây là biển báo gì?",
    )

    assert query.question_type is None
    assert query.choices == {}


@pytest.mark.parametrize(
    ("raw_answer", "expected"),
    [("Đúng", "Đúng"), ("Yes", "Đúng"), ("No", "Sai"), (40, "A")],
)
def test_answer_normalization(raw_answer, expected):
    if expected == "A":
        query = multiple_choice_query(answer=raw_answer)
    else:
        query = Query(
            id="yes_no_1",
            image_id="image_1",
            question="Phát biểu này đúng hay sai?",
            question_type="Yes/No",
            answer=raw_answer,
        )

    assert query.answer == expected


def test_multiple_choice_requires_all_four_choices():
    with pytest.raises(ValidationError, match="choices A, B, C and D"):
        multiple_choice_query(choices={"A": "Một", "B": "Hai"})


def test_evidence_is_compatible_with_flat_qdrant_payload():
    evidence = Evidence(
        law_id=LAW_ID,
        article_id="B.27",
        title="Biển báo tốc độ",
        content="Quy định về tốc độ tối đa cho phép.",
        score=0.91,
        rank=1,
        retrieval_method="fusion",
        sign_name="P.127 Tốc độ tối đa cho phép",
    )

    assert evidence.retrieval_method == RetrievalMethod.FUSION
    assert evidence.to_citation().to_vlsp_reference() == {
        "law_id": LAW_ID,
        "article_id": "B.27",
    }


def test_detected_sign_accepts_reference_project_bbox_format():
    sign = DetectedSign(
        image_name="train_1_crop0.jpg",
        bbox=[10, 20, 110, 220],
        confidence=0.95,
        is_chosen=True,
    )

    assert sign.bbox is not None
    assert sign.bbox.x_max == 110
    assert sign.model_dump()["bbox"] == (10.0, 20.0, 110.0, 220.0)


def test_non_abstained_prediction_requires_citation():
    with pytest.raises(ValidationError, match="requires at least one citation"):
        Prediction(
            question_type="Multiple choice",
            answer="A",
            explanation="Giải thích.",
        )


def test_pipeline_rejects_hallucinated_citation():
    query = multiple_choice_query()
    evidence = Evidence(
        law_id=LAW_ID,
        article_id="22",
        content="Nội dung điều 22.",
        rank=1,
        retrieval_method="text",
    )
    prediction = Prediction(
        id=query.id,
        question_type=query.question_type,
        answer="B",
        citations=[Citation(law_id=LAW_ID, article_id="B.27")],
        explanation="Câu trả lời dựa trên điều luật.",
    )

    with pytest.raises(ValidationError, match="outside retrieved evidence"):
        PipelineResult(query=query, evidence=[evidence], prediction=prediction)


def test_abstained_pipeline_still_rejects_hallucinated_citation():
    query = Query(
        id="free_form_1",
        image_id="upload_1",
        question="Tôi có được đỗ xe tại đây không?",
        question_type="Free-form",
    )
    prediction = Prediction(
        id=query.id,
        question_type=query.question_type,
        answer="Không đủ căn cứ để kết luận.",
        citations=[Citation(law_id=LAW_ID, article_id="22")],
        explanation="Không tìm thấy bằng chứng phù hợp.",
        abstained=True,
        disclaimer="Thông tin chỉ mang tính tham khảo.",
    )

    with pytest.raises(ValidationError, match="outside retrieved evidence"):
        PipelineResult(query=query, evidence=[], prediction=prediction)


def test_valid_pipeline_result_round_trip():
    query = multiple_choice_query()
    evidence = Evidence(
        law_id=LAW_ID,
        article_id="22",
        content="Nội dung điều 22.",
        score=0.85,
        rank=1,
        retrieval_method="text",
    )
    prediction = Prediction(
        id=query.id,
        question_type=query.question_type,
        answer="B",
        citations=[evidence.to_citation(quote="Nội dung điều 22.")],
        explanation="Biển báo và điều 22 hỗ trợ đáp án B.",
        confidence=0.8,
    )

    result = PipelineResult(
        query=query,
        evidence=[evidence],
        prediction=prediction,
        timings_ms={"retrieval": 12.5, "generation": 130.0},
    )
    restored = PipelineResult.model_validate_json(result.model_dump_json())

    assert restored == result
