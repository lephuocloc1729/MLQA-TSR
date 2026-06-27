import json

import pytest

from src.prompts import (
    PromptConfig,
    PromptVariant,
    build_legal_qa_prompt,
    build_vlm_messages,
)
from src.schemas import Evidence, Query
from src.vlm import LegalQAVLM, parse_prediction


LAW_ID = "QCVN 41:2024/BGTVT"


def multiple_choice_query(**overrides) -> Query:
    data = {
        "id": "val_1",
        "image_id": "val_img_1",
        "image_path": "data/raw/train_data/train_images/val_img_1.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "D",
    }
    data.update(overrides)
    return Query.model_validate(data)


def yes_no_query(**overrides) -> Query:
    data = {
        "id": "val_2",
        "image_id": "val_img_2",
        "question": "Xe máy được phép đi vào đường này, đúng hay sai?",
        "question_type": "Yes/No",
        "answer": "Sai",
    }
    data.update(overrides)
    return Query.model_validate(data)


def free_form_query(**overrides) -> Query:
    data = {
        "id": "val_3",
        "image_id": "val_img_3",
        "question": "Tôi có được đỗ xe ở đây không?",
        "question_type": "Free-form",
        "answer": "GOLD_SECRET_DO_NOT_LEAK",
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
        retrieval_method="fusion",
    )


def example(
    sample_id: str,
    answer: str = "A",
    question_type: str = "Multiple choice",
    image_id: str | None = None,
    split: str = "train",
) -> dict:
    payload = {
        "sample_id": sample_id,
        "image_id": image_id or f"img_{sample_id}",
        "image_path": f"data/raw/train_data/train_images/{image_id or sample_id}.jpg",
        "question": "Câu hỏi mẫu đã giải?",
        "question_type": question_type,
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"}
        if question_type == "Multiple choice"
        else {},
        "answer": answer,
        "relevant_articles": [{"law_id": LAW_ID, "article_id": "22"}],
        "split": split,
    }
    return {"payload": payload, "score": 0.88, "rank": 1, "retrieval_mode": "fusion"}


def prompt_config(**overrides) -> PromptConfig:
    data = {
        "image_placeholder": "<QUERY_IMAGE>",
        "example_image_placeholder": "<EXAMPLE_IMAGE_{index}>",
        "max_evidence_chars": 120,
        "max_example_chars": 120,
        "top_examples": 3,
    }
    data.update(overrides)
    return PromptConfig(**data)


def test_few_shot_prompt_for_multiple_choice_contains_query_evidence_examples_and_json():
    prompt = build_legal_qa_prompt(
        multiple_choice_query(),
        [evidence()],
        examples=[example("train_1", answer="B")],
        variant=PromptVariant.FEW_SHOT_RAG,
        prompt_config=prompt_config(),
    )

    assert "Prompt variant:\nfew_shot_rag" in prompt
    assert "<QUERY_IMAGE>" in prompt
    assert "Biển báo này có ý nghĩa gì?" in prompt
    assert "A. Một" in prompt
    assert f"uid={LAW_ID}#22" in prompt
    assert "Retrieved solved training examples" in prompt
    assert "Example 1:" in prompt
    assert "<EXAMPLE_IMAGE_1>" in prompt
    assert "Gold answer:\nB" in prompt
    assert f"Relevant citations:\n{LAW_ID}#22" in prompt
    assert "answer, citations, explanation, confidence, abstained" in prompt


def test_few_shot_prompt_for_yes_no_uses_yes_no_answer_format():
    prompt = build_legal_qa_prompt(
        yes_no_query(),
        [evidence("30")],
        examples=[
            example(
                "train_yes_no",
                answer="Đúng",
                question_type="Yes/No",
            )
        ],
        variant="few_shot_rag",
        prompt_config=prompt_config(),
    )

    assert "Question type:\nYes/No" in prompt
    assert "Đúng\nSai" in prompt
    assert "Gold answer:\nĐúng" in prompt
    assert "answer must be exactly one of Đúng or Sai" in prompt


def test_few_shot_prompt_omits_examples_when_none_are_retrieved():
    prompt = build_legal_qa_prompt(
        multiple_choice_query(),
        [evidence()],
        examples=[],
        variant="few_shot_rag",
        prompt_config=prompt_config(),
    )

    assert "(no retrieved training examples)" in prompt
    assert "Example 1:" not in prompt


def test_few_shot_prompt_limits_examples_to_top_three_by_default():
    prompt = build_legal_qa_prompt(
        multiple_choice_query(),
        [evidence()],
        examples=[
            example("train_1", answer="A"),
            example("train_2", answer="B"),
            example("train_3", answer="C"),
            example("train_4", answer="D"),
        ],
        variant="few_shot_rag",
        prompt_config=prompt_config(),
    )

    assert "train_1" in prompt
    assert "train_2" in prompt
    assert "train_3" in prompt
    assert "train_4" not in prompt
    assert prompt.count("Example ") == 3


def test_validation_gold_answer_is_not_included_in_prompt():
    prompt = build_legal_qa_prompt(
        free_form_query(),
        [evidence()],
        examples=[example("train_1", answer="Không")],
        variant="few_shot_rag",
        prompt_config=prompt_config(),
    )

    assert "GOLD_SECRET_DO_NOT_LEAK" not in prompt
    assert "Gold answer:\nKhông" in prompt


def test_validation_examples_are_rejected_to_prevent_answer_leakage():
    with pytest.raises(ValueError, match="split='train'"):
        build_legal_qa_prompt(
            multiple_choice_query(),
            [evidence()],
            examples=[example("val_1", answer="D", split="val")],
            variant="few_shot_rag",
            prompt_config=prompt_config(),
        )


def test_runtime_wrapper_builds_few_shot_messages_without_model_weights():
    runtime = LegalQAVLM(
        {
            "model": {"name": "mock-vlm"},
            "prompt": {
                "variant": "few_shot_rag",
                "image_placeholder": "<QUERY_IMAGE>",
                "top_examples": 3,
            },
        }
    )

    messages = runtime.build_messages(
        multiple_choice_query(),
        [evidence()],
        examples=[example("train_1", answer="B")],
    )

    prompt = messages[0]["content"][0]["text"]
    assert "Prompt variant:\nfew_shot_rag" in prompt
    assert "Gold answer:\nB" in prompt


def test_vlm_messages_accept_named_zero_shot_and_text_rag_variants():
    zero_shot_prompt = build_vlm_messages(
        multiple_choice_query(),
        [],
        variant="zero_shot",
        prompt_config=prompt_config(),
    )[0]["content"][0]["text"]
    text_rag_prompt = build_vlm_messages(
        multiple_choice_query(),
        [evidence()],
        variant="text_rag",
        prompt_config=prompt_config(),
    )[0]["content"][0]["text"]

    assert "Prompt variant:\nzero_shot" in zero_shot_prompt
    assert "(not provided for zero-shot variant)" in zero_shot_prompt
    assert "Prompt variant:\ntext_rag" in text_rag_prompt
    assert "Retrieved solved training examples" not in text_rag_prompt


def test_parser_still_handles_markdown_fenced_json_with_few_shot_evidence():
    payload = {
        "answer": "B",
        "citations": [{"law_id": LAW_ID, "article_id": "22"}],
        "explanation": "Dựa trên evidence đã truy xuất.",
        "confidence": 0.71,
        "abstained": False,
    }
    raw = "```json\n" + json.dumps(payload, ensure_ascii=False) + "\n```"

    prediction = parse_prediction(raw, multiple_choice_query(), [evidence()])

    assert prediction.answer == "B"
    assert prediction.citations[0].uid == f"{LAW_ID}#22"
