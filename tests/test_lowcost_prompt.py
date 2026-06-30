import json

import pytest

from src.prompts import (
    PromptConfig,
    PromptVariant,
    build_lowcost_answer_only_messages,
    build_vlm_messages,
    normalize_prompt_variant,
)
from src.schemas import Evidence, Query
from src.vlm import parse_answer_only_prediction


LAW_ID = "QCVN 41:2024/BGTVT"


def multiple_choice_query(**overrides) -> Query:
    data = {
        "id": "public_test_1",
        "image_id": "public_test_1_1",
        "image_path": "data/raw/public_test/images/public_test_1_1.jpg",
        "question": "Đây là biển báo gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
    }
    data.update(overrides)
    return Query.model_validate(data)


def yes_no_query(**overrides) -> Query:
    data = {
        "id": "public_test_2",
        "image_id": "public_test_2_1",
        "image_path": "data/raw/public_test/images/public_test_2_1.jpg",
        "question": "Xe này được phép đi thẳng, đúng hay sai?",
        "question_type": "Yes/No",
    }
    data.update(overrides)
    return Query.model_validate(data)


def train_example(**overrides) -> dict:
    data = {
        "sample_id": "train_1",
        "image_id": "train_1_1",
        "image_path": "data/raw/train_data/train_images/train_1_1.jpg",
        "question": "Biển báo này có ý nghĩa gì?",
        "question_type": "Multiple choice",
        "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        "answer": "B",
        "split": "train",
    }
    data.update(overrides)
    return data


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


def image_url_builder(path: str) -> str:
    return "data:image/jpeg;base64," + path.replace("/", "_")


def text_parts(messages: list[dict]) -> list[str]:
    parts: list[str] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
            continue
        parts.extend(part["text"] for part in content if part.get("type") == "text")
    return parts


def image_parts(messages: list[dict]) -> list[dict]:
    parts: list[dict] = []
    for message in messages:
        content = message["content"]
        if isinstance(content, list):
            parts.extend(part for part in content if part.get("type") == "image_url")
    return parts


def test_multiple_choice_lowcost_message_construction():
    messages = build_lowcost_answer_only_messages(
        multiple_choice_query(),
        examples=[train_example()],
        prompt_config=PromptConfig(top_examples=3),
        image_url_builder=image_url_builder,
    )

    joined_text = "\n".join(text_parts(messages))
    assert messages[0]["role"] == "system"
    assert "choose the letter only" in messages[0]["content"]
    assert "Few-shot example 1:" in joined_text
    assert "Question: Biển báo này có ý nghĩa gì?" in joined_text
    assert "A: Một" in joined_text
    assert "Choice: B" in joined_text
    assert "Query:" in joined_text
    assert "Question: Đây là biển báo gì?" in joined_text
    assert joined_text.rstrip().endswith("Choice:")


def test_yes_no_lowcost_message_construction():
    messages = build_lowcost_answer_only_messages(
        yes_no_query(),
        examples=[
            train_example(
                sample_id="train_2",
                question_type="Yes/No",
                choices={},
                answer="Đúng",
            )
        ],
        image_url_builder=image_url_builder,
    )

    joined_text = "\n".join(text_parts(messages))
    assert "Đúng\nSai" in joined_text
    assert "Choice: Đúng" in joined_text
    assert joined_text.rstrip().endswith("Choice:")


def test_example_images_are_included_as_image_url_parts():
    messages = build_lowcost_answer_only_messages(
        multiple_choice_query(),
        examples=[train_example()],
        image_url_builder=image_url_builder,
    )

    parts = image_parts(messages)
    assert len(parts) == 2
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "train_images" in parts[0]["image_url"]["url"]
    assert "public_test" in parts[1]["image_url"]["url"]


def test_validation_answer_leakage_is_rejected():
    with pytest.raises(ValueError, match="split='train'"):
        build_lowcost_answer_only_messages(
            multiple_choice_query(),
            examples=[train_example(split="val")],
        )


def test_same_image_example_is_skipped_before_prompting():
    messages = build_lowcost_answer_only_messages(
        multiple_choice_query(image_id="train_1_1"),
        examples=[
            train_example(sample_id="train_same", image_id="train_1_1", answer="A"),
            train_example(sample_id="train_keep", image_id="train_2_1", answer="C"),
        ],
    )

    joined_text = "\n".join(text_parts(messages))
    assert "Choice: A" not in joined_text
    assert "Choice: C" in joined_text


def test_build_vlm_messages_routes_lowcost_variant():
    messages = build_vlm_messages(
        multiple_choice_query(),
        evidence=[],
        examples=[train_example()],
        variant=PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT,
        image_url_builder=image_url_builder,
    )

    assert messages[0]["role"] == "system"
    assert len(image_parts(messages)) == 2


def test_prompt_variant_normalization_accepts_lowcost_answer_only():
    assert (
        normalize_prompt_variant("lowcost_answer_only_fewshot")
        == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT
    )


def test_answer_only_parser_handles_raw_json_and_xml_labels():
    query = multiple_choice_query()
    assert parse_answer_only_prediction("B", query, [evidence()]).answer == "B"
    assert (
        parse_answer_only_prediction(
            json.dumps({"choice": "B"}, ensure_ascii=False),
            query,
            [evidence()],
        ).answer
        == "B"
    )
    assert (
        parse_answer_only_prediction(
            "<answer>Đúng</answer>",
            yes_no_query(),
            [evidence()],
        ).answer
        == "Đúng"
    )


def test_answer_only_parser_normalizes_yes_no_aliases():
    prediction = parse_answer_only_prediction("Yes", yes_no_query(), [evidence()])

    assert prediction.answer == "Đúng"


def test_answer_only_parser_rejects_invalid_label():
    with pytest.raises(ValueError, match="A, B, C or D"):
        parse_answer_only_prediction("E", multiple_choice_query(), [evidence()])
