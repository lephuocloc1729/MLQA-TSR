from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping

from src.schemas import Citation, Evidence, Query, QuestionType


LEGAL_QA_SYSTEM_PROMPT = """You are a Vietnamese traffic-law QA assistant.
Use the image, question, answer choices, retrieved legal evidence, and solved
training examples only.
Return only one JSON object. Do not include markdown, XML tags, or hidden
chain-of-thought. The explanation must be short and cite the evidence used."""


class PromptVariant(str, Enum):
    ZERO_SHOT = "zero_shot"
    TEXT_RAG = "text_rag"
    FEW_SHOT_RAG = "few_shot_rag"
    STRUCTURED_LEGAL_RAG = "structured_legal_rag"
    LOWCOST_ANSWER_ONLY_FEWSHOT = "lowcost_answer_only_fewshot"


@dataclass(frozen=True)
class PromptConfig:
    image_placeholder: str = "<IMAGE>"
    example_image_placeholder: str = "<EXAMPLE_IMAGE_{index}>"
    max_evidence_chars: int = 1200
    max_example_chars: int = 700
    top_examples: int = 3


def normalize_prompt_variant(variant: PromptVariant | str | None) -> PromptVariant:
    if variant is None:
        return PromptVariant.TEXT_RAG
    if isinstance(variant, PromptVariant):
        return variant
    try:
        return PromptVariant(str(variant))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in PromptVariant)
        raise ValueError(f"prompt variant must be one of: {allowed}") from exc


def resolve_question_type(query: Query) -> QuestionType:
    if query.question_type is not None:
        return query.question_type
    if query.choices:
        return QuestionType.MULTIPLE_CHOICE
    return QuestionType.FREE_FORM


def format_choices(query: Query) -> str:
    question_type = resolve_question_type(query)
    if question_type == QuestionType.MULTIPLE_CHOICE:
        return "\n".join(f"{key}. {query.choices[key]}" for key in sorted(query.choices))
    if question_type == QuestionType.YES_NO:
        return "Đúng\nSai"
    return "(free-form answer; no fixed choices)"


def truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def format_evidence(evidence: Iterable[Evidence], max_chars: int) -> str:
    blocks: list[str] = []
    for item in evidence:
        score = f", score={item.score:.4f}" if item.score is not None else ""
        title = item.title or "(untitled article)"
        blocks.append(
            "\n".join(
                [
                    f"[{item.rank or len(blocks) + 1}] uid={item.uid}{score}",
                    f"law_id={item.law_id}",
                    f"article_id={item.article_id}",
                    f"title={title}",
                    "content=" + truncate_text(item.content, max_chars),
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "(no retrieved legal evidence)"


def _example_payload(example: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = example.get("payload")
    if isinstance(payload, Mapping):
        return payload
    return example


def _format_example_image_placeholder(config: PromptConfig, index: int) -> str:
    return config.example_image_placeholder.format(index=index)


def format_example_citations(references: Iterable[Mapping[str, Any]]) -> str:
    citations: list[str] = []
    for reference in references:
        citation = Citation.model_validate(reference)
        citations.append(citation.uid)
    return ", ".join(citations) if citations else "(no relevant citations)"


def format_example_choices(example: Mapping[str, Any]) -> str:
    choices = example.get("choices") or {}
    if choices:
        return "\n".join(
            f"{key}. {choices[key]}" for key in sorted(choices)
        )
    if example.get("question_type") == QuestionType.YES_NO.value:
        return "Đúng\nSai"
    return "(no fixed choices)"


def lowcost_answer_only_system_prompt() -> str:
    return "\n".join(
        [
            "Given an image and a question both about traffic in Vietnam,",
            "Multiple-choice and yes/no questions may be provided.",
            "If A, B, C, D are given, choose the letter only.",
            "If Đúng/Sai are given, choose Đúng or Sai only.",
            "No explanation is needed.",
        ]
    )


def format_lowcost_answer_text(
    item: Query | Mapping[str, Any],
    include_answer: bool,
) -> str:
    query = item if isinstance(item, Query) else None
    question = query.question if query is not None else str(item.get("question") or "")
    question_type = (
        query.question_type.value
        if query is not None and query.question_type
        else str(item.get("question_type") or "")
    )
    choices = query.choices if query is not None else dict(item.get("choices") or {})
    answer = query.answer if query is not None else item.get("answer")

    lines = ["Question: " + question, "Options:"]
    if question_type == QuestionType.MULTIPLE_CHOICE.value:
        lines.extend(f"{key}: {choices[key]}" for key in sorted(choices))
    elif question_type == QuestionType.YES_NO.value:
        lines.extend(["Đúng", "Sai"])
    else:
        lines.append("(free-form answer)")

    choice_line = "Choice:"
    if include_answer and answer not in (None, ""):
        choice_line += " " + str(answer)
    lines.append(choice_line)
    return "\n".join(lines)


def _item_id(item: Mapping[str, Any]) -> str | None:
    value = item.get("sample_id") or item.get("id")
    return None if value in (None, "") else str(value)


def _item_image_id(item: Mapping[str, Any]) -> str | None:
    value = item.get("image_id")
    return None if value in (None, "") else str(value)


def _item_question_type(item: Mapping[str, Any]) -> str | None:
    value = item.get("question_type")
    return None if value in (None, "") else str(value)


def select_lowcost_few_shot_examples(
    examples: Iterable[Mapping[str, Any]],
    query: Query,
    top_examples: int,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    query_type = resolve_question_type(query).value

    for raw_example in examples:
        example = _example_payload(raw_example)
        split = example.get("split")
        if split != "train":
            raise ValueError("low-cost few-shot examples must include split='train'")
        if query.id and _item_id(example) == query.id:
            continue
        if query.image_id and _item_image_id(example) == query.image_id:
            continue
        if _item_question_type(example) != query_type:
            continue
        if "answer" not in example or example.get("answer") in (None, ""):
            raise ValueError("low-cost few-shot examples must include train gold answers")
        if not (example.get("image_path") or example.get("image_id")):
            raise ValueError("low-cost few-shot examples must include an image reference")

        selected.append(example)
        if len(selected) >= top_examples:
            break

    return selected


def _image_reference_from_item(item: Query | Mapping[str, Any]) -> str:
    if isinstance(item, Query):
        reference = item.image_path or item.image_id
    else:
        reference = item.get("image_path") or item.get("image_id")
    if not reference:
        raise ValueError("low-cost answer-only prompt requires an image reference")
    return str(reference)


def _image_url_part(
    image_reference: str,
    image_url_builder: Any | None = None,
) -> dict[str, Any]:
    url = image_url_builder(image_reference) if image_url_builder else image_reference
    return {"type": "image_url", "image_url": {"url": url}}


def build_lowcost_answer_only_messages(
    query: Query | dict,
    examples: Iterable[Mapping[str, Any]] | None = None,
    prompt_config: PromptConfig | None = None,
    image_url_builder: Any | None = None,
) -> list[dict[str, Any]]:
    query_model = query if isinstance(query, Query) else Query.model_validate(query)
    config = prompt_config or PromptConfig()
    selected_examples = select_lowcost_few_shot_examples(
        examples or [],
        query_model,
        top_examples=config.top_examples,
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": lowcost_answer_only_system_prompt()}
    ]

    for index, example in enumerate(selected_examples, start=1):
        image_reference = _image_reference_from_item(example)
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Few-shot example {index}:\n"
                            + format_lowcost_answer_text(example, include_answer=True)
                        ),
                    },
                    _image_url_part(image_reference, image_url_builder),
                ],
            }
        )

    query_image_reference = _image_reference_from_item(query_model)
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Query:\n"
                        + format_lowcost_answer_text(query_model, include_answer=False)
                    ),
                },
                _image_url_part(query_image_reference, image_url_builder),
            ],
        }
    )
    return messages


def format_few_shot_examples(
    examples: Iterable[Mapping[str, Any]],
    query: Query,
    config: PromptConfig,
) -> str:
    """Format solved train examples while preventing direct query-answer leakage."""
    blocks: list[str] = []
    query_id = query.id
    query_image_id = query.image_id

    for raw_example in examples:
        example = _example_payload(raw_example)
        split = example.get("split")
        if split != "train":
            raise ValueError("few-shot examples must include split='train'")
        if query_id and example.get("sample_id") == query_id:
            continue
        if query_image_id and example.get("image_id") == query_image_id:
            continue
        if "answer" not in example:
            raise ValueError("few-shot examples must include solved training answers")

        index = len(blocks) + 1
        if index > config.top_examples:
            break

        image_reference = example.get("image_path") or example.get("image_id")
        blocks.append(
            "\n".join(
                [
                    f"Example {index}:",
                    "Sample ID:",
                    str(example.get("sample_id") or "(unknown)"),
                    "Image placeholder:",
                    f"{_format_example_image_placeholder(config, index)} ({image_reference})",
                    "Question type:",
                    str(example.get("question_type") or "(unknown)"),
                    "Question:",
                    truncate_text(str(example.get("question", "")), config.max_example_chars),
                    "Choices:",
                    format_example_choices(example),
                    "Gold answer:",
                    str(example["answer"]),
                    "Relevant citations:",
                    format_example_citations(example.get("relevant_articles", [])),
                ]
            )
        )

    if not blocks:
        return "(no retrieved training examples)"
    return "\n\n".join(blocks)


def answer_instruction(query: Query) -> str:
    question_type = resolve_question_type(query)
    if question_type == QuestionType.MULTIPLE_CHOICE:
        return "For this benchmark question, answer must be exactly one of A, B, C, D."
    if question_type == QuestionType.YES_NO:
        return "For this benchmark question, answer must be exactly one of Đúng or Sai."
    return "For free-form demo mode, answer with a short legal conclusion."


def structured_legal_explanation_instruction() -> str:
    return "\n".join(
        [
            "Structured explanation requirements:",
            "- The explanation field must be concise and contain exactly these labeled parts:",
            "  Observation: one short sentence about the visible/question context.",
            "  Legal basis: one short sentence naming the cited article(s) and legal rule.",
            "  Conclusion: one short sentence linking the rule to the final answer.",
            "- Do not expose hidden chain-of-thought, private deliberation, or step-by-step reasoning.",
            "- If evidence is insufficient, set abstained=true and use the three labels to state what is missing.",
        ]
    )


def build_legal_qa_prompt(
    query: Query | dict,
    evidence: Iterable[Evidence | dict],
    examples: Iterable[Mapping[str, Any]] | None = None,
    variant: PromptVariant | str = PromptVariant.TEXT_RAG,
    prompt_config: PromptConfig | None = None,
) -> str:
    """Build the structured legal QA prompt used by the VLM wrapper."""
    query_model = query if isinstance(query, Query) else Query.model_validate(query)
    evidence_models = [
        item if isinstance(item, Evidence) else Evidence.model_validate(item)
        for item in evidence
    ]
    config = prompt_config or PromptConfig()
    prompt_variant = normalize_prompt_variant(variant)
    question_type = resolve_question_type(query_model)
    image_reference = query_model.image_path or query_model.image_id
    evidence_text = (
        "(not provided for zero-shot variant)"
        if prompt_variant == PromptVariant.ZERO_SHOT
        else format_evidence(evidence_models, config.max_evidence_chars)
    )

    sections = [
        LEGAL_QA_SYSTEM_PROMPT,
        "Prompt variant:\n" + prompt_variant.value,
        "Input image placeholder:\n"
        f"{config.image_placeholder} ({image_reference})",
        "Question type:\n" + question_type.value,
        "Question:\n" + query_model.question,
        "Choices:\n" + format_choices(query_model),
        "Retrieved legal evidence:\n" + evidence_text,
    ]

    if prompt_variant == PromptVariant.FEW_SHOT_RAG:
        sections.append(
            "Retrieved solved training examples:\n"
            + format_few_shot_examples(examples or [], query_model, config)
        )

    if prompt_variant == PromptVariant.STRUCTURED_LEGAL_RAG:
        sections.append(structured_legal_explanation_instruction())

    sections.extend(
        [
            "Output requirements:\n"
            "- Return JSON with exactly these top-level fields: "
            "answer, citations, explanation, confidence, abstained.\n"
            "- citations must be a list of objects with law_id and article_id.\n"
            "- citations must refer only to retrieved legal evidence above.\n"
            "- confidence must be a number from 0 to 1.\n"
            "- abstained must be true only when evidence is insufficient.\n"
            "- If abstained is true, keep explanation short and explain what is missing.\n"
            "- Do not include raw chain-of-thought; include only the final short explanation.\n"
            + "- "
            + answer_instruction(query_model),
            "JSON schema:\n"
            '{"answer":"A","citations":[{"law_id":"...","article_id":"..."}],'
            '"explanation":"Observation: ... Legal basis: ... Conclusion: ...",'
            '"confidence":0.0,"abstained":false}',
        ]
    )
    return "\n\n".join(sections)


def build_vlm_messages(
    query: Query | dict,
    evidence: Iterable[Evidence | dict],
    examples: Iterable[Mapping[str, Any]] | None = None,
    variant: PromptVariant | str = PromptVariant.TEXT_RAG,
    prompt_config: PromptConfig | None = None,
    image_url_builder: Any | None = None,
) -> list[dict]:
    """Return a mockable chat-message shape without loading any model weights."""
    prompt_variant = normalize_prompt_variant(variant)
    if prompt_variant == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT:
        return build_lowcost_answer_only_messages(
            query,
            examples=examples,
            prompt_config=prompt_config,
            image_url_builder=image_url_builder,
        )

    prompt = build_legal_qa_prompt(
        query,
        evidence,
        examples=examples,
        variant=prompt_variant,
        prompt_config=prompt_config,
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
