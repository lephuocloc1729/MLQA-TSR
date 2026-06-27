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
            '"explanation":"...","confidence":0.0,"abstained":false}',
        ]
    )
    return "\n\n".join(sections)


def build_vlm_messages(
    query: Query | dict,
    evidence: Iterable[Evidence | dict],
    examples: Iterable[Mapping[str, Any]] | None = None,
    variant: PromptVariant | str = PromptVariant.TEXT_RAG,
    prompt_config: PromptConfig | None = None,
) -> list[dict]:
    """Return a mockable chat-message shape without loading any model weights."""
    prompt = build_legal_qa_prompt(
        query,
        evidence,
        examples=examples,
        variant=variant,
        prompt_config=prompt_config,
    )
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
