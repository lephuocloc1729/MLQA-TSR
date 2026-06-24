from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from src.schemas import Evidence, Query, QuestionType


LEGAL_QA_SYSTEM_PROMPT = """You are a Vietnamese traffic-law QA assistant.
Use the image, question, answer choices, and retrieved legal evidence only.
Return only one JSON object. Do not include markdown, XML tags, or hidden
chain-of-thought. The explanation must be short and cite the evidence used."""


@dataclass(frozen=True)
class PromptConfig:
    image_placeholder: str = "<IMAGE>"
    max_evidence_chars: int = 1200


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
    prompt_config: PromptConfig | None = None,
) -> str:
    """Build the structured legal QA prompt used by the week-1 VLM wrapper."""
    query_model = query if isinstance(query, Query) else Query.model_validate(query)
    evidence_models = [
        item if isinstance(item, Evidence) else Evidence.model_validate(item)
        for item in evidence
    ]
    config = prompt_config or PromptConfig()
    question_type = resolve_question_type(query_model)
    image_reference = query_model.image_path or query_model.image_id

    return "\n\n".join(
        [
            LEGAL_QA_SYSTEM_PROMPT,
            "Input image placeholder:\n"
            f"{config.image_placeholder} ({image_reference})",
            "Question type:\n" + question_type.value,
            "Question:\n" + query_model.question,
            "Choices:\n" + format_choices(query_model),
            "Retrieved legal evidence:\n"
            + format_evidence(evidence_models, config.max_evidence_chars),
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


def build_vlm_messages(
    query: Query | dict,
    evidence: Iterable[Evidence | dict],
    prompt_config: PromptConfig | None = None,
) -> list[dict]:
    """Return a mockable chat-message shape without loading any model weights."""
    prompt = build_legal_qa_prompt(query, evidence, prompt_config=prompt_config)
    return [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
