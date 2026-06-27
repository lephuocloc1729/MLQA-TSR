from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any, Mapping, Protocol

from src.prompts import (
    PromptConfig,
    PromptVariant,
    build_legal_qa_prompt,
    build_vlm_messages,
    normalize_prompt_variant,
)
from src.schemas import Evidence, PipelineResult, Prediction, Query, QuestionType
from src.utils import load_config


REQUIRED_OUTPUT_FIELDS = {
    "answer",
    "citations",
    "explanation",
    "confidence",
    "abstained",
}
JSON_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


class VLMClient(Protocol):
    def generate(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        ...


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def extract_json_object(raw_response: str) -> dict[str, Any]:
    """Extract one JSON object from raw VLM text or a markdown JSON fence."""
    text = raw_response.strip()
    fence_match = JSON_FENCE_PATTERN.search(text)
    candidate = fence_match.group(1).strip() if fence_match else text

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("VLM response does not contain a JSON object") from None
        payload = json.loads(text[start : end + 1])

    if not isinstance(payload, dict):
        raise ValueError("VLM response JSON must be an object")
    return payload


def validate_output_fields(payload: Mapping[str, Any]) -> None:
    fields = set(payload)
    if fields != REQUIRED_OUTPUT_FIELDS:
        missing = sorted(REQUIRED_OUTPUT_FIELDS - fields)
        extra = sorted(fields - REQUIRED_OUTPUT_FIELDS)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(
            "VLM JSON must contain exactly answer, citations, explanation, "
            f"confidence, abstained ({'; '.join(details)})"
        )
    if not isinstance(payload["citations"], list):
        raise ValueError("VLM JSON field 'citations' must be a list")
    if not isinstance(payload["abstained"], bool):
        raise ValueError("VLM JSON field 'abstained' must be a boolean")


def resolve_question_type(query: Query) -> QuestionType:
    if query.question_type is not None:
        return query.question_type
    if query.choices:
        return QuestionType.MULTIPLE_CHOICE
    return QuestionType.FREE_FORM


def validate_raw_answer(answer: Any, question_type: QuestionType) -> str:
    if not isinstance(answer, str):
        raise ValueError("VLM JSON field 'answer' must be a string")

    normalized = normalize_text(answer)
    if question_type == QuestionType.MULTIPLE_CHOICE:
        if normalized not in {"A", "B", "C", "D"}:
            raise ValueError("multiple-choice prediction must be A, B, C or D")
    elif question_type == QuestionType.YES_NO:
        if normalized not in {"Đúng", "Sai"}:
            raise ValueError("Yes/No prediction must be Đúng or Sai")
    elif not normalized:
        raise ValueError("free-form prediction answer must be non-empty")
    return normalized


def parse_prediction(
    raw_response: str,
    query: Query | dict,
    evidence: list[Evidence | dict],
) -> Prediction:
    """Parse raw VLM output and validate it against retrieved evidence."""
    query_model = query if isinstance(query, Query) else Query.model_validate(query)
    evidence_models = [
        item if isinstance(item, Evidence) else Evidence.model_validate(item)
        for item in evidence
    ]
    payload = extract_json_object(raw_response)
    validate_output_fields(payload)

    question_type = resolve_question_type(query_model)
    payload = dict(payload)
    payload["answer"] = validate_raw_answer(payload["answer"], question_type)
    prediction = Prediction.model_validate(
        {
            **payload,
            "id": query_model.id,
            "question_type": question_type,
            "raw_response": raw_response,
        }
    )

    PipelineResult(query=query_model, evidence=evidence_models, prediction=prediction)
    return prediction


class LegalQAVLM:
    """Thin runtime wrapper; inject a fake client in tests to avoid model loads."""

    def __init__(
        self,
        config: Mapping[str, Any],
        client: VLMClient | None = None,
    ) -> None:
        model_config = config.get("model", {})
        prompt_config = config.get("prompt", {})
        self.model_name = model_config.get("name", "Qwen/Qwen2.5-VL-3B-Instruct")
        self.temperature = float(model_config.get("temperature", 0.0))
        self.max_new_tokens = int(model_config.get("max_new_tokens", 512))
        self.prompt_variant = normalize_prompt_variant(
            prompt_config.get("variant", PromptVariant.TEXT_RAG.value)
        )
        self.prompt_config = PromptConfig(
            image_placeholder=prompt_config.get("image_placeholder", "<IMAGE>"),
            example_image_placeholder=prompt_config.get(
                "example_image_placeholder",
                "<EXAMPLE_IMAGE_{index}>",
            ),
            max_evidence_chars=int(prompt_config.get("max_evidence_chars", 1200)),
            max_example_chars=int(prompt_config.get("max_example_chars", 700)),
            top_examples=int(prompt_config.get("top_examples", 3)),
        )
        self.client = client

    @classmethod
    def from_config(
        cls,
        config_path: str = "configs/config.yaml",
        client: VLMClient | None = None,
    ) -> "LegalQAVLM":
        return cls(load_config(config_path), client=client)

    def build_messages(
        self,
        query: Query | dict,
        evidence: list[Evidence | dict],
        examples: list[Mapping[str, Any]] | None = None,
        variant: PromptVariant | str | None = None,
    ) -> list[dict]:
        return build_vlm_messages(
            query,
            evidence,
            examples=examples,
            variant=variant or self.prompt_variant,
            prompt_config=self.prompt_config,
        )

    def answer(
        self,
        query: Query | dict,
        evidence: list[Evidence | dict],
        examples: list[Mapping[str, Any]] | None = None,
        variant: PromptVariant | str | None = None,
    ) -> Prediction:
        if self.client is None:
            raise RuntimeError(
                "No VLM client configured. Inject a client with generate(...) "
                "or connect a model backend in the pipeline."
            )

        messages = self.build_messages(
            query,
            evidence,
            examples=examples,
            variant=variant,
        )
        raw_response = self.client.generate(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        return parse_prediction(raw_response, query=query, evidence=evidence)


def _top_examples_from_config(config: Mapping[str, Any]) -> int:
    prompt_top_k = config.get("prompt", {}).get("top_examples")
    if prompt_top_k is not None:
        return int(prompt_top_k)
    return int(config.get("retrieval", {}).get("example_top_k", 3))


def build_prompt_for_sample(
    config: Mapping[str, Any],
    sample_id: str,
    variant: PromptVariant | str,
    top_k: int | None = None,
    example_top_k: int | None = None,
    retrieval_mode: str = "fusion",
) -> str:
    """Build a prompt from local splits and retrieval outputs for CLI inspection."""
    from src.retrieval import (
        example_collection_exists,
        load_query_sample_by_id,
        retrieve_evidence,
        retrieve_examples,
        retrieve_fused_evidence,
    )

    prompt_variant = normalize_prompt_variant(variant)
    runtime = LegalQAVLM(config)
    query = load_query_sample_by_id(dict(config), sample_id)
    evidence: list[Evidence | dict] = []
    examples: list[Mapping[str, Any]] = []

    if prompt_variant == PromptVariant.TEXT_RAG:
        evidence = retrieve_evidence(query, dict(config), top_k=top_k)
    elif prompt_variant == PromptVariant.FEW_SHOT_RAG:
        if not example_collection_exists(dict(config)):
            raise RuntimeError(
                "Example collection is unavailable. Run "
                "`python -m src.retrieval --mode index-examples --split train` "
                "before building a few_shot_rag prompt."
            )
        resolved_example_top_k = example_top_k or _top_examples_from_config(config)
        fused_result = retrieve_fused_evidence(
            query,
            dict(config),
            top_k=top_k,
            example_top_k=resolved_example_top_k,
            example_mode=retrieval_mode,
            allow_example_failure=False,
        )
        evidence = fused_result.evidence
        examples = retrieve_examples(
            query,
            dict(config),
            mode=retrieval_mode,
            top_k=resolved_example_top_k,
        )
    elif prompt_variant == PromptVariant.ZERO_SHOT:
        evidence = []
        examples = []

    return build_legal_qa_prompt(
        query,
        evidence,
        examples=examples,
        variant=prompt_variant,
        prompt_config=runtime.prompt_config,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured prompt and output parser for legal VLM QA."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--mode",
        choices=["show-config", "build-prompt"],
        default="show-config",
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in PromptVariant],
        default=None,
        help="Prompt variant to build.",
    )
    parser.add_argument("--sample-id", help="Sample ID for --mode build-prompt.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--example-top-k", type=int, default=None)
    parser.add_argument(
        "--retrieval-mode",
        default="fusion",
        choices=["text", "image", "fusion"],
        help="Example retrieval mode for few_shot_rag prompts.",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print runtime model/prompt settings without loading model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime = LegalQAVLM(config)
    mode = "show-config" if args.show_config else args.mode
    if mode == "show-config":
        print(
            json.dumps(
                {
                    "model_name": runtime.model_name,
                    "temperature": runtime.temperature,
                    "max_new_tokens": runtime.max_new_tokens,
                    "prompt_variant": runtime.prompt_variant.value,
                    "prompt": runtime.prompt_config.__dict__,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    elif mode == "build-prompt":
        if not args.sample_id:
            raise SystemExit("--sample-id is required when --mode build-prompt")
        variant = args.variant or runtime.prompt_variant.value
        print(
            build_prompt_for_sample(
                config,
                sample_id=args.sample_id,
                variant=variant,
                top_k=args.top_k,
                example_top_k=args.example_top_k,
                retrieval_mode=args.retrieval_mode,
            )
        )


if __name__ == "__main__":
    main()
