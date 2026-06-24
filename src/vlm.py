from __future__ import annotations

import argparse
import json
import re
import unicodedata
from typing import Any, Mapping, Protocol

from src.prompts import PromptConfig, build_vlm_messages
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
        self.prompt_config = PromptConfig(
            image_placeholder=prompt_config.get("image_placeholder", "<IMAGE>"),
            max_evidence_chars=int(prompt_config.get("max_evidence_chars", 1200)),
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
    ) -> list[dict]:
        return build_vlm_messages(query, evidence, prompt_config=self.prompt_config)

    def answer(
        self,
        query: Query | dict,
        evidence: list[Evidence | dict],
    ) -> Prediction:
        if self.client is None:
            raise RuntimeError(
                "No VLM client configured. Inject a client with generate(...) "
                "or connect a model backend in the pipeline."
            )

        messages = self.build_messages(query, evidence)
        raw_response = self.client.generate(
            messages=messages,
            model_name=self.model_name,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
        )
        return parse_prediction(raw_response, query=query, evidence=evidence)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured prompt and output parser for legal VLM QA."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print runtime model/prompt settings without loading model weights.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = LegalQAVLM.from_config(args.config)
    if args.show_config:
        print(
            json.dumps(
                {
                    "model_name": runtime.model_name,
                    "temperature": runtime.temperature,
                    "max_new_tokens": runtime.max_new_tokens,
                    "prompt": runtime.prompt_config.__dict__,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
