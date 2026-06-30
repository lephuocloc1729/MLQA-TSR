from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from pathlib import Path
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
ANSWER_TAG_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_ONLY_JSON_KEYS = ("choice", "answer")


class VLMClient(Protocol):
    def generate(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        ...


class OpenAICompatibleClient:
    """Minimal OpenAI-compatible chat-completions client using stdlib HTTP."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 0,
        retry_sleep_seconds: float = 0.5,
    ) -> None:
        if not base_url:
            raise ValueError("OpenAI-compatible backend requires a non-empty base_url")
        if not api_key:
            raise ValueError("OpenAI-compatible backend requires a non-empty API key")
        self.endpoint = chat_completions_endpoint(base_url)
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.max_retries = max(0, int(max_retries))
        self.retry_sleep_seconds = max(0.0, float(retry_sleep_seconds))

    def generate(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        last_error: RuntimeError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._generate_once(
                    messages=messages,
                    model_name=model_name,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )
            except RuntimeError as exc:
                last_error = exc
                if attempt >= self.max_retries or not is_retryable_backend_error(exc):
                    raise
                if self.retry_sleep_seconds:
                    time.sleep(self.retry_sleep_seconds)
        assert last_error is not None
        raise last_error

    def _generate_once(
        self,
        messages: list[dict],
        model_name: str,
        temperature: float,
        max_new_tokens: int,
    ) -> str:
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_new_tokens,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"VLM backend HTTP {exc.code}: {error_body[:500]}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"VLM backend request failed: {exc.reason}") from exc

        data = json.loads(raw)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("VLM backend response is missing choices[0].message.content") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("VLM backend returned an empty response")
        return content


def chat_completions_endpoint(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return normalized + "/chat/completions"


def is_retryable_backend_error(error: RuntimeError) -> bool:
    message = str(error).casefold()
    retryable_markers = (
        "empty response",
        "http 408",
        "http 429",
        "http 500",
        "http 502",
        "http 503",
        "http 504",
        "request failed",
        "timed out",
        "timeout",
    )
    return any(marker in message for marker in retryable_markers)


def load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    load_dotenv()


def env_value(
    environ: Mapping[str, str],
    name: str | None,
    default: str | None = None,
) -> str | None:
    if name and environ.get(name):
        return environ[name]
    return default


def create_vlm_client(
    model_config: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> VLMClient | None:
    backend = str(model_config.get("backend", "none")).strip().lower()
    if backend in {"", "none", "mock", "fake"}:
        return None

    if environ is None:
        load_dotenv_if_available()
        environ = os.environ

    if backend in {"openai_compatible", "openai-compatible", "openai"}:
        api_key_env = str(model_config.get("api_key_env", "OPENAI_COMPATIBLE_API_KEY"))
        base_url_env = str(
            model_config.get("base_url_env", "OPENAI_COMPATIBLE_BASE_URL")
        )
        api_key = env_value(environ, api_key_env, model_config.get("api_key"))
        base_url = env_value(environ, base_url_env, model_config.get("base_url"))
        missing = []
        if not api_key:
            missing.append(api_key_env)
        if not base_url:
            missing.append(base_url_env)
        if missing:
            raise RuntimeError(
                "Missing VLM backend configuration. Set environment variable(s): "
                + ", ".join(missing)
            )
        return OpenAICompatibleClient(
            base_url=str(base_url),
            api_key=str(api_key),
            timeout_seconds=float(model_config.get("request_timeout_seconds", 120)),
            max_retries=int(model_config.get("max_retries", 0)),
            retry_sleep_seconds=float(model_config.get("retry_sleep_seconds", 0.5)),
        )

    raise ValueError(f"Unsupported VLM backend: {backend}")


def normalize_text(value: Any) -> str:
    return unicodedata.normalize("NFC", str(value)).strip()


def image_path_to_data_url(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"VLM image input not found: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def query_image_message_part(query: Query) -> dict[str, Any] | None:
    image_path = query.image_path
    if not image_path:
        return None
    return {"type": "image_url", "image_url": {"url": image_path_to_data_url(image_path)}}


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


def normalize_answer_only_label(label: Any, question_type: QuestionType) -> str:
    if not isinstance(label, str):
        raise ValueError("answer-only response label must be a string")

    normalized = normalize_text(label)
    yes_no_mapping = {
        "yes": "Đúng",
        "true": "Đúng",
        "no": "Sai",
        "false": "Sai",
    }
    if question_type == QuestionType.YES_NO:
        normalized = yes_no_mapping.get(normalized.casefold(), normalized)

    return validate_raw_answer(normalized, question_type)


def extract_answer_only_label(raw_response: str) -> str:
    text = normalize_text(raw_response)
    if not text:
        raise ValueError("answer-only response is empty")

    tag_match = ANSWER_TAG_PATTERN.search(text)
    if tag_match:
        return normalize_text(tag_match.group(1))

    fence_match = JSON_FENCE_PATTERN.search(text)
    candidate = fence_match.group(1).strip() if fence_match else text
    if candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = extract_json_object(text)
        if not isinstance(payload, Mapping):
            raise ValueError("answer-only JSON response must be an object")
        for key in ANSWER_ONLY_JSON_KEYS:
            if key in payload:
                return normalize_text(payload[key])
        raise ValueError("answer-only JSON must include choice or answer")

    return text


def answer_only_citations(evidence: list[Evidence]) -> tuple[list[dict[str, str]], bool]:
    if not evidence:
        return [], True
    first = evidence[0].to_citation()
    return [{"law_id": first.law_id, "article_id": first.article_id}], False


def parse_answer_only_prediction(
    raw_response: str,
    query: Query | dict,
    evidence: list[Evidence | dict],
) -> Prediction:
    query_model = query if isinstance(query, Query) else Query.model_validate(query)
    evidence_models = [
        item if isinstance(item, Evidence) else Evidence.model_validate(item)
        for item in evidence
    ]
    question_type = resolve_question_type(query_model)
    label = normalize_answer_only_label(
        extract_answer_only_label(raw_response),
        question_type,
    )
    citations, abstained = answer_only_citations(evidence_models)
    prediction = Prediction.model_validate(
        {
            "id": query_model.id,
            "question_type": question_type,
            "answer": label,
            "citations": citations,
            "explanation": "Answer-only benchmark prompt; no explanation requested.",
            "confidence": None,
            "abstained": abstained,
            "raw_response": raw_response,
        }
    )
    PipelineResult(query=query_model, evidence=evidence_models, prediction=prediction)
    return prediction


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
        environ: Mapping[str, str] | None = None,
    ) -> None:
        model_config = config.get("model", {})
        prompt_config = config.get("prompt", {})
        if environ is None:
            load_dotenv_if_available()
            environ = os.environ
        name_env = model_config.get("name_env")
        self.backend = str(model_config.get("backend", "none")).strip().lower()
        self.model_name = env_value(
            environ,
            str(name_env) if name_env else None,
            model_config.get("name", "Qwen/Qwen2.5-VL-3B-Instruct"),
        )
        self.temperature = float(model_config.get("temperature", 0.0))
        self.max_new_tokens = int(model_config.get("max_new_tokens", 512))
        self.include_image = bool(model_config.get("include_image", False))
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
        self.client = client if client is not None else create_vlm_client(
            model_config,
            environ=environ,
        )

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
        query_model = query if isinstance(query, Query) else Query.model_validate(query)
        messages = build_vlm_messages(
            query_model,
            evidence,
            examples=examples,
            variant=variant or self.prompt_variant,
            prompt_config=self.prompt_config,
            image_url_builder=image_path_to_data_url
            if self.include_image
            and normalize_prompt_variant(variant or self.prompt_variant)
            == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT
            else None,
        )
        effective_variant = normalize_prompt_variant(variant or self.prompt_variant)
        image_part = (
            query_image_message_part(query_model)
            if self.include_image
            and effective_variant != PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT
            else None
        )
        if image_part is not None:
            content = messages[0].setdefault("content", [])
            if not isinstance(content, list):
                raise ValueError("VLM message content must be a list for image input")
            content.append(image_part)
        return messages

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
        effective_variant = normalize_prompt_variant(variant or self.prompt_variant)
        if effective_variant == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT:
            return parse_answer_only_prediction(
                raw_response,
                query=query,
                evidence=evidence,
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

    if prompt_variant in {
        PromptVariant.TEXT_RAG,
        PromptVariant.STRUCTURED_LEGAL_RAG,
    }:
        evidence = retrieve_evidence(query, dict(config), top_k=top_k)
    elif prompt_variant in {
        PromptVariant.FEW_SHOT_RAG,
        PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT,
    }:
        if not example_collection_exists(dict(config)):
            if prompt_variant == PromptVariant.FEW_SHOT_RAG:
                raise RuntimeError(
                    "Example collection is unavailable. Run "
                    "`python -m src.retrieval --mode index-examples --split train` "
                    "before building a few_shot_rag prompt."
                )
            examples = []
            evidence = []
            return json.dumps(
                build_vlm_messages(
                    query,
                    evidence,
                    examples=examples,
                    variant=prompt_variant,
                    prompt_config=runtime.prompt_config,
                ),
                ensure_ascii=False,
                indent=2,
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

    if prompt_variant == PromptVariant.LOWCOST_ANSWER_ONLY_FEWSHOT:
        return json.dumps(
            build_vlm_messages(
                query,
                evidence,
                examples=examples,
                variant=prompt_variant,
                prompt_config=runtime.prompt_config,
            ),
            ensure_ascii=False,
            indent=2,
        )

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
