from __future__ import annotations

import unicodedata
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)


def _normalize_text(value: Any) -> str:
    """Normalize labels and Vietnamese text from the VLSP data."""
    return unicodedata.normalize("NFC", str(value)).strip()


def _normalize_answer(value: Any) -> str:
    # One VLSP training sample uses the legacy integer value 40 for choice A.
    if value == 40 or value == "40":
        return "A"

    answer = _normalize_text(value)
    yes_no_mapping = {
        "yes": "Đúng",
        "true": "Đúng",
        "no": "Sai",
        "false": "Sai",
        "không": "Sai",
        "fail": "Sai",
    }
    return yes_no_mapping.get(answer.casefold(), answer)


class SchemaModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class QuestionType(str, Enum):
    MULTIPLE_CHOICE = "Multiple choice"
    YES_NO = "Yes/No"
    FREE_FORM = "Free-form"


class RetrievalMethod(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    EXAMPLE = "example"
    FUSION = "fusion"
    ORACLE = "oracle"


class Citation(SchemaModel):
    """A stable reference to one article/term in the legal corpus."""

    law_id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    title: str | None = None
    quote: str | None = None

    @field_validator("law_id", "article_id", "title", "quote", mode="before")
    @classmethod
    def normalize_strings(cls, value: Any) -> Any:
        return None if value is None else _normalize_text(value)

    @property
    def uid(self) -> str:
        return f"{self.law_id}#{self.article_id}"

    def to_vlsp_reference(self) -> dict[str, str]:
        return {"law_id": self.law_id, "article_id": self.article_id}


class Evidence(Citation):
    """A retrieved legal passage or visually matched legal item."""

    content: str = Field(min_length=1)
    score: float | None = None
    rank: int | None = Field(default=None, ge=1)
    retrieval_method: RetrievalMethod
    chunk_id: str | None = None
    image_path: str | None = None
    sign_name: str | None = None
    sign_description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "content",
        "chunk_id",
        "image_path",
        "sign_name",
        "sign_description",
        mode="before",
    )
    @classmethod
    def normalize_evidence_strings(cls, value: Any) -> Any:
        return None if value is None else _normalize_text(value)

    def to_citation(self, quote: str | None = None) -> Citation:
        return Citation(
            law_id=self.law_id,
            article_id=self.article_id,
            title=self.title,
            quote=quote,
        )


class BoundingBox(RootModel[tuple[float, float, float, float]]):
    """Bounding box serialized as [x_min, y_min, x_max, y_max]."""

    @model_validator(mode="after")
    def validate_coordinates(self) -> BoundingBox:
        x_min, y_min, x_max, y_max = self.root
        if x_min < 0 or y_min < 0:
            raise ValueError("bounding-box coordinates must be non-negative")
        if x_max <= x_min or y_max <= y_min:
            raise ValueError("bounding box must have positive width and height")
        return self

    @property
    def x_min(self) -> float:
        return self.root[0]

    @property
    def y_min(self) -> float:
        return self.root[1]

    @property
    def x_max(self) -> float:
        return self.root[2]

    @property
    def y_max(self) -> float:
        return self.root[3]


class DetectedSign(SchemaModel):
    """Detector output compatible with the reference projects' sign crops."""

    image_name: str | None = None
    image_path: str | None = None
    bbox: BoundingBox | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    is_chosen: bool = False
    sign_name: str | None = None
    sign_description: str | None = None

    @field_validator(
        "image_name",
        "image_path",
        "sign_name",
        "sign_description",
        mode="before",
    )
    @classmethod
    def normalize_sign_strings(cls, value: Any) -> Any:
        return None if value is None else _normalize_text(value)

    @model_validator(mode="after")
    def require_location(self) -> DetectedSign:
        if not self.image_name and not self.image_path and self.bbox is None:
            raise ValueError("a detected sign needs an image reference or bounding box")
        return self


class Query(SchemaModel):
    """Canonical input for benchmark samples and free-form demo queries."""

    id: str | None = None
    image_id: str = Field(min_length=1)
    image_path: str | None = None
    question: str = Field(min_length=1)
    question_type: QuestionType | None = None
    choices: dict[str, str] = Field(default_factory=dict)
    relevant_articles: list[Citation] = Field(default_factory=list)
    answer: str | None = None

    @field_validator("id", "image_id", "image_path", "question", mode="before")
    @classmethod
    def normalize_query_strings(cls, value: Any) -> Any:
        return None if value is None else _normalize_text(value)

    @field_validator("choices", mode="before")
    @classmethod
    def normalize_choices(cls, value: Any) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("choices must be an object keyed by A, B, C and D")

        choices: dict[str, str] = {}
        for raw_key, raw_text in value.items():
            key = _normalize_text(raw_key).upper()
            if key in choices:
                raise ValueError(f"duplicate choice key after normalization: {key}")
            choices[key] = _normalize_text(raw_text)
        return choices

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_gold_answer(cls, value: Any) -> Any:
        return None if value is None else _normalize_answer(value)

    @model_validator(mode="after")
    def validate_question_shape(self) -> Query:
        if self.question_type == QuestionType.MULTIPLE_CHOICE:
            expected = {"A", "B", "C", "D"}
            if set(self.choices) != expected:
                raise ValueError("multiple-choice questions require choices A, B, C and D")
            if self.answer is not None and self.answer not in expected:
                raise ValueError("multiple-choice answer must be A, B, C or D")

        elif self.question_type == QuestionType.YES_NO:
            if self.choices:
                raise ValueError("Yes/No questions must not define A-D choices")
            if self.answer is not None and self.answer not in {"Đúng", "Sai"}:
                raise ValueError("Yes/No answer must be Đúng or Sai")

        elif self.question_type == QuestionType.FREE_FORM and self.choices:
            raise ValueError("free-form questions must not define choices")

        return self


class Prediction(SchemaModel):
    """Validated VLM output for benchmark and free-form modes."""

    id: str | None = None
    question_type: QuestionType | None = None
    answer: str = Field(min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    explanation: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool = False
    disclaimer: str | None = None
    raw_response: str | None = None

    @field_validator("id", "explanation", "disclaimer", "raw_response", mode="before")
    @classmethod
    def normalize_prediction_strings(cls, value: Any) -> Any:
        return None if value is None else _normalize_text(value)

    @field_validator("answer", mode="before")
    @classmethod
    def normalize_predicted_answer(cls, value: Any) -> str:
        return _normalize_answer(value)

    @model_validator(mode="after")
    def validate_prediction(self) -> Prediction:
        if not self.abstained and not self.citations:
            raise ValueError("a non-abstained prediction requires at least one citation")

        citation_ids = [citation.uid for citation in self.citations]
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("prediction citations must be unique")

        if self.question_type == QuestionType.MULTIPLE_CHOICE and self.answer not in {
            "A",
            "B",
            "C",
            "D",
        }:
            raise ValueError("multiple-choice prediction must be A, B, C or D")
        if self.question_type == QuestionType.YES_NO and self.answer not in {
            "Đúng",
            "Sai",
        }:
            raise ValueError("Yes/No prediction must be Đúng or Sai")

        return self


class PipelineResult(SchemaModel):
    """Complete auditable result from retrieval through answer generation."""

    query: Query
    evidence: list[Evidence] = Field(default_factory=list)
    detected_signs: list[DetectedSign] = Field(default_factory=list)
    prediction: Prediction
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @field_validator("timings_ms")
    @classmethod
    def validate_timings(cls, value: dict[str, float]) -> dict[str, float]:
        if any(duration < 0 for duration in value.values()):
            raise ValueError("pipeline timings must be non-negative")
        return value

    @model_validator(mode="after")
    def validate_grounded_result(self) -> PipelineResult:
        if self.query.id and self.prediction.id and self.query.id != self.prediction.id:
            raise ValueError("query and prediction IDs must match")

        query_type = self.query.question_type
        prediction_type = self.prediction.question_type
        if query_type and prediction_type and query_type != prediction_type:
            raise ValueError("query and prediction question types must match")

        evidence_ids = {item.uid for item in self.evidence}
        unsupported = {
            citation.uid
            for citation in self.prediction.citations
            if citation.uid not in evidence_ids
        }
        if unsupported:
            raise ValueError(
                "prediction cites articles outside retrieved evidence: "
                + ", ".join(sorted(unsupported))
            )

        answer_type = query_type or prediction_type
        if answer_type == QuestionType.MULTIPLE_CHOICE:
            if self.prediction.answer not in self.query.choices:
                raise ValueError("prediction answer is not one of the query choices")
        elif answer_type == QuestionType.YES_NO:
            if self.prediction.answer not in {"Đúng", "Sai"}:
                raise ValueError("Yes/No prediction must be Đúng or Sai")

        return self
