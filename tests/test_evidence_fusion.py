import pytest

from pydantic import ValidationError

from src.data_utils import build_law_article_index
from src.retrieval import (
    ExampleSearchResult,
    fuse_legal_evidence,
    rank_weighted_example_votes,
)
from src.schemas import Citation, Evidence, PipelineResult, Prediction, Query


LAW_ID = "QCVN 41:2024/BGTVT"


def article(article_id: str, title: str | None = None) -> dict:
    return {
        "uid": f"{LAW_ID}#{article_id}",
        "law_id": LAW_ID,
        "law_title": "Quy chuẩn báo hiệu đường bộ",
        "article_id": article_id,
        "title": title or f"Điều {article_id}",
        "content": f"Nội dung {article_id}.",
        "images": [],
        "tables": [],
    }


def direct_evidence(article_id: str, score: float, rank: int) -> Evidence:
    return Evidence(
        law_id=LAW_ID,
        article_id=article_id,
        title=f"Điều {article_id}",
        content=f"Nội dung {article_id}.",
        score=score,
        rank=rank,
        retrieval_method="text",
    )


def example_result(
    sample_id: str,
    rank: int,
    score: float,
    article_ids: list[str],
) -> ExampleSearchResult:
    return ExampleSearchResult(
        payload={
            "sample_id": sample_id,
            "image_id": f"img_{sample_id}",
            "question": "Câu hỏi?",
            "question_type": "Multiple choice",
            "choices": {"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
            "answer": "A",
            "relevant_articles": [
                {"law_id": LAW_ID, "article_id": article_id}
                for article_id in article_ids
            ],
            "image_path": f"images/{sample_id}.jpg",
            "split": "train",
        },
        score=score,
        rank=rank,
        retrieval_mode="fusion",
    )


def article_index(*article_ids: str) -> dict[str, dict]:
    return build_law_article_index([article(article_id) for article_id in article_ids])


def test_rank_weighted_votes_from_retrieved_examples():
    votes, diagnostics = rank_weighted_example_votes(
        [
            example_result("ex1", rank=1, score=0.9, article_ids=["22"]),
            example_result("ex2", rank=2, score=0.6, article_ids=["22", "B.4"]),
        ]
    )

    assert diagnostics == []
    assert votes[f"{LAW_ID}#22"]["score"] == pytest.approx(0.9 + 0.3)
    assert votes[f"{LAW_ID}#22"]["count"] == 2
    assert votes[f"{LAW_ID}#B.4"]["score"] == pytest.approx(0.3)


def test_direct_retrieval_and_example_votes_are_deduplicated():
    result = fuse_legal_evidence(
        direct_evidence=[direct_evidence("22", score=0.7, rank=1)],
        examples=[
            example_result("ex1", rank=1, score=0.8, article_ids=["22", "B.4"]),
        ],
        article_index=article_index("22", "B.4"),
        top_k=5,
        direct_weight=1.0,
        example_vote_weight=1.0,
    )

    assert [item.uid for item in result.evidence] == [f"{LAW_ID}#22", f"{LAW_ID}#B.4"]
    assert result.evidence[0].retrieval_method == "fusion"
    assert result.evidence[0].metadata["direct_score"] == 0.7
    assert result.evidence[0].metadata["example_vote_score"] == pytest.approx(0.8)
    assert result.evidence[0].metadata["source_mode"] == "direct+examples"
    assert result.diagnostics == []


def test_fusion_ranking_is_deterministic_and_configurable():
    result = fuse_legal_evidence(
        direct_evidence=[
            direct_evidence("22", score=0.9, rank=1),
            direct_evidence("B.4", score=0.2, rank=2),
        ],
        examples=[
            example_result("ex1", rank=1, score=0.9, article_ids=["B.4"]),
        ],
        article_index=article_index("22", "B.4"),
        top_k=2,
        direct_weight=0.1,
        example_vote_weight=1.0,
    )

    assert [item.uid for item in result.evidence] == [f"{LAW_ID}#B.4", f"{LAW_ID}#22"]
    assert result.evidence[0].score == pytest.approx((0.1 * 0.2) + 0.9)


def test_unknown_example_citation_is_reported_not_hidden():
    result = fuse_legal_evidence(
        direct_evidence=[direct_evidence("22", score=0.7, rank=1)],
        examples=[example_result("ex1", rank=1, score=0.8, article_ids=["999"])],
        article_index=article_index("22"),
        top_k=5,
    )

    assert [item.uid for item in result.evidence] == [f"{LAW_ID}#22"]
    assert result.diagnostics[0]["type"] == "unknown_law_article"
    assert result.diagnostics[0]["uid"] == f"{LAW_ID}#999"
    assert all(item.article_id != "999" for item in result.evidence)


def test_fused_evidence_validates_with_pipeline_result():
    result = fuse_legal_evidence(
        direct_evidence=[direct_evidence("22", score=0.7, rank=1)],
        examples=[example_result("ex1", rank=1, score=0.8, article_ids=["22"])],
        article_index=article_index("22"),
        top_k=1,
    )
    query = Query(
        id="q1",
        image_id="img1",
        question="Biển báo này có ý nghĩa gì?",
        question_type="Multiple choice",
        choices={"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        answer="A",
    )
    prediction = Prediction(
        id="q1",
        question_type="Multiple choice",
        answer="A",
        citations=[result.evidence[0].to_citation()],
        explanation="Dựa trên Điều 22.",
    )

    pipeline_result = PipelineResult(
        query=query,
        evidence=result.evidence,
        prediction=prediction,
    )

    assert pipeline_result.evidence[0].title == "Điều 22"
    assert pipeline_result.evidence[0].content == "Nội dung 22."


def test_pipeline_still_rejects_hallucinated_citation_after_fusion():
    result = fuse_legal_evidence(
        direct_evidence=[direct_evidence("22", score=0.7, rank=1)],
        examples=[],
        article_index=article_index("22"),
        top_k=1,
    )
    query = Query(
        id="q1",
        image_id="img1",
        question="Biển báo này có ý nghĩa gì?",
        question_type="Multiple choice",
        choices={"A": "Một", "B": "Hai", "C": "Ba", "D": "Bốn"},
        answer="A",
    )
    prediction = Prediction(
        id="q1",
        question_type="Multiple choice",
        answer="A",
        citations=[Citation(law_id=LAW_ID, article_id="B.4")],
        explanation="Trích dẫn sai.",
    )

    with pytest.raises(ValidationError, match="outside retrieved evidence"):
        PipelineResult(query=query, evidence=result.evidence, prediction=prediction)
