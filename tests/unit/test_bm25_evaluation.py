import json

import pytest

from knowledge_hub.evaluation import EvaluationQuestion, evaluate_retriever
from knowledge_hub.evaluation.benchmark import load_questions
from knowledge_hub.models import Chunk, SourceType
from knowledge_hub.retrieval import BM25Retriever
from knowledge_hub.retrieval.types import RankedChunk


def make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        document_id="document",
        chunk_id=chunk_id,
        content=content,
        source_type=SourceType.PDF,
        parent_structure="Chapter 1",
        provenance={"section": "Chapter 1"},
    )


def test_evaluation_uses_explicit_gold_chunk_ids() -> None:
    chunks = (
        make_chunk("gold", "10X Rule goals and action"),
        make_chunk("wrong", "10X Rule unrelated discussion"),
    )

    questions = (
        EvaluationQuestion(
            id="Q1",
            category="factual",
            question="10X Rule goals",
            answerable=True,
            gold_chunk_ids=frozenset({"gold"}),
            graded_relevance={"gold": 3},
        ),
    )

    report = evaluate_retriever(
        BM25Retriever(chunks),
        chunks,
        questions,
        name="bm25",
    )

    query = report.queries[0]

    assert query.relevant_chunk_ids == ("gold",)
    assert query.result_chunk_ids[0] == "gold"
    assert report.metrics.recall_at_1 == 1.0
    assert report.metrics.mrr == 1.0


def test_evaluation_does_not_infer_relevance_from_section_metadata() -> None:
    chunks = (
        make_chunk("gold", "completely different content"),
        make_chunk("wrong", "another unrelated passage"),
    )

    questions = (
        EvaluationQuestion(
            id="Q1",
            category="factual",
            question="gold",
            answerable=True,
            gold_chunk_ids=frozenset({"gold"}),
            graded_relevance={"gold": 3},
        ),
    )

    report = evaluate_retriever(
        BM25Retriever(chunks),
        chunks,
        questions,
        name="bm25",
    )

    assert report.queries[0].relevant_chunk_ids == ("gold",)


def test_unanswerable_question_has_no_gold_chunks() -> None:
    chunks = (
        make_chunk("one", "retrieval architecture"),
        make_chunk("two", "embedding pipeline"),
    )

    questions = (
        EvaluationQuestion(
            id="Q1",
            category="unanswerable",
            question="Qdrant configuration",
            answerable=False,
            gold_chunk_ids=frozenset(),
            graded_relevance={},
        ),
    )

    report = evaluate_retriever(
        BM25Retriever(chunks),
        chunks,
        questions,
        name="bm25",
    )

    query = report.queries[0]

    assert query.relevant_chunk_ids == ()
    assert report.metrics.recall_at_1 == 0.0
    assert report.metrics.recall_at_5 == 0.0


def test_multiple_gold_chunks_are_supported() -> None:
    chunks = (
        make_chunk("gold-1", "first part of the answer"),
        make_chunk("gold-2", "second part of the answer"),
        make_chunk("wrong", "unrelated content"),
    )

    questions = (
        EvaluationQuestion(
            id="Q1",
            category="multi-evidence",
            question="part of the answer",
            answerable=True,
            gold_chunk_ids=frozenset({"gold-1", "gold-2"}),
            graded_relevance={
                "gold-1": 3,
                "gold-2": 3,
            },
        ),
    )

    report = evaluate_retriever(
        BM25Retriever(chunks),
        chunks,
        questions,
        name="bm25",
    )

    assert report.queries[0].relevant_chunk_ids == (
        "gold-1",
        "gold-2",
    )
    assert report.metrics.recall_at_5 == 1.0


class _FixedRetriever:
    def __init__(self, results: dict[str, list[RankedChunk]]) -> None:
        self.results = results

    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        return self.results.get(query, [])[:top_k]


def test_unanswerable_questions_are_excluded_from_top_level_metrics() -> None:
    chunks = (
        make_chunk("gold-1", "answer one"),
        make_chunk("gold-2", "answer two"),
        make_chunk("other", "unrelated"),
    )
    retriever = _FixedRetriever(
        {
            "one": [RankedChunk(chunks[0], 1.0, 1, "bm25")],
            "two": [RankedChunk(chunks[1], 1.0, 1, "bm25")],
            "none": [RankedChunk(chunks[2], 1.0, 1, "bm25")],
        }
    )
    questions = (
        EvaluationQuestion(
            "Q1", "answerable", "one", True, frozenset({"gold-1"}), {"gold-1": 3}
        ),
        EvaluationQuestion(
            "Q2", "answerable", "two", True, frozenset({"gold-2"}), {"gold-2": 3}
        ),
        EvaluationQuestion("Q3", "unanswerable", "none", False, frozenset(), {}),
    )

    report = evaluate_retriever(retriever, chunks, questions, name="fixed")

    assert report.metrics.query_count == 2
    assert report.metrics.recall_at_1 == 1.0
    assert report.metrics.mrr == 1.0
    assert report.metrics.ndcg_at_5 == 1.0
    assert report.by_category["answerable"] == report.metrics
    assert report.by_category["unanswerable"].query_count == 1
    assert report.by_category["unanswerable"].recall_at_1 == 0.0


def test_ndcg_uses_explicit_graded_relevance_and_actual_query_ideal() -> None:
    chunks = (
        make_chunk("primary", "primary"),
        make_chunk("acceptable", "acceptable"),
        make_chunk("other", "other"),
    )
    retriever = _FixedRetriever(
        {
            "primary-first": [
                RankedChunk(chunks[0], 3.0, 1, "bm25"),
                RankedChunk(chunks[1], 2.0, 2, "bm25"),
            ],
            "acceptable-first": [
                RankedChunk(chunks[1], 3.0, 1, "bm25"),
                RankedChunk(chunks[0], 2.0, 2, "bm25"),
            ],
            "primary-second": [
                RankedChunk(chunks[2], 3.0, 1, "bm25"),
                RankedChunk(chunks[0], 3.0, 2, "bm25"),
                RankedChunk(chunks[1], 2.0, 3, "bm25"),
            ],
            "none": [RankedChunk(chunks[2], 1.0, 1, "bm25")],
        }
    )
    gold = {"primary": 3, "acceptable": 2}

    def evaluate(query: str):
        return evaluate_retriever(
            retriever,
            chunks,
            (
                EvaluationQuestion(
                    query,
                    "answerable",
                    query,
                    True,
                    frozenset(gold),
                    gold,
                ),
            ),
            name="fixed",
        )

    first = evaluate("primary-first")
    acceptable_first = evaluate("acceptable-first")
    primary_second = evaluate("primary-second")

    assert first.queries[0].graded_relevance == gold
    assert first.metrics.ndcg_at_5 == pytest.approx(1.0)
    assert acceptable_first.metrics.ndcg_at_5 < first.metrics.ndcg_at_5
    assert primary_second.metrics.ndcg_at_5 < first.metrics.ndcg_at_5

    no_hit = evaluate_retriever(
        retriever,
        chunks,
        (
            EvaluationQuestion(
                "Q-none",
                "answerable",
                "none",
                True,
                frozenset({"primary"}),
                {"primary": 3},
            ),
        ),
        name="fixed",
    )
    assert no_hit.metrics.ndcg_at_5 == 0.0


def test_duplicate_gold_ids_keep_the_highest_grade(tmp_path) -> None:
    path = tmp_path / "gold.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": "Q1",
                    "question": "question",
                    "answerable": True,
                    "m2_gold_evidence": [
                        {"m2_chunk_id": "chunk", "grade": 2},
                        {"m2_chunk_id": "chunk", "grade": 3},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    assert load_questions(path)[0].graded_relevance == {"chunk": 3}


def test_missing_gold_ids_fail_before_retrieval() -> None:
    chunks = (make_chunk("present", "content"),)
    question = EvaluationQuestion(
        "Q-missing",
        "answerable",
        "query",
        True,
        frozenset({"missing"}),
        {"missing": 3},
    )

    with pytest.raises(ValueError, match=r"Q-missing.*missing"):
        evaluate_retriever(BM25Retriever(chunks), chunks, (question,), name="bm25")


def test_binary_relevance_is_derived_from_explicit_grades() -> None:
    chunks = (
        make_chunk("binary-only", "related text"),
        make_chunk("graded", "graded text"),
    )
    retriever = _FixedRetriever(
        {
            "query": [
                RankedChunk(chunks[0], 2.0, 1, "bm25"),
                RankedChunk(chunks[1], 1.0, 2, "bm25"),
            ]
        }
    )
    question = EvaluationQuestion(
        "Q-disagreement",
        "answerable",
        "query",
        True,
        frozenset({"binary-only", "graded"}),
        {"graded": 3},
    )

    report = evaluate_retriever(retriever, chunks, (question,), name="fixed")

    assert report.queries[0].relevant_chunk_ids == ("binary-only", "graded")
    assert report.metrics.recall_at_1 == 0.0
    assert report.metrics.mrr == 0.5
