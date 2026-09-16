"""Shared evaluation helpers for dense and sparse retrievers."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from math import log2
from time import perf_counter

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.types import RankedChunk


@dataclass(frozen=True)
class EvaluationQuestion:
    """A retrieval evaluation question with explicit M2 gold evidence."""

    id: str
    category: str
    question: str
    answerable: bool
    gold_chunk_ids: frozenset[str]
    graded_relevance: dict[str, int]


@dataclass(frozen=True)
class RetrievalMetrics:
    """Evaluation metrics; recall_at_k fields are binary hit@k measures."""

    query_count: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float
    ndcg_at_5: float
    latency_ms: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QueryEvaluation:
    id: str
    category: str
    question: str
    answerable: bool
    relevant_chunk_ids: tuple[str, ...]
    graded_relevance: dict[str, int]
    result_chunk_ids: tuple[str, ...]
    result_scores: tuple[float, ...]
    latency_ms: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationReport:
    retriever: str
    metrics: RetrievalMetrics
    by_category: dict[str, RetrievalMetrics]
    queries: tuple[QueryEvaluation, ...]
    rrf_k: int | None = None

    def as_dict(self) -> dict[str, object]:
        report = {
            "retriever": self.retriever,
            "metrics": self.metrics.as_dict(),
            "by_category": {
                category: metrics.as_dict()
                for category, metrics in self.by_category.items()
            },
            "queries": [query.as_dict() for query in self.queries],
        }
        if self.rrf_k is not None:
            report["rrf_k"] = self.rrf_k
        return report


class Searcher:
    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        raise NotImplementedError


def validate_gold_chunk_ids(
    chunks: Sequence[Chunk],
    questions: Iterable[EvaluationQuestion],
) -> None:
    """Fail clearly when gold evidence does not belong to the current corpus."""
    available_ids = {chunk.chunk_id for chunk in chunks}
    missing_by_question = {
        question.id: sorted(
            (set(question.gold_chunk_ids) | set(question.graded_relevance))
            - available_ids
        )
        for question in questions
        if (set(question.gold_chunk_ids) | set(question.graded_relevance))
        - available_ids
    }
    if missing_by_question:
        details = "; ".join(
            f"{question_id}: {', '.join(chunk_ids)}"
            for question_id, chunk_ids in sorted(missing_by_question.items())
        )
        raise ValueError(f"gold chunk IDs missing from corpus ({details})")


def evaluate_retriever(
    retriever: Searcher,
    chunks: Sequence[Chunk],
    questions: Iterable[EvaluationQuestion],
    *,
    name: str,
    top_k: int = 20,
) -> EvaluationReport:
    """Evaluate a retriever against explicit M2 gold chunk IDs."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    question_list = tuple(questions)
    validate_gold_chunk_ids(chunks, question_list)

    query_evaluations = tuple(
        _evaluate_question(retriever, question, top_k) for question in question_list
    )
    answerable_evaluations = tuple(
        evaluation for evaluation in query_evaluations if evaluation.answerable
    )
    grouped_evaluations = _group_by_category(
        question_list,
        query_evaluations,
    )

    return EvaluationReport(
        retriever=name,
        metrics=_aggregate(answerable_evaluations),
        by_category={
            category: _aggregate(
                category_observations,
                include_retrieval_metrics=any(
                    evaluation.answerable for evaluation in category_observations
                ),
            )
            for category, category_observations in grouped_evaluations.items()
        },
        queries=query_evaluations,
    )


def _evaluate_question(
    retriever: Searcher,
    question: EvaluationQuestion,
    top_k: int,
) -> QueryEvaluation:
    """Evaluate one question using only its explicit gold chunk IDs."""
    relevant_ids = set(question.gold_chunk_ids)

    started = perf_counter()
    results = retriever.search(question.question, top_k)
    elapsed_ms = (perf_counter() - started) * 1000.0

    return QueryEvaluation(
        id=question.id,
        category=question.category,
        question=question.question,
        answerable=question.answerable,
        relevant_chunk_ids=tuple(sorted(relevant_ids)),
        graded_relevance=_normalized_grades(question.graded_relevance),
        result_chunk_ids=tuple(result.chunk.chunk_id for result in results),
        result_scores=tuple(float(result.score) for result in results),
        latency_ms=elapsed_ms,
    )


def _group_by_category(
    questions: Sequence[EvaluationQuestion],
    evaluations: Sequence[QueryEvaluation],
) -> dict[str, list[QueryEvaluation]]:
    grouped: dict[str, list[QueryEvaluation]] = defaultdict(list)

    for question, evaluation in zip(
        questions,
        evaluations,
        strict=True,
    ):
        grouped[question.category].append(evaluation)

    return dict(sorted(grouped.items()))


def _aggregate(
    evaluations: Sequence[QueryEvaluation],
    *,
    include_retrieval_metrics: bool = True,
) -> RetrievalMetrics:
    if not evaluations:
        return RetrievalMetrics(
            query_count=0,
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            recall_at_20=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            latency_ms=0.0,
        )

    if not include_retrieval_metrics:
        return RetrievalMetrics(
            query_count=len(evaluations),
            recall_at_1=0.0,
            recall_at_5=0.0,
            recall_at_10=0.0,
            recall_at_20=0.0,
            mrr=0.0,
            ndcg_at_5=0.0,
            latency_ms=_mean(evaluation.latency_ms for evaluation in evaluations),
        )

    return RetrievalMetrics(
        query_count=len(evaluations),
        recall_at_1=_mean(
            _recall(_effective_relevant_ids(evaluation), evaluation.result_chunk_ids, 1)
            for evaluation in evaluations
        ),
        recall_at_5=_mean(
            _recall(_effective_relevant_ids(evaluation), evaluation.result_chunk_ids, 5)
            for evaluation in evaluations
        ),
        recall_at_10=_mean(
            _recall(
                _effective_relevant_ids(evaluation),
                evaluation.result_chunk_ids,
                10,
            )
            for evaluation in evaluations
        ),
        recall_at_20=_mean(
            _recall(
                _effective_relevant_ids(evaluation),
                evaluation.result_chunk_ids,
                20,
            )
            for evaluation in evaluations
        ),
        mrr=_mean(
            _mrr(
                _effective_relevant_ids(evaluation),
                evaluation.result_chunk_ids,
            )
            for evaluation in evaluations
        ),
        ndcg_at_5=_mean(
            _ndcg(
                evaluation.graded_relevance,
                evaluation.result_chunk_ids,
                5,
            )
            for evaluation in evaluations
        ),
        latency_ms=_mean(evaluation.latency_ms for evaluation in evaluations),
    )


def _recall(
    relevant_ids: Sequence[str],
    result_ids: Sequence[str],
    k: int,
) -> float:
    """Return whether at least one relevant chunk appears in top-k."""
    return float(bool(set(relevant_ids).intersection(result_ids[:k])))


def _mrr(
    relevant_ids: Sequence[str],
    result_ids: Sequence[str],
) -> float:
    relevant = set(relevant_ids)

    for rank, chunk_id in enumerate(result_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank

    return 0.0


def _ndcg(
    graded_relevance: dict[str, int],
    result_ids: Sequence[str],
    k: int,
) -> float:
    """Compute graded nDCG@k from explicit gold evidence only."""
    grades = _normalized_grades(graded_relevance)
    if not grades:
        return 0.0

    dcg = sum(
        (2 ** grades[chunk_id] - 1) / log2(rank + 1)
        for rank, chunk_id in enumerate(result_ids[:k], start=1)
        if chunk_id in grades
    )
    ideal = sum(
        (2**grade - 1) / log2(rank + 1)
        for rank, grade in enumerate(sorted(grades.values(), reverse=True)[:k], start=1)
    )

    return dcg / ideal if ideal else 0.0


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)

    if not values_list:
        return 0.0

    return sum(values_list) / len(values_list)


def _normalized_grades(grades: dict[str, int]) -> dict[str, int]:
    """Keep the highest explicit grade when gold evidence IDs repeat."""
    normalized: dict[str, int] = {}
    for chunk_id, grade in grades.items():
        if grade > 0:
            normalized[chunk_id] = max(normalized.get(chunk_id, 0), int(grade))
    return normalized


def _effective_relevant_ids(evaluation: QueryEvaluation) -> set[str]:
    """Return explicitly graded relevant evidence for hit@k and MRR."""
    return {
        chunk_id
        for chunk_id, grade in evaluation.graded_relevance.items()
        if grade >= 2
    }
