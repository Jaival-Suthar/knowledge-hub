"""Post-hoc complementarity analysis for Dense, BM25, and Hybrid reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

DEFAULT_DENSE_REPORT = Path("evaluation/reports/dense.json")
DEFAULT_BM25_REPORT = Path("evaluation/reports/bm25.json")
DEFAULT_HYBRID_REPORT = Path("evaluation/reports/hybrid.json")
DEFAULT_OUTPUT = Path("evaluation/reports/hybrid_complementarity.json")
TOP_K = 20


def load_report(path: Path) -> dict[str, Any]:
    """Load one existing benchmark report artifact."""
    with path.open(encoding="utf-8") as stream:
        report = json.load(stream)
    if not isinstance(report, dict) or not isinstance(report.get("queries"), list):
        raise TypeError(f"invalid benchmark report: {path}")
    return report


def analyze_reports(
    dense_report: dict[str, Any],
    bm25_report: dict[str, Any],
    hybrid_report: dict[str, Any],
) -> dict[str, Any]:
    """Compare existing report candidates without running retrieval again."""
    reports = {
        "dense": _queries_by_id(dense_report, "dense"),
        "bm25": _queries_by_id(bm25_report, "bm25"),
        "hybrid": _queries_by_id(hybrid_report, "hybrid"),
    }
    _validate_compatibility(reports)

    observations = [
        _analyze_query(
            query_id,
            reports["dense"][query_id],
            reports["bm25"][query_id],
            reports["hybrid"][query_id],
        )
        for query_id in sorted(reports["dense"])
        if reports["dense"][query_id]["answerable"]
    ]
    return {
        "summary": _summary(observations),
        "queries": observations,
    }


def _queries_by_id(report: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    queries: dict[str, dict[str, Any]] = {}
    for query in report["queries"]:
        if not isinstance(query, dict) or not isinstance(query.get("id"), str):
            raise TypeError(f"{name} report contains a query without a string id")
        query_id = query["id"]
        if query_id in queries:
            raise ValueError(f"{name} report contains duplicate query id {query_id}")
        queries[query_id] = query
    return queries


def _validate_compatibility(
    reports: dict[str, dict[str, dict[str, Any]]],
) -> None:
    reference = reports["dense"]
    for name, queries in reports.items():
        if set(queries) != set(reference):
            raise ValueError(f"benchmark reports have mismatched query IDs ({name})")
        for query_id in sorted(reference):
            expected = reference[query_id]
            actual = queries[query_id]
            if bool(actual.get("answerable")) != bool(expected.get("answerable")):
                raise ValueError(f"answerable set differs for query {query_id}")
            if _effective_gold(actual) != _effective_gold(expected):
                raise ValueError(f"gold evidence differs for query {query_id}")


def _analyze_query(
    query_id: str,
    dense: dict[str, Any],
    bm25: dict[str, Any],
    hybrid: dict[str, Any],
) -> dict[str, Any]:
    gold = sorted(_effective_gold(dense))
    dense_view = _retriever_view(dense, include_scores=False, gold=gold)
    bm25_view = _retriever_view(bm25, include_scores=False, gold=gold)
    hybrid_view = _retriever_view(hybrid, include_scores=True, gold=gold)

    dense_ids = set(dense_view["chunk_ids"])
    bm25_ids = set(bm25_view["chunk_ids"])
    dense_hit = dense_view["hit"]
    bm25_hit = bm25_view["hit"]
    hybrid_hit = hybrid_view["hit"]
    intersection = dense_ids & bm25_ids
    union = dense_ids | bm25_ids
    classification = (
        "both"
        if dense_hit and bm25_hit
        else "dense_only"
        if dense_hit
        else "bm25_only"
        if bm25_hit
        else "neither"
    )

    return {
        "id": query_id,
        "question": dense.get("question"),
        "gold_chunk_ids": gold,
        "dense": dense_view,
        "bm25": bm25_view,
        "hybrid": hybrid_view,
        "overlap": {
            "dense_count": len(dense_ids),
            "bm25_count": len(bm25_ids),
            "intersection_count": len(intersection),
            "union_count": len(union),
            "jaccard": len(intersection) / len(union) if union else 0.0,
            "dense_unique_count": len(dense_ids - bm25_ids),
            "bm25_unique_count": len(bm25_ids - dense_ids),
        },
        "classification": classification,
        "bm25_rescued_dense": not dense_hit and bm25_hit and hybrid_hit,
        "dense_rescued_bm25": not bm25_hit and dense_hit and hybrid_hit,
    }


def _effective_gold(query: dict[str, Any]) -> set[str]:
    graded_relevance = query.get("graded_relevance")
    if graded_relevance:
        return {chunk_id for chunk_id, grade in graded_relevance.items() if grade >= 2}
    return set(query.get("relevant_chunk_ids", []))


def _retriever_view(
    query: dict[str, Any],
    *,
    include_scores: bool,
    gold: list[str],
) -> dict[str, Any]:
    chunk_ids = list(query.get("result_chunk_ids", []))[:TOP_K]
    ranks = list(range(1, len(chunk_ids) + 1))
    view: dict[str, Any] = {
        "hit": bool(set(chunk_ids) & set(gold)),
        "gold_rank": _gold_rank(chunk_ids, gold),
        "chunk_ids": chunk_ids,
        "ranks": ranks,
    }
    if include_scores:
        scores = list(query.get("result_scores", []))[:TOP_K]
        view["results"] = [
            {
                "chunk_id": chunk_id,
                "rank": rank,
                "score": scores[index] if index < len(scores) else None,
            }
            for index, (chunk_id, rank) in enumerate(zip(chunk_ids, ranks))
        ]
    return view


def _gold_rank(chunk_ids: list[str], gold: list[str]) -> int | None:
    gold_ids = set(gold)
    return next(
        (
            rank
            for rank, chunk_id in enumerate(chunk_ids, start=1)
            if chunk_id in gold_ids
        ),
        None,
    )


def _summary(observations: list[dict[str, Any]]) -> dict[str, Any]:
    classifications = {
        name: 0 for name in ("dense_only", "bm25_only", "both", "neither")
    }
    for observation in observations:
        classifications[observation["classification"]] += 1
    intersections = [item["overlap"]["intersection_count"] for item in observations]
    jaccards = [item["overlap"]["jaccard"] for item in observations]
    dense_unique = [item["overlap"]["dense_unique_count"] for item in observations]
    bm25_unique = [item["overlap"]["bm25_unique_count"] for item in observations]

    def average(values: list[int | float]) -> float:
        return float(mean(values)) if values else 0.0

    def middle(values: list[int | float]) -> float:
        return float(median(values)) if values else 0.0

    return {
        "total_answerable_queries": len(observations),
        "classification_counts": classifications,
        "bm25_rescued_dense_count": sum(
            item["bm25_rescued_dense"] for item in observations
        ),
        "dense_rescued_bm25_count": sum(
            item["dense_rescued_bm25"] for item in observations
        ),
        "mean_intersection_count": average(intersections),
        "median_intersection_count": middle(intersections),
        "mean_jaccard": average(jaccards),
        "median_jaccard": middle(jaccards),
        "mean_dense_unique_count": average(dense_unique),
        "mean_bm25_unique_count": average(bm25_unique),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-report", type=Path, default=DEFAULT_DENSE_REPORT)
    parser.add_argument("--bm25-report", type=Path, default=DEFAULT_BM25_REPORT)
    parser.add_argument("--hybrid-report", type=Path, default=DEFAULT_HYBRID_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    analysis = analyze_reports(
        load_report(args.dense_report),
        load_report(args.bm25_report),
        load_report(args.hybrid_report),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
