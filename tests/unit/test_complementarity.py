from __future__ import annotations

import copy

import pytest

from knowledge_hub.evaluation.complementarity import analyze_reports


def make_report(queries: list[dict]) -> dict:
    return {"retriever": "fixture", "queries": queries}


def query(
    query_id: str,
    *,
    answerable: bool = True,
    gold: list[str] | None = None,
    results: list[str] | None = None,
    scores: list[float] | None = None,
    graded_relevance: dict[str, int] | None = None,
) -> dict:
    return {
        "id": query_id,
        "question": f"question {query_id}",
        "answerable": answerable,
        "relevant_chunk_ids": gold or [],
        "result_chunk_ids": results or [],
        "result_scores": scores or [],
        **(
            {"graded_relevance": graded_relevance}
            if graded_relevance is not None
            else {}
        ),
    }


def analyze(dense_queries, bm25_queries=None, hybrid_queries=None):
    dense = make_report(dense_queries)
    return analyze_reports(
        dense,
        make_report(bm25_queries or copy.deepcopy(dense_queries)),
        make_report(hybrid_queries or copy.deepcopy(dense_queries)),
    )


def test_classifies_queries_and_counts_rescue_cases() -> None:
    dense = [
        query("both", gold=["a"], results=["a"]),
        query("dense", gold=["b"], results=["b"]),
        query("bm25", gold=["c"], results=["x"]),
        query("neither", gold=["d"], results=["x"]),
        query("unanswerable", answerable=False, gold=[], results=[]),
    ]
    bm25 = [
        query("both", gold=["a"], results=["a"]),
        query("dense", gold=["b"], results=["x"]),
        query("bm25", gold=["c"], results=["c"]),
        query("neither", gold=["d"], results=["y"]),
        query("unanswerable", answerable=False),
    ]
    hybrid = [
        query("both", gold=["a"], results=["a"]),
        query("dense", gold=["b"], results=["b"]),
        query("bm25", gold=["c"], results=["x"]),
        query("neither", gold=["d"], results=["z"]),
        query("unanswerable", answerable=False),
    ]

    report = analyze(dense, bm25, hybrid)

    assert report["summary"]["total_answerable_queries"] == 4
    assert report["summary"]["classification_counts"] == {
        "dense_only": 1,
        "bm25_only": 1,
        "both": 1,
        "neither": 1,
    }


def test_bm25_and_dense_rescue_cases_are_independent() -> None:
    dense = [
        query("bm25-rescue", gold=["a"], results=["x"]),
        query("dense-rescue", gold=["b"], results=["b"]),
    ]
    bm25 = [
        query("bm25-rescue", gold=["a"], results=["a"]),
        query("dense-rescue", gold=["b"], results=["x"]),
    ]
    hybrid = [
        query("bm25-rescue", gold=["a"], results=["a"]),
        query("dense-rescue", gold=["b"], results=["b"]),
    ]

    report = analyze(dense, bm25, hybrid)

    assert report["summary"]["bm25_rescued_dense_count"] == 1
    assert report["summary"]["dense_rescued_bm25_count"] == 1


def test_overlap_and_best_gold_ranks_are_reported() -> None:
    dense = [
        query(
            "q", gold=["gold-a", "gold-b"], results=["shared", "gold-b", "dense-only"]
        )
    ]
    bm25 = [
        query("q", gold=["gold-a", "gold-b"], results=["shared", "bm25-only", "gold-a"])
    ]
    hybrid = [
        query(
            "q",
            gold=["gold-a", "gold-b"],
            results=["other", "gold-a"],
            scores=[0.4, 0.3],
        )
    ]

    observation = analyze(dense, bm25, hybrid)["queries"][0]

    assert observation["dense"]["gold_rank"] == 2
    assert observation["bm25"]["gold_rank"] == 3
    assert observation["hybrid"]["gold_rank"] == 2
    assert observation["overlap"] == {
        "dense_count": 3,
        "bm25_count": 3,
        "intersection_count": 1,
        "union_count": 5,
        "jaccard": 0.2,
        "dense_unique_count": 2,
        "bm25_unique_count": 2,
    }
    assert observation["hybrid"]["results"][1]["score"] == 0.3


def test_reports_require_matching_query_ids_and_gold() -> None:
    dense = [query("q", gold=["gold"], results=["gold"])]
    with pytest.raises(ValueError, match="mismatched query IDs"):
        analyze(dense, [query("other")], [query("q", gold=["gold"])])
    with pytest.raises(ValueError, match="gold evidence differs"):
        analyze(dense, [query("q", gold=["different"])], [query("q", gold=["gold"])])


def test_output_order_and_missing_hits_are_deterministic() -> None:
    dense = [
        query("b", gold=["gold"], results=[]),
        query("a", gold=["gold"], results=[]),
    ]
    report = analyze(
        dense,
        [query("b", gold=["gold"], results=[]), query("a", gold=["gold"], results=[])],
        [query("b", gold=["gold"], results=[]), query("a", gold=["gold"], results=[])],
    )

    assert [item["id"] for item in report["queries"]] == ["a", "b"]
    assert report["queries"][0]["dense"]["gold_rank"] is None


def test_graded_relevance_controls_gold_hits_and_ranks() -> None:
    dense = [
        query(
            "q",
            gold=["a", "b"],
            results=["b", "x"],
            graded_relevance={"a": 3, "b": 1},
        )
    ]
    bm25 = [
        query(
            "q",
            gold=["legacy-only"],
            results=["a"],
            graded_relevance={"a": 3, "b": 1},
        )
    ]
    hybrid = [
        query(
            "q",
            gold=["a", "b"],
            results=["b", "a"],
            graded_relevance={"a": 3, "b": 1},
        )
    ]

    observation = analyze(dense, bm25, hybrid)["queries"][0]

    assert observation["gold_chunk_ids"] == ["a"]
    assert observation["dense"]["hit"] is False
    assert observation["dense"]["gold_rank"] is None
    assert observation["bm25"]["hit"] is True
    assert observation["bm25"]["gold_rank"] == 1
    assert observation["hybrid"]["gold_rank"] == 2
    assert observation["classification"] == "bm25_only"
    assert observation["bm25_rescued_dense"] is True


def test_grade_two_and_three_are_effective_and_grade_one_is_not() -> None:
    dense = [
        query(
            "q",
            gold=["one", "two", "three"],
            results=["one", "two", "three"],
            graded_relevance={"one": 1, "two": 2, "three": 3},
        )
    ]
    report = analyze(dense)["queries"][0]

    assert report["gold_chunk_ids"] == ["three", "two"]
    assert report["dense"]["hit"] is True
    assert report["dense"]["gold_rank"] == 2


def test_effective_gold_mismatch_is_rejected() -> None:
    dense = [query("q", gold=["a"], graded_relevance={"a": 3})]
    bm25 = [query("q", gold=["a"], graded_relevance={"b": 3})]

    with pytest.raises(ValueError, match="gold evidence differs"):
        analyze(dense, bm25, dense)


def test_reports_without_grades_fall_back_to_relevant_chunk_ids() -> None:
    report = analyze(
        [query("q", gold=["a"], results=["a"])],
        [query("q", gold=["a"], results=[])],
        [query("q", gold=["a"], results=[])],
    )

    assert report["queries"][0]["gold_chunk_ids"] == ["a"]
    assert report["queries"][0]["dense"]["hit"] is True
