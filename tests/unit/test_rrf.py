from __future__ import annotations

import pytest

from knowledge_hub.models import Chunk, Provenance, SourceType
from knowledge_hub.retrieval import ReciprocalRankFusion, reciprocal_rank_fusion
from knowledge_hub.retrieval.types import RankedChunk


def make_result(
    chunk_id: str,
    rank: int,
    *,
    score: float = 0.0,
    metadata: dict[str, str] | None = None,
    provenance: Provenance | None = None,
) -> RankedChunk:
    chunk = Chunk(
        document_id=f"document-{chunk_id}",
        chunk_id=chunk_id,
        content=f"content for {chunk_id}",
        source_type=SourceType.CODE,
        source_uri=f"https://example.test/{chunk_id}",
        language="python",
        metadata=metadata or {},
        provenance=provenance
        or Provenance(source_uri=f"https://example.test/{chunk_id}"),
    )
    return RankedChunk(chunk=chunk, score=score, rank=rank, channel="source")


def ids(results: list[RankedChunk]) -> list[str]:
    return [result.chunk.chunk_id for result in results]


def test_single_ranked_list_uses_one_based_rank_contributions() -> None:
    results = reciprocal_rank_fusion(
        [[make_result("A", 1), make_result("B", 2), make_result("C", 3)]],
        k=60,
    )

    assert ids(results) == ["A", "B", "C"]
    assert [result.score for result in results] == pytest.approx(
        [1 / 61, 1 / 62, 1 / 63]
    )
    assert [result.rank for result in results] == [1, 2, 3]


def test_overlapping_lists_accumulate_rank_contributions() -> None:
    results = reciprocal_rank_fusion(
        [
            [make_result("A", 1), make_result("B", 2), make_result("C", 3)],
            [make_result("C", 1), make_result("A", 2), make_result("D", 3)],
        ],
        k=60,
    )

    assert ids(results) == ["A", "C", "B", "D"]
    assert [result.score for result in results] == pytest.approx(
        [1 / 61 + 1 / 62, 1 / 61 + 1 / 63, 1 / 62, 1 / 63]
    )


def test_unique_results_are_retained_with_their_single_contribution() -> None:
    results = reciprocal_rank_fusion(
        [[make_result("A", 1)], [make_result("B", 1), make_result("C", 2)]],
        k=10,
    )

    assert ids(results) == ["A", "B", "C"]
    assert {result.chunk.chunk_id: result.score for result in results} == pytest.approx(
        {"A": 1 / 11, "B": 1 / 11, "C": 1 / 12}
    )


def test_duplicate_chunk_ids_are_emitted_once_and_summed() -> None:
    results = reciprocal_rank_fusion(
        [[make_result("same", 1)], [make_result("same", 2)]], k=60
    )

    assert ids(results) == ["same"]
    assert results[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_raw_retrieval_scores_do_not_affect_fusion() -> None:
    first = reciprocal_rank_fusion(
        [[make_result("A", 1, score=0.1), make_result("B", 2, score=999.0)]]
    )
    second = reciprocal_rank_fusion(
        [[make_result("A", 1, score=-500.0), make_result("B", 2, score=0.001)]]
    )

    assert ids(first) == ids(second)
    assert [result.score for result in first] == pytest.approx(
        [result.score for result in second]
    )


def test_top_k_limits_results_and_renumbers_ranks() -> None:
    ranked = [[make_result(chunk_id, index) for index, chunk_id in enumerate("ABC", 1)]]

    assert len(reciprocal_rank_fusion(ranked, top_k=None)) == 3
    limited = reciprocal_rank_fusion(ranked, top_k=2)
    assert ids(limited) == ["A", "B"]
    assert [result.rank for result in limited] == [1, 2]


@pytest.mark.parametrize("top_k", [0, -1])
def test_non_positive_top_k_returns_no_results(top_k: int) -> None:
    ranked = [[make_result(chunk_id, index) for index, chunk_id in enumerate("ABC", 1)]]

    assert reciprocal_rank_fusion(ranked, top_k=top_k) == []


def test_top_k_none_returns_all_results() -> None:
    ranked = [[make_result(chunk_id, index) for index, chunk_id in enumerate("ABC", 1)]]

    assert ids(reciprocal_rank_fusion(ranked, top_k=None)) == ["A", "B", "C"]


def test_empty_inputs_return_empty_results() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.parametrize("invalid_k", [0, -1])
def test_non_positive_k_is_rejected(invalid_k: int) -> None:
    with pytest.raises(ValueError, match="RRF k must be positive"):
        reciprocal_rank_fusion([], k=invalid_k)


def test_equal_scores_use_chunk_id_as_deterministic_tie_breaker() -> None:
    results = reciprocal_rank_fusion(
        [[make_result("B", 1)], [make_result("A", 1)]], k=60
    )

    assert ids(results) == ["A", "B"]


def test_metadata_and_provenance_are_preserved() -> None:
    provenance = Provenance(
        source_uri="https://example.test/article",
        repository="org/repo",
        path="src/auth.py",
        symbol="AuthService.login",
        line_start=10,
        line_end=20,
        extra={"origin": "fixture"},
    )
    result = make_result("auth", 1, metadata={"role": "method"}, provenance=provenance)

    fused = reciprocal_rank_fusion([[result]])[0]

    assert isinstance(fused, RankedChunk)
    assert fused.chunk is result.chunk
    assert fused.chunk.metadata == {"role": "method"}
    assert fused.chunk.provenance == provenance
    assert fused.channel == "rrf"
    assert fused.rank == 1


def test_compatibility_wrapper_matches_public_function() -> None:
    ranked = [[make_result("A", 1), make_result("B", 2)]]

    expected = reciprocal_rank_fusion(ranked, k=60)
    actual = ReciprocalRankFusion(k=60).fuse(ranked)

    assert ids(actual) == ids(expected)
    assert [result.score for result in actual] == pytest.approx(
        [result.score for result in expected]
    )
