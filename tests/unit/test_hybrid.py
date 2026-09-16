from __future__ import annotations

import pytest

from knowledge_hub.models import Chunk, Provenance, SourceType
from knowledge_hub.retrieval import HybridRetriever, reciprocal_rank_fusion
from knowledge_hub.retrieval.types import RankedChunk


class FakeRetriever:
    def __init__(self, results: list[RankedChunk]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int) -> list[RankedChunk]:
        self.calls.append((query, top_k))
        return self.results


def result(
    chunk_id: str,
    rank: int,
    *,
    score: float = 1.0,
    metadata: dict[str, str] | None = None,
    provenance: Provenance | None = None,
) -> RankedChunk:
    chunk = Chunk(
        document_id=f"doc-{chunk_id}",
        chunk_id=chunk_id,
        content=f"content {chunk_id}",
        source_type=SourceType.CODE,
        source_uri=f"https://example.test/{chunk_id}",
        metadata=metadata or {},
        provenance=provenance
        or Provenance(source_uri=f"https://example.test/{chunk_id}"),
    )
    return RankedChunk(chunk=chunk, score=score, rank=rank, channel="source")


def result_ids(results: list[RankedChunk]) -> list[str]:
    return [item.chunk.chunk_id for item in results]


def test_hybrid_calls_both_retrievers_with_shared_candidate_k() -> None:
    dense = FakeRetriever([result("A", 1), result("B", 2)])
    bm25 = FakeRetriever([result("C", 1), result("A", 2)])

    results = HybridRetriever(dense, bm25).search("auth query", top_k=5, candidate_k=20)

    assert dense.calls == [("auth query", 20)]
    assert bm25.calls == [("auth query", 20)]
    assert len(results) <= 5


def test_hybrid_matches_public_rrf_and_applies_final_top_k() -> None:
    dense_results = [result("A", 1), result("B", 2), result("C", 3)]
    bm25_results = [result("C", 1), result("A", 2), result("D", 3)]
    dense = FakeRetriever(dense_results)
    bm25 = FakeRetriever(bm25_results)

    actual = HybridRetriever(dense, bm25).search(top_k=2, candidate_k=20, query="q")
    expected = reciprocal_rank_fusion([dense_results, bm25_results], k=60, top_k=2)

    assert result_ids(actual) == result_ids(expected)
    assert [item.score for item in actual] == pytest.approx(
        [item.score for item in expected]
    )
    assert [item.rank for item in actual] == [1, 2]


def test_hybrid_forwards_configured_rrf_k() -> None:
    dense_results = [result("A", 1), result("B", 2)]
    bm25_results = [result("B", 1), result("A", 2)]
    hybrid = HybridRetriever(
        FakeRetriever(dense_results), FakeRetriever(bm25_results), rrf_k=20
    )

    actual = hybrid.search("q", top_k=10, candidate_k=10)
    expected = reciprocal_rank_fusion([dense_results, bm25_results], k=20, top_k=10)

    assert result_ids(actual) == result_ids(expected)
    assert [item.score for item in actual] == pytest.approx(
        [item.score for item in expected]
    )


@pytest.mark.parametrize(
    ("dense_results", "bm25_results", "expected"),
    [
        ([], [result("B", 1)], ["B"]),
        ([result("A", 1)], [], ["A"]),
        ([], [], []),
    ],
)
def test_hybrid_preserves_empty_result_behavior(
    dense_results: list[RankedChunk],
    bm25_results: list[RankedChunk],
    expected: list[str],
) -> None:
    hybrid = HybridRetriever(FakeRetriever(dense_results), FakeRetriever(bm25_results))

    assert result_ids(hybrid.search("q", top_k=5, candidate_k=5)) == expected


def test_hybrid_preserves_canonical_metadata_and_provenance() -> None:
    provenance = Provenance(
        source_uri="https://example.test/article",
        repository="org/repo",
        path="src/auth.py",
        symbol="AuthService.login",
    )
    source = result(
        "auth",
        1,
        metadata={"role": "method"},
        provenance=provenance,
    )
    fused = HybridRetriever(FakeRetriever([source]), FakeRetriever([])).search(
        "q", top_k=5, candidate_k=5
    )[0]

    assert fused.chunk is source.chunk
    assert fused.chunk.metadata == {"role": "method"}
    assert fused.chunk.provenance == provenance
    assert fused.channel == "rrf"
