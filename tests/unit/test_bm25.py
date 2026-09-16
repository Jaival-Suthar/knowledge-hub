from __future__ import annotations

from collections.abc import Iterable

import pytest

from knowledge_hub.models import Chunk, Provenance, SourceType
from knowledge_hub.retrieval import BM25Index, BM25Retriever


def make_chunk(
    chunk_id: str,
    content: str,
    *,
    metadata: dict[str, str] | None = None,
    provenance: Provenance | None = None,
) -> Chunk:
    return Chunk(
        document_id=f"document-{chunk_id}",
        chunk_id=chunk_id,
        content=content,
        source_type=SourceType.MARKDOWN,
        source_uri=f"file:///docs/{chunk_id}.md",
        metadata=metadata or {},
        provenance=provenance or Provenance(source_uri=f"file:///docs/{chunk_id}.md"),
    )


def test_empty_corpus_and_empty_query_return_no_results() -> None:
    retriever = BM25Retriever()

    assert retriever.search("anything", top_k=5) == []
    assert retriever.search("   ", top_k=5) == []
    assert retriever.search("anything", top_k=0) == []


def test_relevant_chunk_ranks_above_irrelevant_chunk() -> None:
    retriever = BM25Retriever(
        [
            make_chunk("auth", "AuthService refreshToken rotates credentials."),
            make_chunk("unrelated", "The weather is pleasant today."),
        ]
    )

    results = retriever.search("refreshToken", top_k=10)

    assert [result.chunk.chunk_id for result in results] == ["auth"]
    assert results[0].rank == 1
    assert results[0].score > 0
    assert results[0].channel == "bm25"


def test_top_k_and_ranks_are_applied_after_deterministic_ordering() -> None:
    retriever = BM25Retriever(
        [
            make_chunk("b", "shared retrieval term"),
            make_chunk("a", "shared retrieval term"),
            make_chunk("c", "shared retrieval term"),
        ]
    )

    results = retriever.search("shared", top_k=2)

    assert [result.chunk.chunk_id for result in results] == ["a", "b"]
    assert [result.rank for result in results] == [1, 2]


def test_duplicate_terms_receive_more_lexical_weight() -> None:
    retriever = BM25Retriever(
        [
            make_chunk("repeated", "alpha alpha alpha"),
            make_chunk("single", "alpha beta"),
            make_chunk("other", "beta gamma"),
        ]
    )

    results = retriever.search("alpha", top_k=3)

    assert results[0].chunk.chunk_id == "repeated"


@pytest.mark.parametrize(
    ("query", "chunk_id"),
    [
        ("refreshToken", "refresh"),
        ("AuthService", "auth"),
        ("CandidatePipeline", "candidate"),
        ("knowledge_hub", "knowledge"),
        ("github_acquisition.py", "github"),
        ("TreeSitter", "tree"),
    ],
)
def test_technical_identifiers_remain_searchable(query: str, chunk_id: str) -> None:
    retriever = BM25Retriever(
        [
            make_chunk(
                chunk_id,
                "technical implementation",
                metadata={"symbol": query},
            ),
            make_chunk("other", "ordinary documentation"),
        ]
    )

    results = retriever.search(query, top_k=1)

    assert results[0].chunk.chunk_id == chunk_id


def test_metadata_and_provenance_are_preserved_in_results() -> None:
    source_uri = "https://example.test/repository"
    chunk = make_chunk(
        "source",
        "CandidatePipeline implementation",
        metadata={"path": "src/pipeline.py"},
        provenance=Provenance(
            source_uri=source_uri,
            repository="owner/repository",
            path="src/pipeline.py",
            symbol="CandidatePipeline",
        ),
    )

    result = BM25Retriever([chunk]).search("CandidatePipeline", top_k=1)[0]

    assert result.chunk is chunk
    assert result.chunk.metadata == {"path": "src/pipeline.py"}
    assert result.chunk.provenance == chunk.provenance


def test_no_lexical_match_returns_no_results() -> None:
    retriever = BM25Retriever([make_chunk("one", "retrieval architecture")])

    assert retriever.search("unseen qdrant term", top_k=10) == []


def test_repeated_builds_and_queries_are_deterministic() -> None:
    chunks = [
        make_chunk("two", "shared term"),
        make_chunk("one", "shared term"),
    ]
    retriever = BM25Retriever(chunks)

    first = retriever.search("shared", top_k=10)
    retriever.rebuild(chunk for chunk in chunks)
    second = retriever.search("shared", top_k=10)

    assert first == second


def test_update_replaces_the_indexed_corpus() -> None:
    retriever = BM25Retriever([make_chunk("old", "old terminology")])

    retriever.update([make_chunk("new", "new terminology")])

    assert retriever.search("old", top_k=10) == []
    assert [result.chunk.chunk_id for result in retriever.search("new", top_k=10)] == [
        "new"
    ]


def test_index_accepts_generators_and_larger_top_k() -> None:
    chunks: Iterable[Chunk] = (
        chunk
        for chunk in [
            make_chunk("one", "one term"),
            make_chunk("two", "two term"),
        ]
    )
    index = BM25Index(chunks)

    assert len(index.chunks) == 2
    assert len(BM25Retriever(index=index).search("term", top_k=10)) == 2
