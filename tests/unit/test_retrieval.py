from knowledge_hub.models import Chunk, SourceType
from knowledge_hub.retrieval.fusion import ReciprocalRankFusion
from knowledge_hub.retrieval.types import RankedChunk


def chunk(chunk_id: str) -> Chunk:
    return Chunk(
        document_id="doc",
        chunk_id=chunk_id,
        content=chunk_id,
        source_type=SourceType.MARKDOWN,
    )


def test_rrf_combines_rankings() -> None:
    a, b, c = (chunk("a"), chunk("b"), chunk("c"))
    dense = [
        RankedChunk(a, 0.9, 1, "dense"),
        RankedChunk(b, 0.8, 2, "dense"),
    ]
    sparse = [
        RankedChunk(c, 4.0, 1, "bm25"),
        RankedChunk(a, 3.0, 2, "bm25"),
    ]

    results = ReciprocalRankFusion(k=60).fuse([dense, sparse])
    assert results[0].chunk.chunk_id == "a"
    assert results[0].channel == "rrf"
