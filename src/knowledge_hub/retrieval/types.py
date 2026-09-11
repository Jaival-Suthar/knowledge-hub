from __future__ import annotations

from dataclasses import dataclass, field

from knowledge_hub.models import Chunk


@dataclass(frozen=True)
class RankedChunk:
    chunk: Chunk
    score: float
    rank: int
    channel: str


@dataclass
class RetrievalTrace:
    query: str
    dense_results: list[RankedChunk] = field(default_factory=list)
    bm25_results: list[RankedChunk] = field(default_factory=list)
    fusion_results: list[RankedChunk] = field(default_factory=list)
    filtered_results: list[RankedChunk] = field(default_factory=list)
    reranked_results: list[RankedChunk] = field(default_factory=list)
    final_evidence: list[RankedChunk] = field(default_factory=list)
