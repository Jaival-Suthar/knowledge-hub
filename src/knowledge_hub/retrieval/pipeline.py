from dataclasses import dataclass, field

from knowledge_hub.models import Chunk


@dataclass
class RetrievalTrace:
    query: str
    dense_results: list[Chunk] = field(default_factory=list)
    bm25_results: list[Chunk] = field(default_factory=list)
    fusion_results: list[Chunk] = field(default_factory=list)
    filtered_results: list[Chunk] = field(default_factory=list)
    reranked_results: list[Chunk] = field(default_factory=list)
    final_evidence: list[Chunk] = field(default_factory=list)


class RetrievalPipeline:
    def search(self, query: str) -> RetrievalTrace:
        return RetrievalTrace(query=query)
