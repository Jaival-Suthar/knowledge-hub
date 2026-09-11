from __future__ import annotations

from collections import defaultdict

from knowledge_hub.retrieval.types import RankedChunk


class ReciprocalRankFusion:
    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k must be positive")
        self.k = k

    def fuse(
        self, ranked_lists: list[list[RankedChunk]], top_k: int | None = None
    ) -> list[RankedChunk]:
        scores: dict[str, float] = defaultdict(float)
        chunks: dict[str, RankedChunk] = {}
        for results in ranked_lists:
            for item in results:
                scores[item.chunk.chunk_id] += 1.0 / (self.k + item.rank)
                chunks[item.chunk.chunk_id] = item

        ordered = sorted(scores, key=scores.get, reverse=True)
        if top_k is not None:
            ordered = ordered[:top_k]
        return [
            RankedChunk(
                chunk=chunks[chunk_id].chunk,
                score=scores[chunk_id],
                rank=rank,
                channel="rrf",
            )
            for rank, chunk_id in enumerate(ordered, start=1)
        ]
