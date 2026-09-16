from __future__ import annotations

from knowledge_hub.retrieval.rrf import reciprocal_rank_fusion
from knowledge_hub.retrieval.types import RankedChunk


class ReciprocalRankFusion:
    """Compatibility wrapper for the pure Reciprocal Rank Fusion function."""

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("RRF k must be positive")
        self.k = k

    def fuse(
        self, ranked_lists: list[list[RankedChunk]], top_k: int | None = None
    ) -> list[RankedChunk]:
        return reciprocal_rank_fusion(ranked_lists, k=self.k, top_k=top_k)
