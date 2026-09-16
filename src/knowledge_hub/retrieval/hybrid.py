"""Hybrid retrieval by fusing independent dense and BM25 rankings."""

from __future__ import annotations

from typing import Any

from knowledge_hub.retrieval.rrf import reciprocal_rank_fusion
from knowledge_hub.retrieval.types import RankedChunk


class HybridRetriever:
    """Compose two retrievers with rank-based Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: Any,
        bm25_retriever: Any,
        *,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("RRF k must be positive")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int,
    ) -> list[RankedChunk]:
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        return reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=top_k,
        )
