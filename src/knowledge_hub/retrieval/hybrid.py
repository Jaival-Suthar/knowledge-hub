"""Hybrid retrieval by fusing independent dense and BM25 rankings."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from knowledge_hub.retrieval.rrf import reciprocal_rank_fusion
from knowledge_hub.retrieval.timing import RetrievalTiming
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
        self.last_timing: RetrievalTiming | None = None

    def search(
        self,
        query: str,
        *,
        top_k: int,
        candidate_k: int,
    ) -> list[RankedChunk]:
        started = perf_counter()
        dense_started = perf_counter()
        dense_results = self.dense_retriever.search(query, top_k=candidate_k)
        dense_elapsed_ms = (perf_counter() - dense_started) * 1000.0

        bm25_started = perf_counter()
        bm25_results = self.bm25_retriever.search(query, top_k=candidate_k)
        bm25_ms = (perf_counter() - bm25_started) * 1000.0

        rrf_started = perf_counter()
        fused = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
            top_k=top_k,
        )
        rrf_ms = (perf_counter() - rrf_started) * 1000.0

        dense_timing = getattr(self.dense_retriever, "last_timing", None)
        self.last_timing = RetrievalTiming(
            dense_ms=(
                dense_timing.dense_ms
                if dense_timing is not None and dense_timing.dense_ms is not None
                else dense_elapsed_ms
            ),
            embedding_ms=(
                dense_timing.embedding_ms if dense_timing is not None else None
            ),
            qdrant_ms=dense_timing.qdrant_ms if dense_timing is not None else None,
            bm25_ms=bm25_ms,
            rrf_ms=rrf_ms,
            hybrid_total_ms=(perf_counter() - started) * 1000.0,
        )
        return fused
