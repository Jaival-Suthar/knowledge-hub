"""Structured timings for meaningful retrieval stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalTiming:
    """Optional component timings in milliseconds."""

    dense_ms: float | None = None
    embedding_ms: float | None = None
    qdrant_ms: float | None = None
    bm25_ms: float | None = None
    rrf_ms: float | None = None
    hybrid_total_ms: float | None = None

    def as_dict(self) -> dict[str, float]:
        return {
            name: value
            for name, value in {
                "dense_ms": self.dense_ms,
                "embedding_ms": self.embedding_ms,
                "qdrant_ms": self.qdrant_ms,
                "bm25_ms": self.bm25_ms,
                "rrf_ms": self.rrf_ms,
                "hybrid_total_ms": self.hybrid_total_ms,
            }.items()
            if value is not None
        }
