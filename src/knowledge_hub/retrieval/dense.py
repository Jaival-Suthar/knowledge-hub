from __future__ import annotations

from time import perf_counter

from qdrant_client import QdrantClient

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.timing import RetrievalTiming
from knowledge_hub.retrieval.types import RankedChunk


class QdrantDenseRetriever:
    """Dense retrieval against the M2-owned Qdrant collection."""

    def __init__(self, client: QdrantClient, collection: str, embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder
        self.last_timing: RetrievalTiming | None = None

    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        started = perf_counter()
        if top_k <= 0:
            self.last_timing = RetrievalTiming(dense_ms=0.0)
            return []

        embedding_started = perf_counter()
        vector = self.embedder.embed([query])[0]
        embedding_ms = (perf_counter() - embedding_started) * 1000.0

        qdrant_started = perf_counter()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        qdrant_ms = (perf_counter() - qdrant_started) * 1000.0
        points = response.points
        results: list[RankedChunk] = []
        for rank, point in enumerate(points, start=1):
            payload = dict(point.payload or {})
            chunk = Chunk.model_validate(payload)
            results.append(
                RankedChunk(
                    chunk=chunk, score=float(point.score), rank=rank, channel="dense"
                )
            )
        self.last_timing = RetrievalTiming(
            dense_ms=(perf_counter() - started) * 1000.0,
            embedding_ms=embedding_ms,
            qdrant_ms=qdrant_ms,
        )
        return results
