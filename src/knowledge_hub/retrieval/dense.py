from __future__ import annotations

from qdrant_client import QdrantClient

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.types import RankedChunk


class QdrantDenseRetriever:
    """Dense retrieval against the M2-owned Qdrant collection."""

    def __init__(self, client: QdrantClient, collection: str, embedder) -> None:
        self.client = client
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, top_k: int) -> list[RankedChunk]:
        if top_k <= 0:
            return []
        vector = self.embedder.embed([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
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
        return results
