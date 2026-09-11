from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from knowledge_hub.models import Chunk


class QdrantIndex:
    """Thin Qdrant index owned by M2; it does not import M1."""

    def __init__(self, url: str, collection: str, vector_size: int) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection
        self.vector_size = vector_size

    def ensure_collection(self) -> None:
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.vector_size, distance=Distance.COSINE
                ),
            )

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        self.ensure_collection()
        points = [
            PointStruct(
                id=str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                vector=embedding,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

    def delete(self, document_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id", match=MatchValue(value=document_id)
                    )
                ]
            ),
        )
