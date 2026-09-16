"""Small orchestration boundary between local embeddings and Qdrant."""

from __future__ import annotations

from collections.abc import Iterable

from knowledge_hub.inference.embedder import Embedder
from knowledge_hub.models import Chunk

from .qdrant import QdrantIndex


def embed_and_upsert(
    index: QdrantIndex,
    chunks: Iterable[Chunk],
    embedder: Embedder,
) -> None:
    """Embed canonical chunks and store them in the configured Qdrant index."""
    chunk_list = list(chunks)
    embeddings = embedder.embed([chunk.content for chunk in chunk_list])

    if len(embeddings) != len(chunk_list):
        raise ValueError(
            f"embedder returned {len(embeddings)} vectors for {len(chunk_list)} chunks"
        )
    if any(len(embedding) != index.vector_size for embedding in embeddings):
        raise ValueError(
            "embedder returned a vector with unexpected dimension; "
            f"expected {index.vector_size}"
        )

    index.upsert(chunk_list, embeddings)
