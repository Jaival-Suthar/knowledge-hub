"""Thin API-facing adapters over the existing retrieval contracts."""

from __future__ import annotations

from collections.abc import Iterable

from knowledge_hub.config.settings import Settings
from knowledge_hub.indexing.embedding import embed_and_upsert
from knowledge_hub.indexing.qdrant import QdrantIndex
from knowledge_hub.inference.embedder import Embedder, SentenceTransformerEmbedder
from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.bm25 import BM25Retriever
from knowledge_hub.retrieval.dense import QdrantDenseRetriever
from knowledge_hub.retrieval.metadata import MetadataFilters
from knowledge_hub.retrieval.pipeline import RetrievalPipeline
from knowledge_hub.retrieval.types import RankedChunk

from .contracts import MetadataFiltersRequest, SearchRequest


class ChunksNotFoundError(LookupError):
    """Raised when a direct lookup contains an unknown canonical chunk ID."""

    def __init__(self, chunk_ids: Iterable[str]) -> None:
        self.chunk_ids = tuple(sorted(set(chunk_ids)))
        super().__init__(f"Chunks not found: {', '.join(self.chunk_ids)}")


class KnowledgeRetrievalService:
    """Expose the existing retrieval pipeline and canonical chunk collection."""

    def __init__(
        self,
        pipeline: RetrievalPipeline,
        chunks: Iterable[Chunk],
        *,
        index: QdrantIndex | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.chunks: dict[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}
        self._index = index
        self._embedder = embedder
        self._persisted_chunk_ids = set(self.chunks)

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        """Add canonical chunks to lookup, sparse, and dense retrieval state."""
        incoming = tuple(chunks)
        new_chunks = tuple(
            chunk
            for chunk in incoming
            if chunk.chunk_id not in self._persisted_chunk_ids
        )
        if new_chunks and self._index is not None and self._embedder is not None:
            embed_and_upsert(self._index, new_chunks, self._embedder)
            self._persisted_chunk_ids.update(chunk.chunk_id for chunk in new_chunks)

        self.chunks.update({chunk.chunk_id: chunk for chunk in incoming})
        self.pipeline.sparse.update(self.chunks.values())

    def search(self, request: SearchRequest) -> list[RankedChunk]:
        """Run the existing pipeline with API request values."""
        return self.pipeline.search(
            query=request.query,
            dense_k=request.top_k,
            sparse_k=request.top_k,
            rerank_k=request.top_k,
            metadata_filters=to_metadata_filters(request.metadata_filters),
        ).final_evidence

    def retrieve(self, chunk_ids: Iterable[str]) -> list[Chunk]:
        """Return canonical chunks in request order, or fail deterministically."""
        requested_ids = list(chunk_ids)
        missing_ids = [
            chunk_id for chunk_id in requested_ids if chunk_id not in self.chunks
        ]
        if missing_ids:
            raise ChunksNotFoundError(missing_ids)
        return [self.chunks[chunk_id] for chunk_id in requested_ids]


def build_retrieval_service(
    config: Settings,
    chunks: Iterable[Chunk] = (),
) -> KnowledgeRetrievalService:
    """Build the existing configured dense+sparse retrieval stack lazily."""
    embedder = SentenceTransformerEmbedder(
        model_name=config.embedding_model_name,
        dimension=config.embedding_dimension,
        device=config.embedding_device,
        normalize_embeddings=config.embedding_normalize,
    )
    index = QdrantIndex(
        url=config.qdrant_url,
        collection=config.qdrant_collection,
        vector_size=config.embedding_dimension,
    )
    dense = QdrantDenseRetriever(
        client=index.client,
        collection=index.collection,
        embedder=embedder,
    )
    pipeline = RetrievalPipeline(dense, BM25Retriever(chunks))
    return KnowledgeRetrievalService(
        pipeline,
        chunks,
        index=index,
        embedder=embedder,
    )


def to_metadata_filters(
    request: MetadataFiltersRequest | None,
) -> MetadataFilters | None:
    """Translate the HTTP wrapper into the existing domain filter contract."""
    if request is None:
        return None

    values = request.model_dump(exclude_none=True)
    return MetadataFilters(
        **{
            name: tuple(value) if isinstance(value, list) else value
            for name, value in values.items()
        }
    )
