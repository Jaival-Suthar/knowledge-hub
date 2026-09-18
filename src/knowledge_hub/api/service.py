"""Thin API-facing adapters over the existing retrieval contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from knowledge_hub.models import Chunk
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
    ) -> None:
        self.pipeline = pipeline
        self.chunks: Mapping[str, Chunk] = {chunk.chunk_id: chunk for chunk in chunks}

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
