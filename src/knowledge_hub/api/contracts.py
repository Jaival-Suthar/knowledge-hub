"""Typed HTTP contracts for the Knowledge Hub API foundation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from knowledge_hub.models import Chunk, Document, Provenance, SourceType


class MetadataFiltersRequest(BaseModel):
    """Optional source-aware constraints accepted by future search handlers."""

    source_type: str | list[str] | None = None
    project: str | list[str] | None = None
    language: str | list[str] | None = None
    repository: str | list[str] | None = None
    path: str | list[str] | None = None
    content_role: str | list[str] | None = None


class SearchRequest(BaseModel):
    """Request contract for the future ``POST /search`` endpoint."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=20, gt=0)
    metadata_filters: MetadataFiltersRequest | None = None


class RetrieveRequest(BaseModel):
    """Request contract for the future ``POST /retrieve`` endpoint."""

    chunk_ids: list[str] = Field(min_length=1)


class IngestRequest(BaseModel):
    """Request contract for the future ``POST /ingest`` endpoint."""

    source: str = Field(min_length=1)
    source_type: SourceType | None = None


class RankedChunkResponse(BaseModel):
    """Serializable wrapper around an existing ranked canonical chunk."""

    chunk: Chunk
    score: float
    rank: int = Field(gt=0)
    channel: str = Field(min_length=1)


class SearchResponse(BaseModel):
    """Response contract for future search results."""

    query: str
    results: list[RankedChunkResponse] = Field(default_factory=list)


class RetrieveResponse(BaseModel):
    """Response contract for direct canonical chunk retrieval."""

    chunks: list[Chunk] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Response wrapper for an existing canonical document."""

    document: Document


class SourceResponse(BaseModel):
    """Summary response for a source and its canonical provenance."""

    source_id: str = Field(min_length=1)
    source_type: SourceType
    source_uri: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class IngestResponse(BaseModel):
    """Response contract for future ingestion results."""

    documents: list[Document] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
