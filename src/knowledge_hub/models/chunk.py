from pydantic import BaseModel, Field

from .provenance import Provenance
from .source import SourceType


class Chunk(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    source_type: SourceType
    source_uri: str | None = None
    project: str | None = None
    timestamp: str | None = None
    language: str | None = None
    content_role: str | None = None
    structural_type: str | None = None
    parent_structure: str | None = None
    location: str | None = None
    token_count: int | None = None
    content_hash: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
