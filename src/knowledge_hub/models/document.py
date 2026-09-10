from pydantic import BaseModel, Field

from .provenance import Provenance
from .source import SourceType


class Document(BaseModel):
    document_id: str
    source_type: SourceType
    title: str | None = None
    source_uri: str | None = None
    content: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)
    provenance: Provenance = Field(default_factory=Provenance)
    structure: dict[str, object] = Field(default_factory=dict)
