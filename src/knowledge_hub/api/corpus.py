"""In-process canonical corpus view used by the API layer."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from knowledge_hub.config.settings import Settings
from knowledge_hub.indexing.qdrant import QdrantIndex
from knowledge_hub.models import Chunk, Document, Provenance, SourceType

from .contracts import SourceResponse


class CanonicalChunksNotFoundError(LookupError):
    """Raised when a canonical corpus lookup contains unknown chunk IDs."""

    def __init__(self, chunk_ids: Iterable[str]) -> None:
        self.chunk_ids = tuple(sorted(set(chunk_ids)))
        super().__init__(f"Chunks not found: {', '.join(self.chunk_ids)}")


class CanonicalCorpus:
    """Materialized canonical objects without introducing a second persistence layer."""

    def __init__(
        self,
        documents: Iterable[Document] = (),
        chunks: Iterable[Chunk] = (),
    ) -> None:
        self._documents: dict[str, Document] = {
            item.document_id: item for item in documents
        }
        self._chunks: dict[str, Chunk] = {}
        self.add_chunks(chunks)

    @classmethod
    def from_qdrant(cls, config: Settings) -> CanonicalCorpus:
        """Hydrate the canonical API view from persisted Qdrant payloads."""
        index = QdrantIndex(
            url=config.qdrant_url,
            collection=config.qdrant_collection,
            vector_size=config.embedding_dimension,
        )
        chunks: list[Chunk] = []
        offset = None
        while True:
            points, offset = index.client.scroll(
                collection_name=index.collection,
                limit=1_000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(Chunk.model_validate(point.payload or {}) for point in points)
            if offset is None:
                break

        chunks.sort(key=lambda chunk: chunk.chunk_id)
        documents = _documents_from_chunks(chunks)
        return cls(documents, chunks)

    def add(
        self,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
    ) -> None:
        self._documents.update({item.document_id: item for item in documents})
        self.add_chunks(chunks)

    def add_chunks(self, chunks: Iterable[Chunk]) -> None:
        self._chunks.update({item.chunk_id: item for item in chunks})

    def documents(
        self,
        limit: int,
        offset: int,
        source_type: SourceType | None = None,
    ) -> tuple[Document, ...]:
        values = tuple(
            document
            for document in self._documents.values()
            if source_type is None or document.source_type is source_type
        )
        return values[offset : offset + limit]

    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks.values())

    def chunks_by_ids(self, chunk_ids: Iterable[str]) -> tuple[Chunk, ...]:
        """Return canonical chunks in request order."""
        requested_ids = list(chunk_ids)
        missing_ids = [
            chunk_id for chunk_id in requested_ids if chunk_id not in self._chunks
        ]
        if missing_ids:
            raise CanonicalChunksNotFoundError(missing_ids)
        return tuple(self._chunks[chunk_id] for chunk_id in requested_ids)

    def sources(self) -> tuple[SourceResponse, ...]:
        unique: dict[str, SourceResponse] = {}
        for document in self._documents.values():
            provenance = document.provenance
            source_key = "\0".join(
                (
                    document.source_type.value,
                    document.source_uri or "",
                    provenance.repository or "",
                    document.metadata.get("project", ""),
                )
            )
            source_id = sha256(source_key.encode("utf-8")).hexdigest()
            unique.setdefault(
                source_key,
                SourceResponse(
                    source_id=source_id,
                    source_type=document.source_type,
                    source_uri=document.source_uri,
                    provenance=Provenance(**provenance.model_dump()),
                ),
            )
        return tuple(unique.values())


def _documents_from_chunks(chunks: Iterable[Chunk]) -> tuple[Document, ...]:
    """Build the document index from fields persisted with canonical chunks."""
    documents: dict[str, Document] = {}
    for chunk in chunks:
        if chunk.document_id in documents:
            continue
        source_uri = chunk.source_uri or chunk.provenance.source_uri
        title = (
            chunk.metadata.get("filename")
            or chunk.metadata.get("relative_path")
            or chunk.provenance.path
            or source_uri
        )
        documents[chunk.document_id] = Document(
            document_id=chunk.document_id,
            source_type=chunk.source_type,
            title=title,
            source_uri=source_uri,
            metadata=dict(chunk.metadata),
            provenance=Provenance(**chunk.provenance.model_dump()),
        )
    return tuple(documents.values())
