from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from knowledge_hub.models import Chunk, Document, Provenance, SourceType

from .chunking import SemanticCodeChunk
from .contracts import CodeFile


def document_id_for(code_file: CodeFile) -> str:
    """Return the stable identity for one source file."""
    normalized_path = code_file.relative_path.replace("\\", "/")
    identity = f"{normalized_path}\0{code_file.content_hash}"
    return sha256(identity.encode("utf-8")).hexdigest()


def to_canonical_document(
    code_file: CodeFile,
    *,
    source_type: SourceType = SourceType.CODE,
    provenance: Provenance | None = None,
) -> Document:
    """Map one CodeFile to exactly one canonical Document without I/O."""
    document_id = document_id_for(code_file)
    document_provenance = _document_provenance(code_file, provenance)
    source_uri = document_provenance.source_uri or code_file.relative_path
    metadata = {
        "relative_path": code_file.relative_path,
        "language": code_file.language.value,
        "content_hash": code_file.content_hash,
    }
    metadata.update(_source_metadata(document_provenance))
    return Document(
        document_id=document_id,
        source_type=source_type,
        title=code_file.relative_path,
        source_uri=source_uri,
        content=code_file.content,
        metadata=metadata,
        provenance=document_provenance,
        structure={
            "relative_path": code_file.relative_path,
            "language": code_file.language.value,
        },
    )


def to_canonical_chunks(
    code_file: CodeFile,
    document: Document,
    semantic_chunks: Iterable[SemanticCodeChunk],
) -> tuple[Chunk, ...]:
    """Map semantic code chunks to canonical Chunks for one Document."""
    return tuple(
        _to_canonical_chunk(code_file, document, semantic_chunk)
        for semantic_chunk in semantic_chunks
    )


def canonicalize_code(
    code_file: CodeFile,
    semantic_chunks: Iterable[SemanticCodeChunk],
    *,
    source_type: SourceType = SourceType.CODE,
    provenance: Provenance | None = None,
) -> tuple[Document, tuple[Chunk, ...]]:
    """Convert one CodeFile and its semantic chunks to canonical models."""
    document = to_canonical_document(
        code_file,
        source_type=source_type,
        provenance=provenance,
    )
    return document, to_canonical_chunks(code_file, document, semantic_chunks)


def _to_canonical_chunk(
    code_file: CodeFile,
    document: Document,
    semantic_chunk: SemanticCodeChunk,
) -> Chunk:
    symbol = semantic_chunk.symbol
    symbol_type = (
        symbol.symbol_type.value
        if hasattr(symbol.symbol_type, "value")
        else str(symbol.symbol_type)
    )
    location = (
        f"{code_file.relative_path}:lines:"
        f"{semantic_chunk.start_line}-{semantic_chunk.end_line}"
    )
    if semantic_chunk.part_index:
        location += f":part:{semantic_chunk.part_index}"

    chunk_provenance = document.provenance.model_copy(
        update={
            "path": code_file.relative_path,
            "symbol": symbol.qualified_name,
            "line_start": semantic_chunk.start_line,
            "line_end": semantic_chunk.end_line,
        }
    )
    metadata = {
        "symbol": symbol.name,
        "qualified_name": symbol.qualified_name,
        "part_index": str(semantic_chunk.part_index),
    }
    metadata.update(_source_metadata(chunk_provenance))

    return Chunk(
        document_id=document.document_id,
        chunk_id=semantic_chunk.chunk_id,
        content=semantic_chunk.content,
        source_type=document.source_type,
        source_uri=document.source_uri,
        language=semantic_chunk.language.value,
        structural_type=symbol_type,
        parent_structure=symbol.parent,
        location=location,
        content_hash=sha256(semantic_chunk.content.encode("utf-8")).hexdigest(),
        metadata=metadata,
        provenance=chunk_provenance,
    )


def _document_provenance(
    code_file: CodeFile,
    provenance: Provenance | None,
) -> Provenance:
    base = provenance or Provenance()
    return base.model_copy(
        update={
            "source_uri": base.source_uri or code_file.relative_path,
            "path": base.path or code_file.relative_path,
        }
    )


def _source_metadata(provenance: Provenance) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if provenance.repository is not None:
        metadata["repository"] = provenance.repository
    if provenance.branch is not None:
        metadata["ref"] = provenance.branch
    if provenance.commit_sha is not None:
        metadata["commit_sha"] = provenance.commit_sha
    metadata.update(provenance.extra)
    return metadata
