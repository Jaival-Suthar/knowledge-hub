"""Translate article-specific knowledge into source-neutral canonical models."""

from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256

from knowledge_hub.models import Chunk, Document, Provenance

from .chunking import ArticleChunk
from .contracts import Article


class ArticleCanonicalizationError(ValueError):
    """Raised when an article cannot cross the canonical boundary safely."""


def document_id_for(article: Article) -> str:
    """Return a deterministic identity for this acquired article version.

    The identity is derived from the public source URI and the exact article
    content. It intentionally does not depend on chunk ordering.
    """
    if not article.source_uri:
        raise ArticleCanonicalizationError("article source_uri is required")

    content_hash = _content_hash(article.content)
    identity = f"{article.source_uri}\0{content_hash}"

    return _hash(identity)


def to_canonical_document(article: Article) -> Document:
    """Convert an Article into the source-neutral canonical Document model."""
    _validate_article(article)

    content_hash = _content_hash(article.content)

    return Document(
        document_id=document_id_for(article),
        source_type=article.source_type,
        title=article.title,
        source_uri=article.source_uri,
        content=article.content,
        metadata=_article_metadata(article, content_hash),
        provenance=_document_provenance(article),
        structure={
            "heading_paths": [
                list(section.heading_path) for section in article.sections
            ]
        },
    )


def article_chunk_to_canonical(
    article_chunk: ArticleChunk,
    document: Document,
) -> Chunk:
    """Convert one ArticleChunk without reparsing or modifying its content."""
    _validate_chunk_boundary(article_chunk, document)

    heading_path = " > ".join(article_chunk.heading_path)

    metadata = dict(document.metadata)
    metadata.update(
        {
            "chunk_index": str(article_chunk.chunk_index),
            "content_type": article_chunk.content_type,
            "heading_path": heading_path,
            "block_types": ",".join(article_chunk.block_types),
        }
    )

    if article_chunk.article_id is not None:
        metadata["article_id"] = article_chunk.article_id

    return Chunk(
        document_id=document.document_id,
        chunk_id=article_chunk.chunk_id,
        content=article_chunk.content,
        source_type=article_chunk.source_type,
        source_uri=article_chunk.source_uri,
        timestamp=_timestamp(article_chunk.updated_at or article_chunk.published_at),
        content_role=article_chunk.content_type,
        structural_type=article_chunk.content_type,
        parent_structure=(
            article_chunk.heading_path[-2]
            if len(article_chunk.heading_path) > 1
            else None
        ),
        location=_chunk_location(article_chunk),
        token_count=article_chunk.token_count,
        content_hash=article_chunk.content_hash,
        metadata=metadata,
        provenance=_chunk_provenance(article_chunk, heading_path),
    )


def to_canonical_chunks(
    article: Article,
    document: Document,
    article_chunks: Iterable[ArticleChunk],
) -> tuple[Chunk, ...]:
    """Convert article chunks while preserving their supplied order."""
    _validate_article_document(article, document)

    return tuple(
        article_chunk_to_canonical(article_chunk, document)
        for article_chunk in article_chunks
    )


def canonicalize_article(
    article: Article,
    article_chunks: Iterable[ArticleChunk],
) -> tuple[Document, tuple[Chunk, ...]]:
    """Convert an Article and its already-created chunks canonically.

    Chunking is intentionally outside this function. A5 consumes the
    deterministic ArticleChunk objects produced by A4 and does not re-chunk,
    normalize, or otherwise transform their content.
    """
    document = to_canonical_document(article)
    chunks = to_canonical_chunks(article, document, article_chunks)

    return document, chunks


def _validate_article(article: Article) -> None:
    """Validate invariants required before crossing into canonical models."""
    if not article.source_uri:
        raise ArticleCanonicalizationError("article source_uri is required")

    if not article.source_type:
        raise ArticleCanonicalizationError("article source_type is required")

    if article.content is None:
        raise ArticleCanonicalizationError("article content is required")


def _validate_article_document(
    article: Article,
    document: Document,
) -> None:
    """Ensure a canonical Document belongs to the supplied Article."""
    expected_document_id = document_id_for(article)

    if document.document_id != expected_document_id:
        raise ArticleCanonicalizationError(
            "canonical document does not match the supplied article"
        )

    if document.source_uri != article.source_uri:
        raise ArticleCanonicalizationError(
            "canonical document source_uri does not match article"
        )

    if document.source_type is not article.source_type:
        raise ArticleCanonicalizationError(
            "canonical document source_type does not match article"
        )

    if document.content != article.content:
        raise ArticleCanonicalizationError(
            "canonical document content does not match article"
        )

    if document.provenance.source_uri != article.source_uri:
        raise ArticleCanonicalizationError(
            "canonical document provenance does not match article"
        )


def _validate_chunk_boundary(
    article_chunk: ArticleChunk,
    document: Document,
) -> None:
    """Validate invariants required for ArticleChunk → Chunk conversion."""
    if not article_chunk.source_uri:
        raise ArticleCanonicalizationError("article chunk source_uri is required")

    if not article_chunk.chunk_id:
        raise ArticleCanonicalizationError("article chunk chunk_id is required")

    if not article_chunk.content_hash:
        raise ArticleCanonicalizationError("article chunk content_hash is required")

    if not article_chunk.content:
        raise ArticleCanonicalizationError("article chunk content is required")

    if document.source_uri != article_chunk.source_uri:
        raise ArticleCanonicalizationError(
            "article chunk source_uri does not match canonical document"
        )

    if document.source_type is not article_chunk.source_type:
        raise ArticleCanonicalizationError(
            "article chunk source_type does not match canonical document"
        )

    document_article_id = document.metadata.get("article_id")

    if (
        article_chunk.article_id is not None
        and document_article_id is not None
        and article_chunk.article_id != document_article_id
    ):
        raise ArticleCanonicalizationError(
            "article chunk article_id does not match canonical document"
        )

    expected_content_hash = _content_hash(article_chunk.content)

    if article_chunk.content_hash != expected_content_hash:
        raise ArticleCanonicalizationError(
            "article chunk content_hash does not match content"
        )

    provenance_uri = article_chunk.provenance.source_uri

    if provenance_uri != article_chunk.source_uri:
        raise ArticleCanonicalizationError(
            "article chunk provenance source_uri does not match chunk"
        )


def _document_provenance(article: Article) -> Provenance:
    """Build document provenance while preserving existing provenance data."""
    extra = dict(article.provenance.extra)

    if article.article_id is not None:
        extra["article_id"] = article.article_id

    return article.provenance.model_copy(
        update={
            "source_uri": article.source_uri,
            "extra": extra,
        }
    )


def _chunk_provenance(
    article_chunk: ArticleChunk,
    heading_path: str,
) -> Provenance:
    """Build chunk provenance while preserving existing provenance data."""
    extra = dict(article_chunk.provenance.extra)

    extra.update(
        {
            "chunk_id": article_chunk.chunk_id,
            "chunk_index": str(article_chunk.chunk_index),
            "content_type": article_chunk.content_type,
            "heading_path": heading_path,
        }
    )

    if article_chunk.article_id is not None:
        extra["article_id"] = article_chunk.article_id

    return article_chunk.provenance.model_copy(
        update={
            "source_uri": article_chunk.source_uri,
            "section": heading_path or None,
            "extra": extra,
        }
    )


def _article_metadata(
    article: Article,
    content_hash: str,
) -> dict[str, str]:
    """Map supported article metadata into source-neutral metadata."""
    metadata = {
        "source": article.source,
        "content_hash": content_hash,
    }

    if article.article_id is not None:
        metadata["article_id"] = article.article_id

    if article.author is not None:
        metadata["author"] = article.author

    if article.published_at is not None:
        metadata["published_at"] = article.published_at.isoformat()

    if article.updated_at is not None:
        metadata["updated_at"] = article.updated_at.isoformat()

    if article.description is not None:
        metadata["description"] = article.description

    if article.tags:
        metadata["tags"] = ",".join(article.tags)

    if article.series is not None:
        metadata["series"] = article.series

    metadata.update(article.provenance.extra)

    return metadata


def _chunk_location(article_chunk: ArticleChunk) -> str:
    """Return a deterministic human-readable structural location."""
    heading_path = " > ".join(article_chunk.heading_path) or "<root>"

    return f"{heading_path}:chunk:{article_chunk.chunk_index}"


def _timestamp(value: object) -> str | None:
    """Serialize an optional timestamp without introducing new values."""
    return value.isoformat() if value is not None else None


def _content_hash(content: str) -> str:
    """Return the canonical SHA-256 hash used for content identity."""
    return sha256(content.encode("utf-8")).hexdigest()


def _hash(value: str) -> str:
    """Return a deterministic SHA-256 digest for an identity string."""
    return sha256(value.encode("utf-8")).hexdigest()
