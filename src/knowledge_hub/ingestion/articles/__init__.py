"""Contracts for source-neutral article ingestion."""

from .canonical import (
    ArticleCanonicalizationError,
    article_chunk_to_canonical,
    canonicalize_article,
    document_id_for,
    to_canonical_chunks,
    to_canonical_document,
)
from .chunking import ArticleChunk, ArticleChunker
from .contracts import Article, ArticleBlock, ArticleBlockType, ArticleSection
from .parser import ArticleParseError, ArticleParser

__all__ = [
    "Article",
    "ArticleBlock",
    "ArticleBlockType",
    "ArticleCanonicalizationError",
    "ArticleChunk",
    "ArticleChunker",
    "ArticleParseError",
    "ArticleParser",
    "ArticleSection",
    "article_chunk_to_canonical",
    "canonicalize_article",
    "document_id_for",
    "to_canonical_chunks",
    "to_canonical_document",
]
