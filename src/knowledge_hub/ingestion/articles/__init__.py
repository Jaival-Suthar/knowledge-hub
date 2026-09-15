"""Contracts for source-neutral article ingestion."""

from .chunking import ArticleChunk, ArticleChunker
from .contracts import Article, ArticleBlock, ArticleBlockType, ArticleSection
from .parser import ArticleParseError, ArticleParser

__all__ = [
    "Article",
    "ArticleBlock",
    "ArticleBlockType",
    "ArticleChunk",
    "ArticleChunker",
    "ArticleParseError",
    "ArticleParser",
    "ArticleSection",
]
