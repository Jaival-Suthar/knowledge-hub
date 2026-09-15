"""Source-domain contracts for technical articles."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from knowledge_hub.models import Provenance, SourceType


class ArticleBlockType(StrEnum):
    """Structured content block kinds an article parser may produce."""

    PARAGRAPH = "paragraph"
    HEADING = "heading"
    CODE = "code"
    LIST = "list"
    TABLE = "table"
    QUOTE = "quote"
    LINK = "link"


@dataclass(frozen=True)
class ArticleBlock:
    """One structured block of article content.

    The contract carries source structure without prescribing how a parser
    obtains or renders it.  ``content`` is the block's source text; the
    optional fields preserve structure useful to later adapters.
    """

    type: ArticleBlockType | str
    content: str = ""
    position: int = 0
    language: str | None = None
    items: tuple[str, ...] = ()
    rows: tuple[tuple[str, ...], ...] = ()
    url: str | None = None

    @property
    def block_type(self) -> ArticleBlockType | str:
        """Alias with an explicit name for callers that prefer it."""
        return self.type


@dataclass(frozen=True)
class ArticleSection:
    """A hierarchical article section with its original position."""

    heading: str | None
    heading_level: int
    heading_path: tuple[str, ...]
    content: str = ""
    position: int = 0
    blocks: tuple[ArticleBlock, ...] = ()


@dataclass(frozen=True)
class Article:
    """Source-neutral article data before canonical conversion."""

    source: str
    source_uri: str
    article_id: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    updated_at: datetime | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    series: str | None = None
    content: str = ""
    sections: tuple[ArticleSection, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)
    source_type: SourceType = SourceType.ARTICLE
