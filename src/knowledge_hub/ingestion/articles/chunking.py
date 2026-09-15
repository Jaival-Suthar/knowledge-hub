"""Article-aware, deterministic chunking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from hashlib import sha256

import tiktoken

from knowledge_hub.models import Provenance, SourceType

from .contracts import Article, ArticleBlock, ArticleBlockType, ArticleSection

DEFAULT_ARTICLE_MAX_TOKENS = 512
_BLOCK_SEPARATOR = "\n\n"


@dataclass(frozen=True)
class ArticleChunk:
    """One source-domain chunk with article structure and provenance."""

    article_id: str | None
    source_type: SourceType
    source_uri: str
    title: str | None
    author: str | None
    published_at: datetime | None
    updated_at: datetime | None
    heading_path: tuple[str, ...]
    heading_level: int
    chunk_index: int
    content: str
    content_type: str
    token_count: int
    content_hash: str
    chunk_id: str
    block_types: tuple[str, ...] = ()
    provenance: Provenance = field(default_factory=Provenance)


class ArticleChunker:
    """Group article blocks semantically and enforce a token limit."""

    def __init__(self, max_tokens: int = DEFAULT_ARTICLE_MAX_TOKENS) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, article: Article) -> tuple[ArticleChunk, ...]:
        """Create ordered chunks without merging unrelated article sections."""
        output: list[ArticleChunk] = []
        chunk_index = 0

        for section in article.sections:
            pending: list[tuple[ArticleBlock, str]] = []
            block_occurrences: dict[tuple[str, str], int] = {}

            for block in _content_blocks(section):
                local_id = self._local_block_id(block, block_occurrences)

                if self._token_count(block.content) > self.max_tokens:
                    if pending:
                        output.append(
                            self._make_pending_chunk(
                                article,
                                section,
                                pending,
                                chunk_index,
                            )
                        )
                        chunk_index += 1
                        pending.clear()

                    for piece_index, piece in enumerate(self._split_block(block)):
                        output.append(
                            self._make_chunk(
                                article,
                                section,
                                piece,
                                (_block_type(block),),
                                f"{local_id}:part:{piece_index}",
                                chunk_index,
                            )
                        )
                        chunk_index += 1
                    continue

                candidate = _join_blocks(
                    [
                        pending_block
                        for pending_block, _ in [*pending, (block, local_id)]
                    ]
                )

                if pending and self._token_count(candidate) > self.max_tokens:
                    output.append(
                        self._make_pending_chunk(
                            article,
                            section,
                            pending,
                            chunk_index,
                        )
                    )
                    chunk_index += 1
                    pending.clear()

                pending.append((block, local_id))

            if pending:
                output.append(
                    self._make_pending_chunk(
                        article,
                        section,
                        pending,
                        chunk_index,
                    )
                )
                chunk_index += 1

        return tuple(output)

    def _make_pending_chunk(
        self,
        article: Article,
        section: ArticleSection,
        pending: list[tuple[ArticleBlock, str]],
        chunk_index: int,
    ) -> ArticleChunk:
        blocks = [block for block, _ in pending]
        local_structure = "|".join(local_id for _, local_id in pending)

        return self._make_chunk(
            article,
            section,
            _join_blocks(blocks),
            tuple(_block_type(block) for block in blocks),
            local_structure,
            chunk_index,
        )

    def _local_block_id(
        self,
        block: ArticleBlock,
        occurrences: dict[tuple[str, str], int],
    ) -> str:
        """Return a deterministic identity for a block within its section."""
        block_type = _block_type(block)
        signature = (block_type, block.content)
        occurrence = occurrences.get(signature, 0)
        occurrences[signature] = occurrence + 1

        return f"{block_type}:{occurrence}:{block.content}"

    def _make_chunk(
        self,
        article: Article,
        section: ArticleSection,
        content: str,
        block_types: tuple[str, ...],
        local_structure: str,
        chunk_index: int,
    ) -> ArticleChunk:
        unique_block_types = tuple(dict.fromkeys(block_types))
        content_type = (
            unique_block_types[0] if len(unique_block_types) == 1 else "section"
        )
        content_hash = sha256(content.encode("utf-8")).hexdigest()

        identity = "\0".join(
            (
                article.source_uri,
                str(section.position),
                " > ".join(section.heading_path),
                local_structure,
                content,
            )
        )

        provenance = Provenance(**article.provenance.model_dump())

        return ArticleChunk(
            article_id=article.article_id,
            source_type=article.source_type,
            source_uri=article.source_uri,
            title=article.title,
            author=article.author,
            published_at=article.published_at,
            updated_at=article.updated_at,
            heading_path=section.heading_path,
            heading_level=section.heading_level,
            chunk_index=chunk_index,
            content=content,
            content_type=content_type,
            token_count=self._token_count(content),
            content_hash=content_hash,
            chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
            block_types=unique_block_types,
            provenance=provenance,
        )

    def _split_block(self, block: ArticleBlock) -> tuple[str, ...]:
        if block.type == ArticleBlockType.LIST and block.items:
            return self._pack_parts(block.items)

        if block.type == ArticleBlockType.TABLE and block.rows:
            lines = block.content.splitlines(keepends=True)
            if lines:
                return self._pack_parts(
                    tuple(lines),
                    preserve_separators=True,
                )

        if block.type == ArticleBlockType.CODE:
            lines = block.content.splitlines(keepends=True)
            if lines:
                return self._pack_parts(
                    tuple(lines),
                    preserve_separators=True,
                )

        return self._split_text(block.content)

    def _pack_parts(
        self,
        parts: tuple[str, ...],
        *,
        preserve_separators: bool = False,
    ) -> tuple[str, ...]:
        output: list[str] = []
        current = ""
        separator = "" if preserve_separators else "\n"

        for part in parts:
            if self._token_count(part) > self.max_tokens:
                if current:
                    output.append(current)
                    current = ""
                output.extend(self._split_text(part))
                continue

            candidate = f"{current}{separator}{part}" if current else part

            if current and self._token_count(candidate) > self.max_tokens:
                output.append(current)
                current = part
            else:
                current = candidate

        if current:
            output.append(current)

        return tuple(output)

    def _split_text(self, text: str) -> tuple[str, ...]:
        tokens = self.encoding.encode(text)

        if len(tokens) <= self.max_tokens:
            return (text,)

        _, offsets = self.encoding.decode_with_offsets(tokens)
        pieces: list[str] = []
        token_start = 0
        char_start = 0

        while token_start < len(tokens):
            token_end = min(token_start + self.max_tokens, len(tokens))
            char_end = _token_end_offset(text, offsets, token_end)

            while (
                token_end > token_start + 1
                and char_end > char_start
                and self._token_count(text[char_start:char_end]) > self.max_tokens
            ):
                token_end -= 1
                char_end = _token_end_offset(text, offsets, token_end)

            if char_end <= char_start:
                char_end = min(len(text), char_start + 1)

            piece = text[char_start:char_end]
            if not piece:
                break

            pieces.append(piece)
            char_start = char_end
            token_start = token_end

        if char_start < len(text):
            pieces.append(text[char_start:])

        return tuple(pieces)

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))


def _content_blocks(section: ArticleSection) -> tuple[ArticleBlock, ...]:
    return tuple(
        block
        for block in section.blocks
        if block.type != ArticleBlockType.HEADING and block.content
    )


def _join_blocks(blocks: list[ArticleBlock]) -> str:
    return _BLOCK_SEPARATOR.join(block.content for block in blocks)


def _block_type(block: ArticleBlock) -> str:
    return (
        block.type.value
        if isinstance(block.type, ArticleBlockType)
        else str(block.type)
    )


def _token_end_offset(text: str, offsets: list[int], token_end: int) -> int:
    return len(text) if token_end >= len(offsets) else offsets[token_end]
