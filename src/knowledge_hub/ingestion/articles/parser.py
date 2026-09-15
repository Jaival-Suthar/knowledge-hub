"""Markdown parsing for source-neutral article ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from knowledge_hub.models import Provenance, SourceType

from .contracts import Article, ArticleBlock, ArticleBlockType, ArticleSection
from .hashnode.contracts import HashnodeArticlePayload

_ATX_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_UNORDERED_ITEM = re.compile(r"^\s*[-+*]\s+(.*)$")
_ORDERED_ITEM = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_QUOTE_LINE = re.compile(r"^\s*>[ \t]?(.*)$")
_LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass(frozen=True)
class _ParsedBlock:
    block: ArticleBlock
    heading: tuple[str, int] | None = None


class ArticleParseError(ValueError):
    """Raised when article metadata cannot be converted safely."""


class ArticleParser:
    """Convert acquired raw article Markdown into an ``Article``."""

    def parse(self, payload: HashnodeArticlePayload) -> Article:
        """Parse Markdown structure without changing the source content."""
        if not isinstance(payload, HashnodeArticlePayload):
            raise TypeError("payload must be a HashnodeArticlePayload")

        blocks = _parse_blocks(payload.content)
        sections = _build_sections(blocks)
        title = _article_title(blocks)
        extra = {
            key: value
            for key, value in (
                ("slug", payload.slug),
                ("publication_host", payload.publication_host),
            )
            if value is not None
        }
        provenance = Provenance(
            source_uri=payload.source_uri,
            extra=extra,
        )

        return Article(
            source="hashnode",
            source_uri=payload.source_uri,
            article_id=payload.article_id,
            title=title,
            author=payload.author,
            published_at=_parse_date(payload.published_at, "published_at"),
            updated_at=_parse_date(payload.updated_at, "updated_at"),
            description=payload.description,
            tags=payload.tags,
            series=payload.series,
            content=payload.content,
            sections=sections,
            provenance=provenance,
            source_type=SourceType.ARTICLE,
        )


def _article_title(blocks: tuple[_ParsedBlock, ...]) -> str | None:
    for parsed in blocks:
        if parsed.heading is not None and parsed.heading[1] == 1:
            return parsed.heading[0] or None
    return None


def _parse_date(value: str | None, field_name: str) -> datetime | None:
    if value is None or not value.strip():
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ArticleParseError(f"invalid ISO-8601 {field_name}: {value!r}") from error


def _parse_blocks(content: str) -> tuple[_ParsedBlock, ...]:
    lines = content.splitlines(keepends=True)
    blocks: list[_ParsedBlock] = []
    index = 0

    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue

        parsed, index = _parse_one_block(lines, index, len(blocks))
        blocks.append(parsed)

    return tuple(blocks)


def _parse_one_block(
    lines: list[str], start: int, position: int
) -> tuple[_ParsedBlock, int]:
    line = lines[start]
    heading = _heading(line)
    if heading is not None:
        text, level = heading
        return (
            _ParsedBlock(
                block=ArticleBlock(
                    type=ArticleBlockType.HEADING,
                    content=text,
                    position=position,
                ),
                heading=(text, level),
            ),
            start + 1,
        )

    fence = _fence(line)
    if fence is not None:
        marker, language = fence
        end = start + 1
        while end < len(lines) and not _closing_fence(lines[end], marker):
            end += 1
        code = "".join(lines[start + 1 : end])
        next_index = min(end + 1, len(lines)) if end < len(lines) else end
        return _ParsedBlock(
            block=ArticleBlock(
                type=ArticleBlockType.CODE,
                content=code,
                position=position,
                language=language,
            )
        ), next_index

    if _is_table_start(lines, start):
        end = start + 2
        while end < len(lines) and _is_table_row(lines[end]):
            end += 1
        rows = tuple(
            _table_row(lines[row]) for row in range(start, end) if row != start + 1
        )
        return _ParsedBlock(
            block=ArticleBlock(
                type=ArticleBlockType.TABLE,
                content="".join(lines[start:end]).strip(),
                position=position,
                rows=rows,
            )
        ), end

    if _list_item(line) is not None:
        items: list[str] = []
        end = start
        while end < len(lines):
            item = _list_item(lines[end])
            if item is None:
                break
            items.append(item)
            end += 1
        return _ParsedBlock(
            block=ArticleBlock(
                type=ArticleBlockType.LIST,
                content="\n".join(items),
                position=position,
                items=tuple(items),
            )
        ), end

    if _QUOTE_LINE.match(line):
        quoted: list[str] = []
        end = start
        while end < len(lines):
            match = _QUOTE_LINE.match(lines[end])
            if match is None:
                break
            quoted.append(match.group(1))
            end += 1
        text = "".join(quoted).strip()
        return _ParsedBlock(
            block=ArticleBlock(
                type=ArticleBlockType.QUOTE,
                content=text,
                position=position,
            )
        ), end

    paragraph: list[str] = [line]
    end = start + 1
    while end < len(lines) and lines[end].strip():
        if _starts_structural_block(lines, end):
            break
        paragraph.append(lines[end])
        end += 1
    text = "".join(paragraph).strip()
    link = _LINK.search(text)
    return _ParsedBlock(
        block=ArticleBlock(
            type=ArticleBlockType.LINK if link else ArticleBlockType.PARAGRAPH,
            content=text,
            position=position,
            url=link.group(1) if link else None,
        )
    ), end


def _build_sections(blocks: tuple[_ParsedBlock, ...]) -> tuple[ArticleSection, ...]:
    if not blocks:
        return ()

    sections: list[ArticleSection] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading: str | None = None
    current_level = 0
    current_path: tuple[str, ...] = ()
    current_blocks: list[ArticleBlock] = []

    def flush() -> None:
        if not current_blocks:
            return
        content = "\n\n".join(
            block.content
            for block in current_blocks
            if block.type is not ArticleBlockType.HEADING and block.content
        )
        sections.append(
            ArticleSection(
                heading=current_heading,
                heading_level=current_level,
                heading_path=current_path,
                content=content,
                position=len(sections),
                blocks=tuple(current_blocks),
            )
        )

    for parsed in blocks:
        block = parsed.block
        if parsed.heading is not None:
            flush()
            heading, level = parsed.heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, heading))
            current_heading = heading
            current_level = level
            current_path = tuple(item[1] for item in heading_stack)
            current_blocks = [block]
        else:
            current_blocks.append(block)

    flush()
    return tuple(sections)


def _heading(line: str) -> tuple[str, int] | None:
    match = _ATX_HEADING.match(line.rstrip("\r\n"))
    if match is None:
        return None
    text = (match.group(2) or "").strip()
    text = re.sub(r"[ \t]+#+[ \t]*$", "", text).rstrip()
    return text, len(match.group(1))


def _fence(line: str) -> tuple[str, str | None] | None:
    match = _FENCE_OPEN.match(line.rstrip("\r\n"))
    if match is None:
        return None
    marker = match.group(1)
    info = match.group(2).strip()
    return marker, info.split(None, 1)[0] if info else None


def _closing_fence(line: str, opening: str) -> bool:
    marker = re.escape(opening[0])
    return bool(
        re.match(rf"^ {{0,3}}{marker}{{{len(opening)},}}[ \t]*$", line.rstrip("\r\n"))
    )


def _is_table_start(lines: list[str], index: int) -> bool:
    return (
        index + 1 < len(lines)
        and _is_table_row(lines[index])
        and _is_table_delimiter(lines[index + 1])
    )


def _is_table_row(line: str) -> bool:
    return "|" in line and line.strip().startswith("|")


def _is_table_delimiter(line: str) -> bool:
    if not _is_table_row(line):
        return False
    cells = line.strip().strip("|").split("|")
    return bool(cells) and all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in cells)


def _table_row(line: str) -> tuple[str, ...]:
    return tuple(cell.strip() for cell in line.strip().strip("|").split("|"))


def _list_item(line: str) -> str | None:
    match = _UNORDERED_ITEM.match(line) or _ORDERED_ITEM.match(line)
    return match.group(1).rstrip() if match else None


def _starts_structural_block(lines: list[str], index: int) -> bool:
    return (
        _heading(lines[index]) is not None
        or _fence(lines[index]) is not None
        or _list_item(lines[index]) is not None
        or bool(_QUOTE_LINE.match(lines[index]))
        or _is_table_start(lines, index)
    )
