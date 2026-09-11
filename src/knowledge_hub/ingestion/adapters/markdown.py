from __future__ import annotations

import re
from hashlib import sha256
from pathlib import Path

from knowledge_hub.models import Document, Provenance, SourceType


class MarkdownAdapter:
    """Read Markdown into the canonical document model with block structure."""

    _extensions = frozenset({".md", ".markdown"})

    def supports(self, source: str | Path) -> bool:
        return Path(source).suffix.lower() in self._extensions

    def extract(self, source: str | Path) -> Document:
        path = Path(source)
        if not self.supports(path):
            raise ValueError(f"unsupported Markdown extension: {path.suffix}")

        raw = path.read_bytes()
        content = raw.decode("utf-8-sig")
        source_uri = path.resolve().as_uri()
        document_id = sha256(raw).hexdigest()
        blocks = _parse_blocks(content)

        return Document(
            document_id=document_id,
            source_type=SourceType.MARKDOWN,
            title=path.stem,
            source_uri=source_uri,
            content=content,
            metadata={
                "filename": path.name,
                "path": str(path),
                "source_type": SourceType.MARKDOWN.value,
            },
            provenance=Provenance(source_uri=source_uri, path=str(path)),
            structure={"blocks": blocks},
        )


def _parse_blocks(content: str) -> list[dict[str, object]]:
    lines = content.splitlines()
    blocks: list[dict[str, object]] = []
    heading_stack: list[tuple[int, str]] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line.strip():
            index += 1
            continue

        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level = len(heading.group(1))
            text = heading.group(2).strip()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            blocks.append(
                {
                    "type": "heading",
                    "level": level,
                    "text": text,
                    "heading_path": [item[1] for item in heading_stack],
                }
            )
            index += 1
            continue

        block_type, end = _block_end(lines, index)
        text = "\n".join(lines[index:end]).strip()
        blocks.append(
            {
                "type": block_type,
                "text": text,
                "heading_path": [item[1] for item in heading_stack],
            }
        )
        index = end

    return blocks


def _block_end(lines: list[str], start: int) -> tuple[str, int]:
    line = lines[start]
    if line.lstrip().startswith("```") or line.lstrip().startswith("~~~"):
        fence = line.lstrip()[:3]
        end = start + 1
        while end < len(lines) and not lines[end].lstrip().startswith(fence):
            end += 1
        return "code_block", min(end + 1, len(lines))
    if re.match(r"^\s*(?:[-+*]|\d+[.)])\s+", line):
        return _consecutive(lines, start, lambda value: bool(re.match(r"^\s*(?:[-+*]|\d+[.)])\s+", value)))
    if line.lstrip().startswith(">"):
        return _consecutive(lines, start, lambda value: value.lstrip().startswith(">"))
    if _is_table_line(line):
        end = start
        while end < len(lines) and lines[end].strip() and _is_table_line(lines[end]):
            end += 1
        return "table", end
    return _paragraph_end(lines, start)


def _consecutive(lines: list[str], start: int, predicate) -> tuple[str, int]:
    end = start
    while end < len(lines) and lines[end].strip() and predicate(lines[end]):
        end += 1
    return ("list" if re.match(r"^\s*(?:[-+*]|\d+[.)])\s+", lines[start]) else "blockquote", end)


def _is_table_line(line: str) -> bool:
    return "|" in line and line.strip().startswith("|")


def _paragraph_end(lines: list[str], start: int) -> tuple[str, int]:
    end = start + 1
    while end < len(lines) and lines[end].strip():
        if re.match(r"^(#{1,6})\s+", lines[end]):
            break
        if lines[end].lstrip().startswith((">", "```", "~~~")):
            break
        end += 1
    return "paragraph", end
