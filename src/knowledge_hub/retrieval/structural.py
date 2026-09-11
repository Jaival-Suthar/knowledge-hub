from __future__ import annotations

import re

from knowledge_hub.models import Chunk

EXCLUDED_CONTENT_ROLES = frozenset({"navigation", "metadata", "reference"})

_STRUCTURAL_LABELS = frozenset(
    {"contents", "table of contents", "index", "glossary", "references", "reference",
     "copyright", "publication information", "permissions", "legal disclaimer", "disclaimer"}
)
_STRUCTURAL_WORDS = frozenset(
    {"navigation", "metadata", "reference", "references", "index", "contents", "glossary",
     "copyright", "permissions", "legal", "disclaimer", "front_matter", "back_matter"}
)
_LEGAL_PATTERNS = (
    r"all rights reserved", r"copyright clearance center",
    r"no part of this .* may be reproduced", r"may not be reproduced",
    r"reproduction in any form", r"permission to reproduce", r"published by",
    r"first published", r"isbn(?:[-\s]|\d)", r"library of congress",
)


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _field_values(chunk: Chunk) -> list[str]:
    values = [chunk.structural_type, chunk.parent_structure, chunk.provenance.section,
              *chunk.metadata.keys(), *chunk.metadata.values()]
    return [_normal(value) for value in values if _normal(value)]


def _has_structural_metadata(chunk: Chunk) -> bool:
    for value in _field_values(chunk):
        if value in _STRUCTURAL_LABELS:
            return True
        words = set(re.findall(r"[a-z_]+", value))
        if words & _STRUCTURAL_WORDS and len(value) <= 80:
            return True
    return False


def _is_contents(chunk: Chunk, content: str) -> bool:
    parent = _normal(chunk.parent_structure)
    first_line = _normal(content.splitlines()[0]) if content.splitlines() else ""
    return parent in {"contents", "table of contents"} or first_line in {
        "contents",
        "table of contents",
    }


def _is_legal_metadata(content: str, parent: str) -> bool:
    matches = sum(bool(re.search(pattern, content)) for pattern in _LEGAL_PATTERNS)
    copyright_signal = bool(re.search(r"(?:©|\bcopyright\b)\s*(?:\d{4}|by\b)", content))
    section_signal = any(term in parent for term in ("copyright", "publication", "permissions", "legal", "disclaimer"))
    # A single mention of copyright in ordinary prose is intentionally insufficient.
    return matches >= 2 or (copyright_signal and (matches >= 1 or section_signal))


def _looks_like_index(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    short_lines = sum(len(line.split()) <= 12 for line in lines)
    entry_lines = sum(bool(re.match(r"^[A-Za-z][A-Za-z'’ -]{1,45}(?:,|\.{2,}|\s+\d|$)", line)) for line in lines)
    # Index pages are lists of compact entries, unlike paragraph-oriented prose.
    return short_lines / len(lines) >= 0.72 and entry_lines / len(lines) >= 0.55


def _is_reference_or_glossary(chunk: Chunk, content: str) -> bool:
    parent = _normal(chunk.parent_structure)
    if parent in {"glossary", "reference", "references"}:
        return True
    if re.match(r"^(?:glossary|references?)\s*$", content):
        return True
    if re.match(r"^(?:glossary|references?)\b", content) and "\n" in content:
        return True
    return _looks_like_index(content)


def classify_content_role(chunk: Chunk) -> str:
    """Classify a canonical chunk, prioritising structural evidence."""
    content = chunk.content.strip().lower()
    content_flat = _normal(chunk.content)
    parent = _normal(chunk.parent_structure)

    if _has_structural_metadata(chunk) or _is_contents(chunk, content):
        if any(term in parent for term in ("copyright", "publication", "permission", "legal")):
            return "metadata"
        if parent in {"glossary", "reference", "references"}:
            return "reference"
        if parent in {"index", "contents", "table of contents"} or _is_contents(chunk, content):
            return "navigation"
    if _is_legal_metadata(content_flat, parent):
        return "metadata"
    if _is_reference_or_glossary(chunk, content):
        return "reference"
    if chunk.content_role:
        return _normal(chunk.content_role)
    return "documentation"


def structural_eligible(chunk: Chunk) -> bool:
    """Return whether one canonical chunk is eligible for retrieval."""
    return classify_content_role(chunk) not in EXCLUDED_CONTENT_ROLES
