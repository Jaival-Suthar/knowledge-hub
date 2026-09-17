from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.types import RankedChunk


class EligibilityStatus(StrEnum):
    """Outcome of structural eligibility evaluation."""

    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNKNOWN = "unknown"


STRUCTURAL_ELIGIBILITY_POLICY = {
    "documentation": EligibilityStatus.ELIGIBLE,
    "implementation": EligibilityStatus.ELIGIBLE,
    "evidence": EligibilityStatus.ELIGIBLE,
    "reference": EligibilityStatus.ELIGIBLE,
    "test": EligibilityStatus.ELIGIBLE,
    "configuration": EligibilityStatus.ELIGIBLE,
    "navigation": EligibilityStatus.INELIGIBLE,
    "metadata": EligibilityStatus.INELIGIBLE,
}

EXCLUDED_CONTENT_ROLES = frozenset(
    role
    for role, status in STRUCTURAL_ELIGIBILITY_POLICY.items()
    if status is EligibilityStatus.INELIGIBLE
)
_LEGACY_EXCLUDED_ROLES = frozenset({"navigation", "metadata", "reference"})

_STRUCTURAL_LABELS = frozenset(
    {
        "contents",
        "table of contents",
        "index",
        "glossary",
        "references",
        "reference",
        "copyright",
        "publication information",
        "permissions",
        "legal disclaimer",
        "disclaimer",
    }
)
_STRUCTURAL_WORDS = frozenset(
    {
        "navigation",
        "metadata",
        "reference",
        "references",
        "index",
        "contents",
        "glossary",
        "copyright",
        "permissions",
        "legal",
        "disclaimer",
        "front_matter",
        "back_matter",
    }
)
_LEGAL_PATTERNS = (
    r"all rights reserved",
    r"copyright clearance center",
    r"no part of this .* may be reproduced",
    r"may not be reproduced",
    r"reproduction in any form",
    r"permission to reproduce",
    r"published by",
    r"first published",
    r"isbn(?:[-\s]|\d)",
    r"library of congress",
)


def _normal(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _field_values(chunk: Chunk) -> list[str]:
    values = [
        chunk.structural_type,
        chunk.parent_structure,
        chunk.provenance.section,
        *chunk.metadata.keys(),
        *chunk.metadata.values(),
    ]
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
    section_signal = any(
        term in parent
        for term in ("copyright", "publication", "permissions", "legal", "disclaimer")
    )
    # A single mention of copyright in ordinary prose is intentionally insufficient.
    return matches >= 2 or (copyright_signal and (matches >= 1 or section_signal))


def _looks_like_index(content: str) -> bool:
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if len(lines) < 8:
        return False
    short_lines = sum(len(line.split()) <= 12 for line in lines)
    entry_lines = sum(
        bool(re.match(r"^[A-Za-z][A-Za-z'’ -]{1,45}(?:,|\.{2,}|\s+\d|$)", line))
        for line in lines
    )
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


def infer_legacy_structural_role(chunk: Chunk) -> str | None:
    """Infer a role for chunks created before canonical role metadata."""
    content = chunk.content.strip().lower()
    content_flat = _normal(chunk.content)
    parent = _normal(chunk.parent_structure)

    if _has_structural_metadata(chunk) or _is_contents(chunk, content):
        if any(
            term in parent
            for term in ("copyright", "publication", "permission", "legal")
        ):
            return "metadata"
        if parent in {"glossary", "reference", "references"}:
            return "reference"
        if parent in {"index", "contents", "table of contents"} or _is_contents(
            chunk, content
        ):
            return "navigation"
    if _is_legal_metadata(content_flat, parent):
        return "metadata"
    if _is_reference_or_glossary(chunk, content):
        return "reference"
    return None


def structural_eligible(chunk: Chunk) -> bool:
    """Return whether one canonical chunk is eligible for retrieval.

    Explicit canonical roles use ``STRUCTURAL_ELIGIBILITY_POLICY``. Chunks
    without a role retain the prior conservative structural heuristics for
    backward compatibility; otherwise unknown roles remain eligible.
    """
    return StructuralEligibility().is_eligible(chunk)


class StructuralEligibility:
    """Source-agnostic post-RRF structural eligibility stage.

    Navigation and metadata are initially excluded; documentation,
    implementation, evidence, reference, test, and configuration remain
    eligible. Missing or unrecognized roles are unknown and are retained.
    The stage does not inspect a source system or mutate ranked results.
    """

    policy = STRUCTURAL_ELIGIBILITY_POLICY

    def classify(self, chunk: Chunk) -> EligibilityStatus:
        """Return the policy decision for one canonical chunk."""
        if chunk.content_role:
            role = _normal(chunk.content_role)
            return self.policy.get(role, EligibilityStatus.UNKNOWN)

        # Preserve the existing inferred structural exclusions for old
        # chunks that predate the canonical content_role field.
        inferred = infer_legacy_structural_role(chunk)
        if inferred in _LEGACY_EXCLUDED_ROLES:
            return EligibilityStatus.INELIGIBLE
        return EligibilityStatus.UNKNOWN

    def is_eligible(self, chunk: Chunk) -> bool:
        """Return true for eligible and unknown chunks, false otherwise."""
        return self.classify(chunk) is not EligibilityStatus.INELIGIBLE

    def filter(self, results: Iterable[RankedChunk]) -> list[RankedChunk]:
        """Filter ranked results while preserving objects and original ranks."""
        return [result for result in results if self.is_eligible(result.chunk)]
