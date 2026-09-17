"""Source-agnostic filtering over canonical chunk metadata."""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass
from enum import StrEnum

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.types import RankedChunk

MetadataValue = str | StrEnum | Collection[str | StrEnum]


@dataclass(frozen=True)
class MetadataFilters:
    """AND-combined exact metadata constraints for canonical chunks."""

    source_type: MetadataValue | None = None
    project: MetadataValue | None = None
    language: MetadataValue | None = None
    repository: MetadataValue | None = None
    path: MetadataValue | None = None
    content_role: MetadataValue | None = None


class MetadataFilter:
    """Filter ranked chunks without changing their order or contents.

    Values within one dimension use OR semantics; supplied dimensions use AND
    semantics. Comparisons are trimmed and case-insensitive, while returned
    chunks retain their original metadata.
    """

    _DIMENSIONS = (
        "source_type",
        "project",
        "language",
        "repository",
        "path",
        "content_role",
    )

    def filter(
        self,
        results: Iterable[RankedChunk],
        filters: MetadataFilters | None = None,
    ) -> list[RankedChunk]:
        """Return chunks matching all supplied constraints."""
        if filters is None:
            return list(results)
        return [result for result in results if self.matches(result.chunk, filters)]

    def matches(self, chunk: Chunk, filters: MetadataFilters) -> bool:
        """Return whether a canonical chunk satisfies the metadata filters."""
        values = {
            "source_type": chunk.source_type,
            "project": chunk.project,
            "language": chunk.language,
            "repository": chunk.provenance.repository,
            "path": chunk.provenance.path,
            "content_role": chunk.content_role,
        }
        return all(
            constraint is None
            or _normal(values[dimension]) in _normal_values(constraint)
            for dimension in self._DIMENSIONS
            if (constraint := getattr(filters, dimension)) is not None
        )


def _normal(value: object) -> str:
    return str(value).strip().casefold() if value is not None else ""


def _normal_values(value: MetadataValue) -> frozenset[str]:
    if isinstance(value, (str, StrEnum)):
        return frozenset({_normal(value)})
    return frozenset(_normal(item) for item in value)
