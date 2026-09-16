"""Deterministic BM25 retrieval over canonical knowledge chunks."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from rank_bm25 import BM25Plus

from knowledge_hub.models import Chunk
from knowledge_hub.retrieval.types import RankedChunk

_TOKEN = re.compile(r"[A-Za-z0-9]+(?:[._$:/@-][A-Za-z0-9]+)*")
_SEPARATOR = re.compile(r"[._$:/@-]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _tokens(text: str) -> list[str]:
    """Tokenize prose and technical identifiers without discarding either.

    Complete identifiers are retained (for example ``refreshToken`` and
    ``github_acquisition.py``), while separator-delimited and camel-case
    components are also indexed so natural-language queries can find them.
    All tokens are case-folded for case-insensitive lexical matching.
    """
    output: list[str] = []
    seen: set[str] = set()

    for match in _TOKEN.finditer(text):
        raw = match.group(0)
        candidates = [raw]
        for segment in _SEPARATOR.split(raw):
            if not segment:
                continue
            candidates.append(segment)
            candidates.extend(_CAMEL_BOUNDARY.split(segment))

        for candidate in candidates:
            token = candidate.casefold()
            if token and token not in seen:
                seen.add(token)
                output.append(token)

    return output


def _chunk_text(chunk: Chunk) -> str:
    """Build the indexed text while retaining canonical source context."""
    values = [
        chunk.content,
        chunk.source_uri or "",
        chunk.project or "",
        chunk.timestamp or "",
        chunk.language or "",
        chunk.content_role or "",
        chunk.structural_type or "",
        chunk.parent_structure or "",
        chunk.location or "",
        *chunk.metadata.keys(),
        *chunk.metadata.values(),
        chunk.provenance.source_uri or "",
        chunk.provenance.repository or "",
        chunk.provenance.branch or "",
        chunk.provenance.commit_sha or "",
        chunk.provenance.path or "",
        chunk.provenance.section or "",
        chunk.provenance.symbol or "",
        *chunk.provenance.extra.keys(),
        *chunk.provenance.extra.values(),
    ]
    return " ".join(values)


@dataclass(frozen=True)
class _IndexedChunk:
    chunk: Chunk
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    position: int


class BM25Index:
    """Rebuildable in-memory index hiding the third-party BM25 implementation."""

    def __init__(self, chunks: Iterable[Chunk] = ()) -> None:
        self._entries: tuple[_IndexedChunk, ...] = ()
        self._model: BM25Plus | None = None
        self.rebuild(chunks)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """Return the indexed chunks without exposing mutable index state."""
        return tuple(entry.chunk for entry in self._entries)

    def rebuild(self, chunks: Iterable[Chunk]) -> None:
        """Replace the corpus with a deterministic snapshot of ``chunks``."""
        entries: list[_IndexedChunk] = []
        for position, chunk in enumerate(chunks):
            tokens = tuple(_tokens(_chunk_text(chunk)))
            entries.append(
                _IndexedChunk(
                    chunk=chunk,
                    tokens=tokens,
                    token_set=frozenset(tokens),
                    position=position,
                )
            )

        self._entries = tuple(entries)
        self._model = (
            BM25Plus([list(entry.tokens) for entry in self._entries])
            if self._entries
            else None
        )

    def update(self, chunks: Iterable[Chunk]) -> None:
        """Replace the indexed corpus; incremental synchronization is deferred."""
        self.rebuild(chunks)

    def search(self, query: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Return matching chunks and scores in deterministic rank order."""
        if top_k <= 0 or self._model is None:
            return []

        query_tokens = _tokens(query)
        if not query_tokens:
            return []

        query_terms = frozenset(query_tokens)
        scores = self._model.get_scores(query_tokens)
        candidates = [
            (entry, float(scores[entry.position]))
            for entry in self._entries
            if entry.token_set.intersection(query_terms)
        ]
        candidates.sort(
            key=lambda item: (
                -item[1],
                item[0].chunk.chunk_id,
                item[0].chunk.document_id,
                item[0].position,
            )
        )
        return [(entry.chunk, score) for entry, score in candidates[:top_k]]


class BM25Retriever:
    """Independent sparse retrieval over canonical ``Chunk`` objects."""

    def __init__(
        self,
        chunks: Iterable[Chunk] = (),
        *,
        index: BM25Index | None = None,
    ) -> None:
        chunk_snapshot = tuple(chunks)
        if index is not None and chunk_snapshot:
            raise ValueError("provide chunks or index, not both")
        self._index = index or BM25Index(chunk_snapshot)

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return self._index.chunks

    def index(self, chunks: Iterable[Chunk]) -> None:
        """Build or replace the in-memory BM25 corpus."""
        self._index.rebuild(chunks)

    def rebuild(self, chunks: Iterable[Chunk]) -> None:
        """Alias for ``index`` for callers that prefer explicit terminology."""
        self.index(chunks)

    def update(self, chunks: Iterable[Chunk]) -> None:
        """Replace the current corpus without implying incremental sync."""
        self.index(chunks)

    def search(self, query: str, top_k: int = 10) -> list[RankedChunk]:
        """Return deterministic BM25 results with one-based ranks."""
        return [
            RankedChunk(
                chunk=chunk,
                score=score,
                rank=rank,
                channel="bm25",
            )
            for rank, (chunk, score) in enumerate(
                self._index.search(query, top_k),
                start=1,
            )
        ]
