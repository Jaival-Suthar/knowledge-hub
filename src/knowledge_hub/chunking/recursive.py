from __future__ import annotations

import re
from hashlib import sha256

from knowledge_hub.models import Chunk, Document, Provenance


class RecursiveChunker:
    """General-purpose text chunker derived from the M1 recursive strategy."""

    def __init__(self, max_chars: int = 4000, overlap_chars: int = 400) -> None:
        if max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if overlap_chars < 0 or overlap_chars >= max_chars:
            raise ValueError("overlap_chars must be >= 0 and < max_chars")
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        for index, text in enumerate(self._split(document.content)):
            content = text.strip()
            if not content:
                continue
            chunks.append(
                Chunk(
                    document_id=document.document_id,
                    chunk_id=sha256(
                        f"{document.document_id}:{index}:{content}".encode()
                    ).hexdigest(),
                    content=content,
                    source_type=document.source_type,
                    source_uri=document.source_uri,
                    project=document.metadata.get("project"),
                    language=document.metadata.get("language"),
                    content_role=document.metadata.get("content_role", "documentation"),
                    structural_type="prose",
                    location=f"chunk:{index}",
                    token_count=len(re.findall(r"\S+", content)),
                    content_hash=sha256(content.encode()).hexdigest(),
                    metadata=dict(document.metadata),
                    provenance=Provenance(**document.provenance.model_dump()),
                )
            )
        return chunks

    def _split(self, text: str) -> list[str]:
        text = re.sub(r"[ \t]+", " ", text).strip()
        if len(text) <= self.max_chars:
            return [text]

        pieces = re.split(r"(?<=\n)\n+|(?<=[.!?])\s+", text)
        pieces = [piece.strip() for piece in pieces if piece.strip()]
        if not pieces:
            pieces = [text]

        output: list[str] = []
        current = ""
        for piece in pieces:
            candidate = f"{current} {piece}".strip()
            if current and len(candidate) > self.max_chars:
                output.append(current)
                current = current[-self.overlap_chars :] if self.overlap_chars else ""
                candidate = f"{current} {piece}".strip()
            if len(candidate) <= self.max_chars:
                current = candidate
            else:
                for start in range(0, len(piece), self.max_chars - self.overlap_chars):
                    output.append(piece[start : start + self.max_chars])
                current = ""
        if current:
            output.append(current)
        return output

    def _split_preserving(self, text: str) -> list[str]:
        """Split while retaining source whitespace and block formatting."""
        source = text.strip()
        if len(source) <= self.max_chars:
            return [source]

        blocks = re.split(r"(\n\s*\n)", source)
        pieces: list[str] = []
        current = ""
        for block in blocks:
            candidate = f"{current}{block}"
            if current and len(candidate) > self.max_chars:
                pieces.extend(self._hard_split_preserving(current))
                current = block
            else:
                current = candidate
        if current:
            pieces.extend(self._hard_split_preserving(current))
        return [piece for piece in pieces if piece.strip()]

    def _hard_split_preserving(self, text: str) -> list[str]:
        pieces: list[str] = []
        remaining = text
        while len(remaining) > self.max_chars:
            cut = remaining.rfind(" ", 1, self.max_chars + 1)
            if cut <= 0:
                cut = remaining.rfind("\n", 1, self.max_chars + 1)
            if cut <= 0:
                cut = self.max_chars
            pieces.append(remaining[:cut])
            remaining = remaining[cut:]
        if remaining:
            pieces.append(remaining)
        return pieces
