from __future__ import annotations

from hashlib import sha256

import tiktoken

from knowledge_hub.models import Chunk, Document, Provenance


class PdfChunker:
    """M1-compatible page-preserving recursive chunker for text-layer PDFs."""

    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
            raise ValueError("invalid PDF chunk limits")

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def chunk(self, document: Document) -> list[Chunk]:
        pages = document.structure.get("pages", [])
        output: list[Chunk] = []
        index = 0

        for page in pages:
            if not isinstance(page, dict):
                continue

            text = str(page.get("text", "")).strip()
            if not text:
                continue

            page_number = int(page.get("page", 0))
            section = page.get("section")

            for part in self._split(text):
                content = part.strip()

                provenance_data = document.provenance.model_dump()
                provenance_data.update(
                    page=page_number,
                    section=str(section) if section else None,
                )

                output.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_id=sha256(
                            f"{document.document_id}:{index}:{content}".encode()
                        ).hexdigest(),
                        content=content,
                        source_type=document.source_type,
                        source_uri=document.source_uri,
                        language="english",
                        content_role="documentation",
                        parent_structure=str(section) if section else None,
                        location=f"page:{page_number}",
                        token_count=len(self.encoding.encode(content)),
                        content_hash=sha256(content.encode()).hexdigest(),
                        metadata={
                            **document.metadata,
                            "page_number": str(page_number),
                        },
                        provenance=Provenance(**provenance_data),
                    )
                )

                index += 1

        return output

    def _split(self, text: str) -> list[str]:
        if len(self.encoding.encode(text)) <= self.max_tokens:
            return [text]

        pieces = [piece.strip() for piece in text.split("\n\n") if piece.strip()]

        if len(pieces) == 1:
            pieces = [piece.strip() for piece in text.split("\n") if piece.strip()]

        if len(pieces) == 1:
            tokens = self.encoding.encode(text)
            return [
                self.encoding.decode(tokens[start : start + self.max_tokens]).strip()
                for start in range(
                    0,
                    len(tokens),
                    self.max_tokens - self.overlap_tokens,
                )
            ]

        chunks: list[str] = []
        current: list[int] = []

        for piece in pieces:
            tokens = self.encoding.encode(piece)

            if current and len(current) + len(tokens) > self.max_tokens:
                chunks.append(self.encoding.decode(current).strip())

                current = current[-self.overlap_tokens :] if self.overlap_tokens else []

            current.extend(tokens)

        if current:
            chunks.append(self.encoding.decode(current).strip())

        return [chunk for chunk in chunks if chunk]
