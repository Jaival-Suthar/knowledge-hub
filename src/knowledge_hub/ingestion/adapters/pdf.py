from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import pymupdf

from knowledge_hub.models import Document, Provenance, SourceType


class PdfAdapter:
    """M1-derived text-layer PDF adapter that emits the canonical Document model."""

    def supports(self, source: str | Path) -> bool:
        return Path(source).suffix.lower() == ".pdf"

    def extract(self, source: str | Path) -> Document:
        path = Path(source)
        pages: list[dict[str, object]] = []

        with pymupdf.open(path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text").strip()

                if not text:
                    continue

                first_line = next(
                    (line.strip() for line in text.splitlines() if line.strip()),
                    None,
                )

                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                        "section": first_line[:80] if first_line else None,
                    }
                )

        document_id = _document_id(path)
        source_uri = path.resolve().as_uri()

        return Document(
            document_id=document_id,
            source_type=SourceType.PDF,
            title=path.stem,
            source_uri=source_uri,
            content="\n\n".join(str(page["text"]) for page in pages),
            metadata={"filename": path.name},
            provenance=Provenance(
                source_uri=source_uri,
                path=str(path),
            ),
            structure={"pages": pages},
        )


def _document_id(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()
