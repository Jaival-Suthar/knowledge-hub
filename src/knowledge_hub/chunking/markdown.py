from __future__ import annotations

from hashlib import sha256

from knowledge_hub.chunking.recursive import RecursiveChunker
from knowledge_hub.models import Chunk, Document, Provenance


class MarkdownChunker:
    """Chunk Markdown sections while retaining their heading context."""

    def __init__(self, max_chars: int = 4000, overlap_chars: int = 400) -> None:
        self._recursive = RecursiveChunker(max_chars=max_chars, overlap_chars=overlap_chars)

    def chunk(self, document: Document) -> list[Chunk]:
        sections = self._sections(document)
        output: list[Chunk] = []
        for section_index, section in enumerate(sections):
            pieces = self._recursive._split_preserving("\n\n".join(section["lines"]))
            for part_index, content in enumerate(pieces):
                content = content.strip()
                if not content:
                    continue
                path = section["heading_path"]
                path_text = " > ".join(path) if path else None
                block_types = sorted(set(section["block_types"]))
                structural_type = block_types[0] if len(block_types) == 1 else "section"
                metadata = dict(document.metadata)
                metadata.update(
                    {
                        "heading_path": path_text or "",
                        "heading_level": str(section["heading_level"] or ""),
                        "block_types": ",".join(block_types),
                    }
                )
                output.append(
                    Chunk(
                        document_id=document.document_id,
                        chunk_id=sha256(
                            f"{document.document_id}:{section_index}:{part_index}:{content}".encode()
                        ).hexdigest(),
                        content=content,
                        source_type=document.source_type,
                        source_uri=document.source_uri,
                        project=document.metadata.get("project"),
                        language=document.metadata.get("language"),
                        content_role=(
                            "implementation"
                            if block_types == ["code_block"]
                            else "documentation"
                        ),
                        structural_type=structural_type,
                        parent_structure=path_text,
                        location=f"section:{section_index}:part:{part_index}",
                        token_count=len(content.split()),
                        content_hash=sha256(content.encode()).hexdigest(),
                        metadata=metadata,
                        provenance=Provenance(**document.provenance.model_dump()),
                    )
                )
        return output

    @staticmethod
    def _sections(document: Document) -> list[dict[str, object]]:
        blocks = document.structure.get("blocks", [])
        sections: list[dict[str, object]] = []
        current: dict[str, object] | None = None

        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type", "paragraph"))
            path = [str(item) for item in block.get("heading_path", [])]
            if block_type == "heading":
                current = {
                    "heading_path": path,
                    "heading_level": int(block.get("level", 0)),
                    "lines": [f"{'#' * int(block.get('level', 0))} {block.get('text', '')}"],
                    "block_types": ["heading"],
                }
                sections.append(current)
                continue

            if current is None or current["heading_path"] != path:
                current = {
                    "heading_path": path,
                    "heading_level": len(path) or None,
                    "lines": [],
                    "block_types": [],
                }
                sections.append(current)
            current["lines"].append(str(block.get("text", "")))
            current["block_types"].append(block_type)

        return [
            section
            for section in sections
            if section["lines"] and section["block_types"] != ["heading"]
        ]
