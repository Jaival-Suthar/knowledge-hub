from pathlib import Path

from knowledge_hub.chunking.markdown import MarkdownChunker
from knowledge_hub.ingestion.adapters.markdown import MarkdownAdapter


def markdown_document(tmp_path: Path, content: str):
    path = tmp_path / "guide.md"
    path.write_text(content, encoding="utf-8")
    return MarkdownAdapter().extract(path)


def test_chunks_preserve_nested_heading_paths(tmp_path: Path) -> None:
    document = markdown_document(
        tmp_path,
        "# Authentication\n\nOverview.\n\n## JWT\n\nJWT explanation.\n\n"
        "### Refresh Tokens\n\nRefresh token explanation.\n\n## OAuth\n\nOAuth explanation.",
    )
    chunks = MarkdownChunker().chunk(document)
    assert [chunk.parent_structure for chunk in chunks] == [
        "Authentication",
        "Authentication > JWT",
        "Authentication > JWT > Refresh Tokens",
        "Authentication > OAuth",
    ]
    assert all(chunk.provenance.path == document.provenance.path for chunk in chunks)
    assert all(chunk.source_type == document.source_type for chunk in chunks)


def test_chunks_handle_body_before_headings_and_block_types(tmp_path: Path) -> None:
    document = markdown_document(
        tmp_path,
        "Body first.\n\n# Examples\n\n```python\nprint('x')\n```\n\n"
        "- one\n- two\n\n| A | B |\n| --- | --- |\n| 1 | 2 |\n\n> quote",
    )
    chunks = MarkdownChunker().chunk(document)
    assert chunks[0].parent_structure is None
    assert "Examples" in {chunk.parent_structure for chunk in chunks}
    example = next(chunk for chunk in chunks if chunk.parent_structure == "Examples")
    assert example.structural_type == "section"
    assert set(example.metadata["block_types"].split(",")) >= {
        "code_block",
        "list",
        "table",
        "blockquote",
    }


def test_large_section_splits_without_losing_heading_context(tmp_path: Path) -> None:
    body = " ".join(f"token-{index}" for index in range(300))
    document = markdown_document(
        tmp_path, f"# Authentication\n\n## JWT\n\n### Refresh Tokens\n\n{body}"
    )
    chunks = MarkdownChunker(max_chars=120, overlap_chars=10).chunk(document)
    refresh_chunks = [
        chunk
        for chunk in chunks
        if chunk.parent_structure == "Authentication > JWT > Refresh Tokens"
    ]
    assert len(refresh_chunks) > 1
    assert all(
        chunk.metadata["heading_path"] == "Authentication > JWT > Refresh Tokens"
        for chunk in refresh_chunks
    )


def test_empty_sections_do_not_break_following_sections(tmp_path: Path) -> None:
    document = markdown_document(tmp_path, "# Empty\n\n## Child\n\nChild content")
    chunks = MarkdownChunker().chunk(document)
    assert [chunk.parent_structure for chunk in chunks] == ["Empty > Child"]


def test_chunk_text_preserves_markdown_semantics(tmp_path: Path) -> None:
    document = markdown_document(
        tmp_path,
        "# Fidelity\n\n"
        "source-agnostic indexing and retrieval\n\n"
        "Markdown files enter through the adapter\n\n"
        "Document → Chunk → retrieval\n\n"
        "`MarkdownAdapter` produces the canonical `Document`\n\n"
        "```python\n    return {'value': 1}\n```\n\n"
        "- first item\n- second item\n\n"
        "> quoted text\n\n"
        "| Name | Value |\n| --- | --- |\n| alpha | beta |",
    )
    content = "\n\n".join(chunk.content for chunk in MarkdownChunker().chunk(document))
    assert "source-agnostic indexing and retrieval" in content
    assert "indexingand retrieval" not in content
    assert "Markdown files enter through the adapter" in content
    assert "Document → Chunk → retrieval" in content
    assert "`MarkdownAdapter` produces the canonical `Document`" in content
    assert "```python\n    return {'value': 1}\n```" in content
    assert "- first item\n- second item" in content
    assert "> quoted text" in content
    assert "| alpha | beta |" in content
