from pathlib import Path

from knowledge_hub.ingestion.adapters.markdown import MarkdownAdapter
from knowledge_hub.models import SourceType


def write_markdown(tmp_path: Path, name: str = "guide.md", content: str = "# Guide\n\nBody") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_markdown_extensions_and_rejection(tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    assert adapter.supports(write_markdown(tmp_path))
    assert adapter.supports(write_markdown(tmp_path, "guide.markdown"))
    assert not adapter.supports(tmp_path / "guide.txt")


def test_markdown_document_identity_and_content(tmp_path: Path) -> None:
    adapter = MarkdownAdapter()
    first = write_markdown(tmp_path, content="# Guide\n\nBody")
    same = write_markdown(tmp_path, "same.md", "# Guide\n\nBody")
    changed = write_markdown(tmp_path, "changed.md", "# Guide\n\nChanged")

    document = adapter.extract(first)
    assert document.document_id == adapter.extract(same).document_id
    assert document.document_id != adapter.extract(changed).document_id
    assert document.content == first.read_bytes().decode("utf-8")
    assert document.source_type is SourceType.MARKDOWN
    assert document.source_uri == first.resolve().as_uri()
    assert document.provenance.source_uri == document.source_uri
    assert document.provenance.path == str(first)
    assert document.metadata["filename"] == "guide.md"
    assert document.metadata["path"] == str(first)


def test_markdown_structure_and_heading_paths(tmp_path: Path) -> None:
    content = (
        "Text before headings.\n\n# A\n\n## B\n\n### C\n\n## D\n\n# E\n\n"
        "```python\nprint('x')\n```\n\n- one\n- two\n\n> quote\n\n"
        "| Name | Value |\n| --- | --- |\n| A | 1 |"
    )
    blocks = MarkdownAdapter().extract(write_markdown(tmp_path, content=content)).structure["blocks"]
    headings = [block for block in blocks if block["type"] == "heading"]
    assert [(item["level"], item["heading_path"]) for item in headings] == [
        (1, ["A"]),
        (2, ["A", "B"]),
        (3, ["A", "B", "C"]),
        (2, ["A", "D"]),
        (1, ["E"]),
    ]
    assert blocks[0]["heading_path"] == []
    assert {block["type"] for block in blocks} >= {
        "code_block", "list", "table", "blockquote", "paragraph"
    }
