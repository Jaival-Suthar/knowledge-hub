from knowledge_hub.models import Chunk, Document, Provenance, SourceType


def test_canonical_models() -> None:
    p = Provenance(source_uri="file:///example.md", path="example.md")
    d = Document(
        document_id="doc-1",
        source_type=SourceType.MARKDOWN,
        title="Example",
        provenance=p,
    )
    c = Chunk(
        document_id=d.document_id,
        chunk_id="chunk-1",
        content="Example",
        source_type=d.source_type,
        provenance=p,
        content_role="documentation",
    )
    assert c.provenance.path == "example.md"
