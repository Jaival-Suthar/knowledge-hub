from datetime import UTC, datetime
from hashlib import sha256

import pytest

from knowledge_hub.ingestion.articles import (
    Article,
    ArticleCanonicalizationError,
    ArticleChunk,
    ArticleSection,
    article_chunk_to_canonical,
    canonicalize_article,
    document_id_for,
    to_canonical_document,
)
from knowledge_hub.models import Provenance, SourceType

SOURCE_URI = "https://example.hashnode.dev/article"


def make_article() -> Article:
    return Article(
        source="hashnode",
        source_uri=SOURCE_URI,
        article_id="article-1",
        title="Canonical article",
        author="Author",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 3, tzinfo=UTC),
        description="An article description.",
        tags=("python", "architecture"),
        series="Knowledge Hub",
        content="# Canonical article\n\nBody.\n",
        sections=(
            ArticleSection(
                heading="Body",
                heading_level=2,
                heading_path=("Canonical article", "Body"),
                content="Body.",
                position=0,
            ),
        ),
        provenance=Provenance(
            source_uri=SOURCE_URI,
            extra={
                "slug": "article",
                "publication_host": "example.hashnode.dev",
            },
        ),
        source_type=SourceType.ARTICLE,
    )


def make_chunk(
    content: str,
    chunk_id: str,
    index: int = 0,
) -> ArticleChunk:
    return ArticleChunk(
        article_id="article-1",
        source_type=SourceType.ARTICLE,
        source_uri=SOURCE_URI,
        title="Canonical article",
        author="Author",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 3, tzinfo=UTC),
        heading_path=("Canonical article", "Body"),
        heading_level=2,
        chunk_index=index,
        content=content,
        content_type="paragraph",
        token_count=3,
        content_hash=sha256(content.encode("utf-8")).hexdigest(),
        chunk_id=chunk_id,
        block_types=("paragraph",),
        provenance=Provenance(
            source_uri=SOURCE_URI,
            extra={
                "slug": "article",
                "publication_host": "example.hashnode.dev",
            },
        ),
    )


def test_article_maps_to_one_canonical_document() -> None:
    article = make_article()

    document = to_canonical_document(article)

    assert document.source_type is SourceType.ARTICLE
    assert document.source_uri == SOURCE_URI
    assert document.title == article.title
    assert document.content == article.content
    assert document.document_id == document_id_for(article)
    assert document.provenance.source_uri == SOURCE_URI
    assert document.provenance.extra["article_id"] == "article-1"


def test_article_metadata_and_structure_are_preserved() -> None:
    document = to_canonical_document(make_article())

    assert document.metadata["article_id"] == "article-1"
    assert document.metadata["author"] == "Author"
    assert document.metadata["published_at"] == ("2026-01-02T00:00:00+00:00")
    assert document.metadata["updated_at"] == ("2026-01-03T00:00:00+00:00")
    assert document.metadata["tags"] == "python,architecture"
    assert document.metadata["series"] == "Knowledge Hub"
    assert document.metadata["slug"] == "article"

    assert document.structure == {
        "heading_paths": [
            ["Canonical article", "Body"],
        ]
    }


def test_article_chunk_conversion_preserves_content_identity_and_hash() -> None:
    article = make_article()
    document = to_canonical_document(article)

    content = "  exact body\n\nwith whitespace  "
    article_chunk = make_chunk(content, "chunk-1")

    canonical = article_chunk_to_canonical(
        article_chunk,
        document,
    )

    assert canonical.document_id == document.document_id
    assert canonical.chunk_id == "chunk-1"
    assert canonical.content == content
    assert canonical.content_hash == article_chunk.content_hash
    assert canonical.source_type is SourceType.ARTICLE
    assert canonical.source_uri == SOURCE_URI
    assert canonical.structural_type == "paragraph"
    assert canonical.content_role == "paragraph"


def test_article_chunk_provenance_and_heading_context_are_preserved() -> None:
    document = to_canonical_document(make_article())

    canonical = article_chunk_to_canonical(
        make_chunk("body", "chunk-1"),
        document,
    )

    assert canonical.provenance.source_uri == SOURCE_URI
    assert canonical.provenance.section == ("Canonical article > Body")
    assert canonical.provenance.extra["chunk_id"] == "chunk-1"
    assert canonical.provenance.extra["article_id"] == "article-1"
    assert canonical.provenance.extra["chunk_index"] == "0"
    assert canonical.provenance.extra["heading_path"] == ("Canonical article > Body")

    assert canonical.metadata["heading_path"] == ("Canonical article > Body")
    assert canonical.parent_structure == "Canonical article"
    assert canonical.location == ("Canonical article > Body:chunk:0")


def test_repeated_conversion_is_deterministic() -> None:
    document = to_canonical_document(make_article())
    article_chunk = make_chunk("body", "chunk-1")

    first = article_chunk_to_canonical(
        article_chunk,
        document,
    )
    second = article_chunk_to_canonical(
        article_chunk,
        document,
    )

    assert first == second


def test_document_identity_changes_when_article_content_changes() -> None:
    article = make_article()

    changed = Article(
        **{
            **article.__dict__,
            "content": "changed",
        }
    )

    assert document_id_for(article) != document_id_for(changed)


def test_multiple_chunks_keep_order_and_isolated_provenance() -> None:
    article = make_article()

    first = make_chunk("first", "chunk-1", index=0)
    second = make_chunk("second", "chunk-2", index=1)

    document, chunks = canonicalize_article(
        article,
        (first, second),
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        "chunk-1",
        "chunk-2",
    ]
    assert [chunk.content for chunk in chunks] == [
        "first",
        "second",
    ]
    assert [chunk.metadata["chunk_index"] for chunk in chunks] == [
        "0",
        "1",
    ]
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.provenance.source_uri == SOURCE_URI for chunk in chunks)
    assert len({chunk.chunk_id for chunk in chunks}) == 2


def test_mismatched_chunk_source_fails_explicitly() -> None:
    document = to_canonical_document(make_article())

    chunk = ArticleChunk(
        **{
            **make_chunk("body", "chunk-1").__dict__,
            "source_uri": "https://other.example",
        }
    )

    with pytest.raises(
        ArticleCanonicalizationError,
        match="source_uri",
    ):
        article_chunk_to_canonical(
            chunk,
            document,
        )


def test_mismatched_chunk_article_id_fails_explicitly() -> None:
    article = make_article()
    document = to_canonical_document(article)

    chunk = ArticleChunk(
        **{
            **make_chunk("body", "chunk-1").__dict__,
            "article_id": "different-article",
        }
    )

    # Article identity is intentionally carried by the chunk metadata.
    # A chunk belonging to another article must not silently cross the
    # canonical boundary under this document.
    with pytest.raises(
        ArticleCanonicalizationError,
        match="article_id",
    ):
        article_chunk_to_canonical(
            chunk,
            document,
        )


def test_invalid_chunk_content_hash_fails_explicitly() -> None:
    document = to_canonical_document(make_article())

    chunk = ArticleChunk(
        **{
            **make_chunk("body", "chunk-1").__dict__,
            "content_hash": "invalid-hash",
        }
    )

    with pytest.raises(
        ArticleCanonicalizationError,
        match="content_hash",
    ):
        article_chunk_to_canonical(
            chunk,
            document,
        )


def test_mismatched_chunk_provenance_fails_explicitly() -> None:
    document = to_canonical_document(make_article())

    chunk = ArticleChunk(
        **{
            **make_chunk("body", "chunk-1").__dict__,
            "provenance": Provenance(
                source_uri="https://other.example",
                extra={},
            ),
        }
    )

    with pytest.raises(
        ArticleCanonicalizationError,
        match="provenance source_uri",
    ):
        article_chunk_to_canonical(
            chunk,
            document,
        )


def test_mismatched_document_content_fails_explicitly() -> None:
    article = make_article()
    document = to_canonical_document(article)

    mismatched_document = document.model_copy(
        update={"content": "different article content"}
    )

    with pytest.raises(
        ArticleCanonicalizationError,
        match="content does not match",
    ):
        # The supplied document is intentionally invalid for the article.
        from knowledge_hub.ingestion.articles.canonical import (
            to_canonical_chunks,
        )

        to_canonical_chunks(
            article,
            mismatched_document,
            (),
        )
