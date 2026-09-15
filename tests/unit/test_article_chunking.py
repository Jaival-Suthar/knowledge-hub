from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256

from knowledge_hub.ingestion.articles import (
    Article,
    ArticleBlock,
    ArticleBlockType,
    ArticleChunker,
    ArticleSection,
)
from knowledge_hub.models import Provenance, SourceType


def make_article(sections: tuple[ArticleSection, ...]) -> Article:
    return Article(
        source="hashnode",
        source_uri="https://example.hashnode.dev/article",
        article_id=None,
        title="Article title",
        author="Author",
        published_at=datetime(2026, 1, 2, tzinfo=UTC),
        updated_at=datetime(2026, 1, 3, tzinfo=UTC),
        content="source markdown",
        sections=sections,
        provenance=Provenance(
            source_uri="https://example.hashnode.dev/article",
            extra={"slug": "article", "publication_host": "example.hashnode.dev"},
        ),
        source_type=SourceType.ARTICLE,
    )


def make_section(
    heading: str,
    path: tuple[str, ...],
    *blocks: ArticleBlock,
    level: int = 2,
    position: int = 0,
) -> ArticleSection:
    heading_block = ArticleBlock(
        type=ArticleBlockType.HEADING,
        content=heading,
        position=position,
    )
    return ArticleSection(
        heading=heading,
        heading_level=level,
        heading_path=path,
        content="\n\n".join(block.content for block in blocks),
        position=position,
        blocks=(heading_block, *blocks),
    )


def test_chunks_group_blocks_within_sections_and_preserve_context() -> None:
    article = make_article(
        (
            make_section(
                "Retrieval",
                ("Knowledge Hub", "Retrieval"),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "Dense retrieval."),
                ArticleBlock(
                    ArticleBlockType.LIST,
                    "one\ntwo",
                    items=("one", "two"),
                ),
            ),
            make_section(
                "Evaluation",
                ("Knowledge Hub", "Evaluation"),
                ArticleBlock(ArticleBlockType.QUOTE, "Measure what matters."),
                position=1,
            ),
        )
    )

    chunks = ArticleChunker(max_tokens=100).chunk(article)

    assert len(chunks) == 2
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert chunks[0].heading_path == ("Knowledge Hub", "Retrieval")
    assert chunks[0].heading_level == 2
    assert chunks[0].content == "Dense retrieval.\n\none\ntwo"
    assert chunks[0].block_types == ("paragraph", "list")
    assert chunks[1].heading_path == ("Knowledge Hub", "Evaluation")
    assert chunks[1].content == "Measure what matters."


def test_code_table_and_link_content_are_preserved() -> None:
    blocks = (
        ArticleBlock(
            ArticleBlockType.CODE,
            'print("hello")\n',
            language="python",
        ),
        ArticleBlock(
            ArticleBlockType.TABLE,
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
            rows=(("A", "B"), ("1", "2")),
        ),
        ArticleBlock(
            ArticleBlockType.LINK,
            "Read more at [Docs](https://docs.example)",
            url="https://docs.example",
        ),
    )
    chunk = ArticleChunker(max_tokens=100).chunk(
        make_article((make_section("Examples", ("Examples",), *blocks),))
    )[0]

    assert 'print("hello")\n' in chunk.content
    assert "| A | B |" in chunk.content
    assert "Read more at [Docs](https://docs.example)" in chunk.content
    assert chunk.block_types == ("code", "table", "link")
    assert chunk.content_type == "section"


def test_oversized_block_is_split_by_tokens_without_content_loss() -> None:
    content = " ".join(f"word-{index}" for index in range(100))
    article = make_article(
        (
            make_section(
                "Large",
                ("Large",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, content),
            ),
        )
    )

    chunks = ArticleChunker(max_tokens=16).chunk(article)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 16 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content
    assert all(chunk.heading_path == ("Large",) for chunk in chunks)


def test_oversized_code_block_is_split_without_dropping_source() -> None:
    content = "".join(f"value_{index} = {index}\n" for index in range(80))
    article = make_article(
        (
            make_section(
                "Code",
                ("Code",),
                ArticleBlock(ArticleBlockType.CODE, content, language="python"),
            ),
        )
    )

    chunks = ArticleChunker(max_tokens=16).chunk(article)

    assert len(chunks) > 1
    assert all(chunk.token_count <= 16 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content
    assert all(chunk.content_type == "code" for chunk in chunks)


def test_oversized_text_preserves_whitespace_exactly() -> None:
    content = "  hello   world\n\n\tvalue  "
    article = make_article(
        (
            make_section(
                "Whitespace",
                ("Whitespace",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, content),
            ),
        )
    )

    chunks = ArticleChunker(max_tokens=2).chunk(article)

    assert all(chunk.token_count <= 2 for chunk in chunks)
    assert "".join(chunk.content for chunk in chunks) == content


def test_chunk_identity_and_metadata_are_deterministic() -> None:
    article = make_article(
        (
            make_section(
                "Section",
                ("Section",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "Stable content."),
            ),
        )
    )
    first = ArticleChunker().chunk(article)
    second = ArticleChunker().chunk(article)

    assert first == second
    assert first[0].source_type is SourceType.ARTICLE
    assert first[0].source_uri == article.source_uri
    assert first[0].title == article.title
    assert first[0].author == article.author
    assert first[0].published_at == article.published_at
    assert first[0].updated_at == article.updated_at
    assert first[0].provenance == article.provenance
    assert first[0].content_hash == sha256(first[0].content.encode("utf-8")).hexdigest()


def test_chunk_identity_is_stable_when_global_order_shifts() -> None:
    first_article = make_article(
        (
            make_section(
                "Section A",
                ("Section A",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "alpha", position=0),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "beta", position=1),
                position=0,
            ),
            make_section(
                "Section B",
                ("Section B",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "gamma", position=0),
                position=1,
            ),
        )
    )
    changed_article = make_article(
        (
            make_section(
                "Section A",
                ("Section A",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "alpha", position=0),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "new", position=1),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "beta", position=2),
                position=0,
            ),
            make_section(
                "Section B",
                ("Section B",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "gamma", position=0),
                position=1,
            ),
        )
    )

    first_chunks = ArticleChunker(max_tokens=1).chunk(first_article)
    changed_chunks = ArticleChunker(max_tokens=1).chunk(changed_article)
    first_beta = next(chunk for chunk in first_chunks if chunk.content == "beta")
    changed_beta = next(chunk for chunk in changed_chunks if chunk.content == "beta")

    assert changed_beta.chunk_index > first_beta.chunk_index
    assert changed_beta.content == first_beta.content
    assert changed_beta.content_hash == first_beta.content_hash
    assert changed_beta.chunk_id == first_beta.chunk_id


def test_changed_content_changes_chunk_identity_and_hash() -> None:
    first_article = make_article(
        (
            make_section(
                "Section",
                ("Section",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "Original content."),
            ),
        )
    )
    changed_article = make_article(
        (
            make_section(
                "Section",
                ("Section",),
                ArticleBlock(ArticleBlockType.PARAGRAPH, "Changed content."),
            ),
        )
    )

    first = ArticleChunker().chunk(first_article)[0]
    changed = ArticleChunker().chunk(changed_article)[0]

    assert first.chunk_id != changed.chunk_id
    assert first.content_hash != changed.content_hash


def test_empty_article_and_empty_sections_produce_no_chunks() -> None:
    empty_article = make_article(())
    empty_section = make_section("Empty", ("Empty",))
    empty_article_with_section = make_article((empty_section,))

    assert ArticleChunker().chunk(empty_article) == ()
    assert ArticleChunker().chunk(empty_article_with_section) == ()
