from __future__ import annotations

import os

import pytest

from knowledge_hub.ingestion.articles import (
    ArticleChunker,
    ArticleParser,
    article_chunk_to_canonical,
    canonicalize_article,
)
from knowledge_hub.ingestion.articles.hashnode import HashnodeAcquirer
from knowledge_hub.models import SourceType

pytestmark = pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_HUB_RUN_NETWORK_TESTS") != "1",
    reason="set KNOWLEDGE_HUB_RUN_NETWORK_TESTS=1 to run network integration tests",
)

ARTICLE_URLS = (
    "https://jaivalsuthar.hashnode.dev/what-google-docs-taught-us-about-building-the-impossible",
    "https://jaivalsuthar.hashnode.dev/building-a-local-llm-benchmark-harness-with-typescript-ollama-sqlite",
    "https://jaivalsuthar.hashnode.dev/dual-storage-for-time-designing-timezone-safe-systems-at-scale",
)


def _process_payload(payload):
    article = ArticleParser().parse(payload)
    article_chunks = ArticleChunker().chunk(article)
    document, canonical_chunks = canonicalize_article(article, article_chunks)
    return article, article_chunks, document, canonical_chunks


def test_three_real_hashnode_articles_pass_the_a6_confidence_gate() -> None:
    acquirer = HashnodeAcquirer()
    results = []

    for article_url in ARTICLE_URLS:
        payload = acquirer.acquire(article_url)
        assert payload.content
        assert payload.source_uri == article_url

        first = _process_payload(payload)
        second = _process_payload(payload)
        article, article_chunks, document, canonical_chunks = first
        repeated_article, repeated_chunks, repeated_document, repeated_canonical = (
            second
        )

        assert article.source == "hashnode"
        assert article.source_type is SourceType.ARTICLE
        assert article.source_uri == article_url
        assert article.title
        assert article.content
        assert article.content == payload.content
        assert article.sections

        assert article_chunks
        assert len(canonical_chunks) == len(article_chunks)
        assert document.source_type is SourceType.ARTICLE
        assert document.source_uri == article.source_uri
        assert document.content == article.content
        assert document.provenance.source_uri == article.source_uri
        assert document.structure.get("heading_paths")

        assert repeated_article == article
        assert repeated_chunks == article_chunks
        assert repeated_document == document
        assert repeated_canonical == canonical_chunks

        assert len({chunk.chunk_id for chunk in canonical_chunks}) == len(
            canonical_chunks
        )
        assert [chunk.metadata["chunk_index"] for chunk in canonical_chunks] == [
            str(article_chunk.chunk_index) for article_chunk in article_chunks
        ]

        for article_chunk, canonical_chunk in zip(
            article_chunks, canonical_chunks, strict=True
        ):
            assert canonical_chunk.document_id == document.document_id
            assert canonical_chunk.source_type is SourceType.ARTICLE
            assert canonical_chunk.source_uri == article.source_uri
            assert canonical_chunk.content == article_chunk.content
            assert canonical_chunk.content_hash == article_chunk.content_hash
            assert canonical_chunk.chunk_id == article_chunk.chunk_id
            assert canonical_chunk.structural_type
            assert canonical_chunk.content_role
            assert canonical_chunk.provenance.source_uri == article.source_uri
            assert (
                canonical_chunk.provenance.extra["chunk_id"] == article_chunk.chunk_id
            )
            assert canonical_chunk.provenance.extra["chunk_index"] == str(
                article_chunk.chunk_index
            )
            assert canonical_chunk.provenance.extra["heading_path"] == " > ".join(
                article_chunk.heading_path
            )
            assert (
                article_chunk_to_canonical(article_chunk, document) == canonical_chunk
            )

        results.append((article_url, document, canonical_chunks))

    assert len({url for url, _, _ in results}) == len(ARTICLE_URLS)
    assert len({document.document_id for _, document, _ in results}) == len(
        ARTICLE_URLS
    )
    for article_url, document, canonical_chunks in results:
        assert document.source_uri == article_url
        assert all(chunk.source_uri == article_url for chunk in canonical_chunks)
        assert all(
            chunk.provenance.source_uri == article_url for chunk in canonical_chunks
        )
