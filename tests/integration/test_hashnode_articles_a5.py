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


def test_real_hashnode_articles_reach_canonical_knowledge() -> None:
    acquirer = HashnodeAcquirer()
    parser = ArticleParser()
    chunker = ArticleChunker()
    results = []

    for article_url in ARTICLE_URLS:
        payload = acquirer.acquire(article_url)
        article = parser.parse(payload)
        article_chunks = chunker.chunk(article)
        document, canonical_chunks = canonicalize_article(article, article_chunks)

        assert article.source_uri == article_url
        assert article.title
        assert article.content == payload.content
        assert article.sections
        assert article_chunks
        assert len(canonical_chunks) == len(article_chunks)
        assert document.source_type is SourceType.ARTICLE
        assert document.source_uri == article_url
        assert document.content == article.content
        assert document.provenance.source_uri == article_url

        for article_chunk, canonical_chunk in zip(
            article_chunks, canonical_chunks, strict=True
        ):
            assert canonical_chunk.content == article_chunk.content
            assert canonical_chunk.chunk_id == article_chunk.chunk_id
            assert canonical_chunk.content_hash == article_chunk.content_hash
            assert canonical_chunk.source_uri == article_url
            assert canonical_chunk.provenance.source_uri == article_url
            assert (
                article_chunk_to_canonical(article_chunk, document) == canonical_chunk
            )

        results.append((article_url, document, canonical_chunks))

    assert {document.source_uri for _, document, _ in results} == set(ARTICLE_URLS)
    assert len({document.document_id for _, document, _ in results}) == len(
        ARTICLE_URLS
    )
    for article_url, document, canonical_chunks in results:
        assert document.source_uri == article_url
        assert all(chunk.source_uri == article_url for chunk in canonical_chunks)
        assert all(
            chunk.provenance.source_uri == article_url for chunk in canonical_chunks
        )
