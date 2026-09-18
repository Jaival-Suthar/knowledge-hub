from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from knowledge_hub.ingestion.articles.hashnode.contracts import HashnodeArticlePayload
from knowledge_hub.models import SourceType

_SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "ingest_corpus.py"
_SPEC = importlib.util.spec_from_file_location("ingest_corpus", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ingest_corpus = importlib.util.module_from_spec(_SPEC)
sys.modules["ingest_corpus"] = ingest_corpus
_SPEC.loader.exec_module(ingest_corpus)


def _canonical_chunk(source_type: SourceType) -> SimpleNamespace:
    return SimpleNamespace(
        source_type=source_type,
        metadata={},
    )


def test_print_summary_reports_canonical_chunk_source_type(capsys) -> None:
    chunk = _canonical_chunk(SourceType.GITHUB)

    ingest_corpus.print_summary([], [chunk])

    output = capsys.readouterr().out
    assert "github          1" in output
    assert "unknown" not in output


def test_ingest_corpus_combines_local_github_and_articles(
    monkeypatch, tmp_path
) -> None:
    local_document = SimpleNamespace(source_type=SourceType.PDF)
    local_chunk = _canonical_chunk(SourceType.CODE)
    github_document = SimpleNamespace(source_type=SourceType.GITHUB)
    github_chunk = _canonical_chunk(SourceType.GITHUB)
    github_result = SimpleNamespace(
        documents=(github_document,),
        chunks=(github_chunk,),
        repository=SimpleNamespace(url=ingest_corpus.GITHUB_URL),
        ref=None,
        commit_sha="a" * 40,
        files=(object(),),
    )
    article_documents = {}
    for url in ingest_corpus.ARTICLE_URLS:
        article_documents[url] = (
            SimpleNamespace(source_type=SourceType.ARTICLE),
            (SimpleNamespace(),),
        )

    monkeypatch.setattr(
        ingest_corpus,
        "ingest_local_corpus",
        lambda _path: ([local_document], [local_chunk]),
    )
    monkeypatch.setattr(
        ingest_corpus,
        "ingest_github_repository",
        lambda _url: github_result,
    )
    monkeypatch.setattr(
        ingest_corpus,
        "ingest_article",
        lambda url: article_documents[url],
    )

    result = ingest_corpus.ingest_corpus(tmp_path)

    assert result.documents == (
        local_document,
        github_document,
        *(document for document, _ in article_documents.values()),
    )
    expected_article_chunks = tuple(
        chunk for _, chunks in article_documents.values() for chunk in chunks
    )
    assert result.chunks == (local_chunk, github_chunk, *expected_article_chunks)
    assert result.github is github_result
    assert result.articles == tuple((url, 1) for url in ingest_corpus.ARTICLE_URLS)


def test_ingest_article_uses_existing_hashnode_to_canonical_pipeline() -> None:
    url = ingest_corpus.ARTICLE_URLS[0]
    payload = HashnodeArticlePayload(
        source_uri=url,
        content="# Real title\n\nArticle content.\n",
        slug="real-title",
        publication_host="jaivalsuthar.hashnode.dev",
    )

    class FakeAcquirer:
        def acquire(self, article_url: str) -> HashnodeArticlePayload:
            assert article_url == url
            return payload

    document, chunks = ingest_corpus.ingest_article(
        url,
        acquirer=FakeAcquirer(),
    )

    assert document.source_type is SourceType.ARTICLE
    assert document.source_uri == url
    assert document.title == "Real title"
    assert chunks
    assert all(chunk.source_type is SourceType.ARTICLE for chunk in chunks)
    assert all(chunk.source_uri == url for chunk in chunks)
    assert all(chunk.provenance.source_uri == url for chunk in chunks)
