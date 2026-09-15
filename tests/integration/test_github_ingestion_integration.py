from __future__ import annotations

import os
from hashlib import sha256

import pytest

from knowledge_hub.ingestion.github import ingest_github_repository
from knowledge_hub.models import SourceType

pytestmark = pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_HUB_RUN_NETWORK_TESTS") != "1",
    reason="set KNOWLEDGE_HUB_RUN_NETWORK_TESTS=1 to run network integration tests",
)


def test_rippletalk_flows_to_canonical_code_knowledge() -> None:
    repository_url = "https://github.com/Jaival-Suthar/RippleTalk"
    result = ingest_github_repository(
        repository_url,
        ref="main",
    )

    assert result.repository.owner == "Jaival-Suthar"
    assert result.repository.name == "RippleTalk"
    assert result.repository.url == repository_url
    assert result.ref == "main"
    assert result.commit_sha is not None
    assert len(result.commit_sha) == 40
    assert result.commit_sha == result.commit_sha.lower()
    assert all(character in "0123456789abcdef" for character in result.commit_sha)
    assert result.files
    assert result.parsed_files
    assert result.semantic_chunks
    assert result.documents
    assert result.chunks
    assert len(result.files) > 1
    assert len(result.parsed_files) > 1
    assert len(result.documents) > 1

    file_result = next(
        file_result
        for file_result in result.file_results
        if file_result.semantic_chunks and file_result.canonical_chunks
    )
    code_file = file_result.code_file
    semantic_chunk = file_result.semantic_chunks[0]
    document = file_result.document
    chunk = file_result.canonical_chunks[0]
    repository_name = "Jaival-Suthar/RippleTalk"

    assert document.source_type is SourceType.GITHUB
    assert document.source_uri == repository_url
    assert document.title == code_file.relative_path
    assert document.content == code_file.content
    assert document.metadata["relative_path"] == code_file.relative_path
    assert document.metadata["language"] == code_file.language.value
    assert document.metadata["content_hash"] == code_file.content_hash
    assert document.structure["relative_path"] == code_file.relative_path
    assert document.provenance.source_uri == repository_url
    assert document.provenance.repository == repository_name
    assert document.provenance.branch == "main"
    assert document.provenance.commit_sha == result.commit_sha
    assert document.provenance.path == code_file.relative_path

    assert chunk.document_id == document.document_id
    assert chunk.source_type is SourceType.GITHUB
    assert chunk.source_uri == document.source_uri
    assert chunk.language == code_file.language.value
    assert chunk.language == semantic_chunk.language.value
    assert chunk.structural_type == semantic_chunk.symbol.symbol_type.value
    assert chunk.content == semantic_chunk.content
    assert chunk.content_hash == sha256(chunk.content.encode("utf-8")).hexdigest()
    assert chunk.provenance.source_uri == repository_url
    assert chunk.provenance.repository == repository_name
    assert chunk.provenance.branch == "main"
    assert chunk.provenance.commit_sha == result.commit_sha
    assert chunk.provenance.path == code_file.relative_path
    assert chunk.provenance.symbol == semantic_chunk.symbol.qualified_name
    assert chunk.provenance.line_start == semantic_chunk.start_line
    assert chunk.provenance.line_end == semantic_chunk.end_line
    assert semantic_chunk.start_line <= semantic_chunk.end_line
    assert chunk.provenance.line_start <= chunk.provenance.line_end
    assert code_file.relative_path in (chunk.location or "")

    assert semantic_chunk.symbol.name
    assert semantic_chunk.symbol.qualified_name
    assert semantic_chunk.symbol.symbol_type

    for file_result in result.file_results:
        assert file_result.document.source_type is SourceType.GITHUB
        assert file_result.document.provenance.commit_sha == result.commit_sha
        assert all(
            canonical_chunk.document_id == file_result.document.document_id
            for canonical_chunk in file_result.canonical_chunks
        )

    assert all(
        document.provenance.commit_sha == result.commit_sha
        for document in result.documents
    )
    assert all(
        chunk.provenance.commit_sha == result.commit_sha for chunk in result.chunks
    )
