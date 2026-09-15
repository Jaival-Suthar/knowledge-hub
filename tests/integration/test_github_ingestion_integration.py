from __future__ import annotations

import os

import pytest

from knowledge_hub.ingestion.github import ingest_github_repository
from knowledge_hub.models import SourceType

pytestmark = pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_HUB_RUN_NETWORK_TESTS") != "1",
    reason="set KNOWLEDGE_HUB_RUN_NETWORK_TESTS=1 to run network integration tests",
)


def test_rippletalk_flows_to_canonical_code_knowledge() -> None:
    result = ingest_github_repository(
        "https://github.com/Jaival-Suthar/RippleTalk",
        ref="main",
    )

    assert result.repository.owner == "Jaival-Suthar"
    assert result.repository.name == "RippleTalk"
    assert result.ref == "main"
    assert result.commit_sha is not None
    assert len(result.commit_sha) == 40
    assert result.files
    assert result.parsed_files
    assert result.semantic_chunks
    assert result.documents
    assert result.chunks

    document = result.documents[0]
    chunk = result.chunks[0]
    assert document.source_type is SourceType.GITHUB
    assert chunk.source_type is SourceType.GITHUB
    assert document.provenance.repository == "Jaival-Suthar/RippleTalk"
    assert document.provenance.branch == "main"
    assert document.provenance.commit_sha == result.commit_sha
    assert chunk.provenance.path
    assert chunk.provenance.symbol
    assert chunk.provenance.line_start is not None
    assert chunk.provenance.line_end is not None
