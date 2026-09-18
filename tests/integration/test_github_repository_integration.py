from __future__ import annotations

import os
from pathlib import PurePosixPath

import pytest

from knowledge_hub.ingestion.code import (
    CodeChunker,
    CodeFile,
    ParserRegistry,
    canonicalize_code,
)
from knowledge_hub.ingestion.github import (
    GitHubRepositorySource,
    discover_github_repository,
    github_provenance,
    ingest_github_repository,
)
from knowledge_hub.models import SourceType

pytestmark = pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_HUB_RUN_NETWORK_TESTS") != "1",
    reason="set KNOWLEDGE_HUB_RUN_NETWORK_TESTS=1 to run network integration tests",
)


def test_real_github_repository_flows_through_code_discovery() -> None:
    result = discover_github_repository(
        GitHubRepositorySource(url="https://github.com/pallets/markupsafe")
    )

    try:
        assert result.repository.owner == "pallets"
        assert result.repository.name == "markupsafe"
        assert result.snapshot.root_path.is_dir()
        assert result.files
        assert all(isinstance(code_file, CodeFile) for code_file in result.files)
        assert all(code_file.relative_path for code_file in result.files)
        assert all(
            not str(code_file.relative_path).startswith(str(result.snapshot.root_path))
            for code_file in result.files
        )
        assert all(
            ".git" not in PurePosixPath(code_file.relative_path).parts
            for code_file in result.files
        )
    finally:
        result.cleanup()

    assert not result.snapshot.root_path.exists()


def test_rippletalk_code_artifact_preserves_github_provenance() -> None:
    result = discover_github_repository(
        GitHubRepositorySource(
            url="https://github.com/Jaival-Suthar/RippleTalk",
            ref="main",
        )
    )
    registry = ParserRegistry.default()
    chunker = CodeChunker()

    try:
        assert result.repository.owner == "Jaival-Suthar"
        assert result.repository.name == "RippleTalk"
        for code_file in result.files:
            parsed = registry.get(code_file.language).parse(code_file)
            semantic_chunks = chunker.chunk(code_file, parsed)
            if not semantic_chunks:
                continue

            document, chunks = canonicalize_code(
                code_file,
                semantic_chunks,
                source_type=SourceType.GITHUB,
                provenance=github_provenance(result.snapshot),
            )
            chunk = chunks[0]
            assert document.source_type is SourceType.GITHUB
            assert document.source_uri == result.repository.url
            assert document.provenance.repository == "Jaival-Suthar/RippleTalk"
            assert document.provenance.branch == "main"
            assert document.provenance.commit_sha == result.snapshot.commit_sha
            assert chunk.provenance.path == code_file.relative_path
            assert chunk.provenance.symbol
            assert chunk.provenance.line_start is not None
            assert chunk.provenance.line_end is not None
            return

        pytest.fail("RippleTalk produced no semantic code artifact")
    finally:
        result.cleanup()


def test_rippletalk_tsx_files_parse_into_semantic_chunks() -> None:
    result = ingest_github_repository(
        "https://github.com/Jaival-Suthar/RippleTalk",
        ref="main",
    )

    expected_paths = (
        "src/App.tsx",
        "src/components/Navbar.tsx",
        "src/context/AuthProvider.tsx",
        "src/pages/home/page.tsx",
        "src/pages/login/page.tsx",
        "src/pages/registration/page.tsx",
    )

    files_by_path = {item.code_file.relative_path: item for item in result.file_results}
    for path in expected_paths:
        file_result = files_by_path[f"rippletalk/{path}"]
        assert file_result.parse_result.errors == ()

    app_result = files_by_path["rippletalk/src/App.tsx"]
    assert app_result.parse_result.symbols
    assert app_result.canonical_chunks
    assert len(result.chunks) > 25
