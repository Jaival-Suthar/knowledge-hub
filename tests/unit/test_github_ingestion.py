from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from knowledge_hub.ingestion.code import CodeDiscovery, CodeSource
from knowledge_hub.ingestion.github import (
    GitHubDiscoveryResult,
    GitHubIngestionResult,
    GitHubRepository,
    GitHubRepositoryError,
    GitHubSnapshot,
    ingest_github_repository,
)
from knowledge_hub.ingestion.github import ingestion as github_ingestion
from knowledge_hub.models import SourceType

REPOSITORY_URL = "https://github.com/example/project"
COMMIT_SHA = "a" * 40


def make_discovery_result(
    tmp_path: Path,
    *,
    content: str = "def run():\n    return True\n",
) -> tuple[GitHubDiscoveryResult, tempfile.TemporaryDirectory[str]]:
    temporary_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary_directory.name)
    (root / "app.py").write_text(content, encoding="utf-8")
    repository = GitHubRepository(
        owner="example",
        name="project",
        url=REPOSITORY_URL,
        ref="main",
        commit_sha=COMMIT_SHA,
    )
    snapshot = GitHubSnapshot(
        repository=repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
        _cleanup_callback=temporary_directory.cleanup,
    )
    discovery = CodeDiscovery().discover(CodeSource.directory(root))
    return GitHubDiscoveryResult(
        snapshot=snapshot, discovery=discovery
    ), temporary_directory


def test_ingestion_composes_existing_pipeline_and_returns_canonical_output(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovery_result, _temporary_directory = make_discovery_result(tmp_path)
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: discovery_result,
    )

    result = ingest_github_repository(REPOSITORY_URL, ref="main")

    assert isinstance(result, GitHubIngestionResult)
    assert result.repository.name == "project"
    assert result.ref == "main"
    assert result.commit_sha == COMMIT_SHA
    assert result.files
    assert result.parsed_files
    assert result.semantic_chunks
    assert result.documents
    assert result.chunks
    assert result.documents[0].source_type is SourceType.GITHUB
    assert result.chunks[0].source_type is SourceType.GITHUB
    assert result.chunks[0].provenance.repository == "example/project"
    assert not discovery_result.snapshot.root_path.exists()
    assert result.documents[0].content == result.files[0].content
    assert result.chunks[0].content == result.semantic_chunks[0].content


def test_acquisition_or_discovery_failure_propagates_before_parsing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = GitHubRepositoryError("acquisition failed")
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: (_ for _ in ()).throw(error),
    )

    with pytest.raises(GitHubRepositoryError, match="acquisition failed"):
        ingest_github_repository(REPOSITORY_URL)


def test_parser_failure_cleans_up_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovery_result, _temporary_directory = make_discovery_result(tmp_path)
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: discovery_result,
    )

    class FailingParser:
        def parse(self, code_file: object) -> object:
            raise RuntimeError("parser failed")

    class FailingRegistry:
        def get(self, language: object) -> FailingParser:
            return FailingParser()

    with pytest.raises(RuntimeError, match="parser failed"):
        ingest_github_repository(
            REPOSITORY_URL,
            parser_registry=FailingRegistry(),  # type: ignore[arg-type]
        )

    assert not discovery_result.snapshot.root_path.exists()


def test_chunker_failure_cleans_up_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovery_result, _temporary_directory = make_discovery_result(tmp_path)
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: discovery_result,
    )

    class FailingChunker:
        def chunk(self, code_file: object, parse_result: object) -> object:
            raise RuntimeError("chunking failed")

    with pytest.raises(RuntimeError, match="chunking failed"):
        ingest_github_repository(  # type: ignore[arg-type]
            REPOSITORY_URL,
            chunker=FailingChunker(),
        )

    assert not discovery_result.snapshot.root_path.exists()


def test_canonicalization_failure_cleans_up_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    discovery_result, _temporary_directory = make_discovery_result(tmp_path)
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: discovery_result,
    )

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("canonicalization failed")

    monkeypatch.setattr(github_ingestion, "canonicalize_code", fail)

    with pytest.raises(RuntimeError, match="canonicalization failed"):
        ingest_github_repository(REPOSITORY_URL)

    assert not discovery_result.snapshot.root_path.exists()


def test_empty_discovery_returns_empty_canonical_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temporary_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary_directory.name)
    repository = GitHubRepository(
        owner="example",
        name="project",
        url=REPOSITORY_URL,
        ref="main",
        commit_sha=COMMIT_SHA,
    )
    snapshot = GitHubSnapshot(
        repository=repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
        _cleanup_callback=temporary_directory.cleanup,
    )
    discovery = CodeDiscovery().discover(CodeSource.directory(root))
    discovery_result = GitHubDiscoveryResult(snapshot, discovery)
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: discovery_result,
    )

    result = ingest_github_repository(REPOSITORY_URL)

    assert result.files == ()
    assert result.documents == ()
    assert result.chunks == ()
    assert not root.exists()


def test_ingestion_ids_are_deterministic_for_same_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first, _first_directory = make_discovery_result(tmp_path)
    second, _second_directory = make_discovery_result(tmp_path)
    results = iter((first, second))
    monkeypatch.setattr(
        github_ingestion,
        "discover_github_repository",
        lambda source, discovery=None: next(results),
    )

    first_result = ingest_github_repository(REPOSITORY_URL)
    second_result = ingest_github_repository(REPOSITORY_URL)

    assert (
        first_result.documents[0].document_id == second_result.documents[0].document_id
    )
    assert first_result.chunks[0].chunk_id == second_result.chunks[0].chunk_id
    assert first_result.chunks[0].content_hash == second_result.chunks[0].content_hash
