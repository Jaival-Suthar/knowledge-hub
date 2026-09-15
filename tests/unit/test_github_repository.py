from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from knowledge_hub.ingestion.code import (
    CodeDiscovery,
    CodeFile,
    CodeSource,
    DiscoveryResult,
    Language,
)
from knowledge_hub.ingestion.github import (
    GitHubRepository,
    GitHubRepositoryError,
    GitHubRepositorySource,
    GitHubSnapshot,
    discover_github_repository,
)
from knowledge_hub.ingestion.github import repository as github_repository

REPOSITORY_URL = "https://github.com/example/project"
COMMIT_SHA = "a" * 40


def make_snapshot(root: Path) -> GitHubSnapshot:
    repository = GitHubRepository(
        owner="example",
        name="project",
        url=REPOSITORY_URL,
        ref="main",
        commit_sha=COMMIT_SHA,
    )
    return GitHubSnapshot(
        repository=repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
    )


def test_snapshot_root_is_passed_to_existing_code_discovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    snapshot = make_snapshot(tmp_path)
    source = GitHubRepositorySource(url=REPOSITORY_URL)
    discovered = CodeDiscovery().discover(CodeSource.directory(tmp_path))
    received: list[CodeSource] = []

    @dataclass
    class RecordingDiscovery:
        def discover(self, code_source: CodeSource) -> DiscoveryResult:
            received.append(code_source)
            return discovered

    monkeypatch.setattr(
        github_repository, "acquire_github_repository", lambda _: snapshot
    )

    result = discover_github_repository(source, discovery=RecordingDiscovery())

    assert received == [CodeSource.directory(tmp_path)]
    assert result.discovery is discovered
    assert result.snapshot is snapshot


def test_github_snapshot_uses_real_code_discovery_and_preserves_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('app')\n", encoding="utf-8")
    (tmp_path / "src" / "service.py").write_text(
        "def run():\n    return True\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "def test_app():\n    assert True\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("readme\n", encoding="utf-8")
    snapshot = make_snapshot(tmp_path)
    monkeypatch.setattr(
        github_repository, "acquire_github_repository", lambda _: snapshot
    )

    result = discover_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))

    assert all(isinstance(code_file, CodeFile) for code_file in result.files)
    assert {code_file.relative_path for code_file in result.files} == {
        "src/app.py",
        "src/service.py",
        "tests/test_app.py",
    }
    assert all(
        not code_file.relative_path.startswith(str(tmp_path))
        for code_file in result.files
    )
    assert result.files[0].language is Language.PYTHON


def test_existing_discovery_exclusions_apply_to_github_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("pass\n", encoding="utf-8")
    for directory, filename in (
        ("node_modules", "fake.js"),
        ("dist", "generated.js"),
        (".git", "config"),
    ):
        excluded = tmp_path / directory
        excluded.mkdir()
        (excluded / filename).write_text("ignored\n", encoding="utf-8")
    snapshot = make_snapshot(tmp_path)
    monkeypatch.setattr(
        github_repository, "acquire_github_repository", lambda _: snapshot
    )

    result = discover_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))

    assert [code_file.relative_path for code_file in result.files] == ["src/app.py"]
    assert all(".git" not in code_file.relative_path for code_file in result.files)


def test_acquisition_failure_remains_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = GitHubRepositoryError("clone failed")

    def fail(_: GitHubRepositorySource) -> GitHubSnapshot:
        raise error

    monkeypatch.setattr(
        github_repository,
        "acquire_github_repository",
        fail,
    )

    with pytest.raises(GitHubRepositoryError, match="clone failed"):
        discover_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))


def test_discovery_result_keeps_snapshot_alive_until_cleanup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temporary_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary_directory.name)
    (root / "app.py").write_text("pass\n", encoding="utf-8")
    snapshot = GitHubSnapshot(
        repository=make_snapshot(root).repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
        _cleanup_callback=temporary_directory.cleanup,
    )
    monkeypatch.setattr(
        github_repository, "acquire_github_repository", lambda _: snapshot
    )

    result = discover_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))

    assert result.files[0].path == root / "app.py"
    assert root.exists()
    result.cleanup()
    assert not root.exists()


def test_discovery_result_context_manager_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temporary_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary_directory.name)
    (root / "app.py").write_text("pass\n", encoding="utf-8")
    snapshot = GitHubSnapshot(
        repository=make_snapshot(root).repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
        _cleanup_callback=temporary_directory.cleanup,
    )
    monkeypatch.setattr(
        github_repository,
        "acquire_github_repository",
        lambda _: snapshot,
    )

    with discover_github_repository(
        GitHubRepositorySource(url=REPOSITORY_URL)
    ) as result:
        assert result.files
        assert root.exists()

    assert not root.exists()


def test_discovery_result_cleanup_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temporary_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary_directory.name)
    (root / "app.py").write_text("pass\n", encoding="utf-8")
    snapshot = GitHubSnapshot(
        repository=make_snapshot(root).repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
        _cleanup_callback=temporary_directory.cleanup,
    )
    monkeypatch.setattr(
        github_repository,
        "acquire_github_repository",
        lambda _: snapshot,
    )

    result = discover_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))

    result.cleanup()
    result.cleanup()

    assert not root.exists()


def test_discovery_failure_cleans_up_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    temporary_directory = tempfile.TemporaryDirectory(dir=tmp_path)
    root = Path(temporary_directory.name)
    (root / "app.py").write_text("pass\n", encoding="utf-8")
    snapshot = GitHubSnapshot(
        repository=make_snapshot(root).repository,
        commit_sha=COMMIT_SHA,
        root_path=root,
        _cleanup_callback=temporary_directory.cleanup,
    )
    monkeypatch.setattr(
        github_repository,
        "acquire_github_repository",
        lambda _: snapshot,
    )

    class FailingDiscovery:
        def discover(self, code_source: CodeSource) -> DiscoveryResult:
            raise RuntimeError("discovery failed")

    with pytest.raises(RuntimeError, match="discovery failed"):
        discover_github_repository(
            GitHubRepositorySource(url=REPOSITORY_URL),
            discovery=FailingDiscovery(),
        )

    assert not root.exists()
