from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import pytest

from knowledge_hub.ingestion.github import (
    GitHubRepositoryError,
    GitHubRepositorySource,
    acquire_github_repository,
    acquisition,
)

COMMIT_SHA = "a" * 40
REPOSITORY_URL = "https://github.com/example/project"


def fake_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    default_ref: str = "main",
    commit_sha: str = COMMIT_SHA,
) -> list[tuple[list[str], dict[str, object]]]:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(
        arguments: list[str],
        **kwargs: object,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        if arguments[1:2] == ["clone"]:
            target = Path(arguments[-1])
            target.mkdir(parents=True)
            (target / ".git").mkdir()
            return subprocess.CompletedProcess(arguments, 0, "", "")
        if arguments[-2:] == ["rev-parse", "--git-dir"]:
            return subprocess.CompletedProcess(arguments, 0, ".git\n", "")
        if arguments[-3:] == ["symbolic-ref", "--short", "HEAD"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                f"{default_ref}\n",
                "",
            )
        if arguments[-2:] == ["rev-parse", "HEAD"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                f"{commit_sha}\n",
                "",
            )
        raise AssertionError(f"unexpected Git command: {arguments}")

    monkeypatch.setattr(acquisition.subprocess, "run", run)
    return calls


def force_temp_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[Path]:
    real_temporary_directory = tempfile.TemporaryDirectory
    created: list[Path] = []

    def factory(*args: object, **kwargs: object) -> tempfile.TemporaryDirectory[str]:
        kwargs["dir"] = str(tmp_path)
        temporary_directory = real_temporary_directory(*args, **kwargs)
        created.append(Path(temporary_directory.name))
        return temporary_directory

    monkeypatch.setattr(acquisition.tempfile, "TemporaryDirectory", factory)
    return created


def test_acquisition_resolves_explicit_ref_and_can_clean_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_git(monkeypatch)

    snapshot = acquire_github_repository(
        GitHubRepositorySource(url=REPOSITORY_URL, ref="release/v1"),
        timeout=7,
    )

    assert snapshot.root_path.is_dir()
    assert snapshot.repository.ref == "release/v1"
    assert snapshot.repository.commit_sha == COMMIT_SHA
    assert snapshot.commit_sha == COMMIT_SHA
    assert not snapshot.root_path.resolve().is_relative_to(Path.cwd().resolve())

    clone_call, clone_options = calls[0]
    assert clone_call[:5] == ["git", "clone", "--branch", "release/v1", "--"]
    assert clone_call[-2] == REPOSITORY_URL
    assert clone_options["timeout"] == 7
    assert all(options.get("shell") is not True for _, options in calls)
    assert any(command[-2:] == ["rev-parse", "HEAD"] for command, _ in calls)

    root_path = snapshot.root_path
    snapshot.cleanup()
    assert not root_path.exists()
    snapshot.close()


def test_acquisition_records_actual_default_ref_without_assuming_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = fake_git(monkeypatch, default_ref="trunk")

    snapshot = acquire_github_repository(
        GitHubRepositorySource(url=REPOSITORY_URL),
    )

    assert snapshot.repository.ref == "trunk"
    assert snapshot.repository.commit_sha == snapshot.commit_sha
    assert calls[0][0][:3] == ["git", "clone", "--"]
    assert any(
        command[-3:] == ["symbolic-ref", "--short", "HEAD"] for command, _ in calls
    )
    snapshot.cleanup()


@pytest.mark.parametrize(
    "failure",
    [
        subprocess.TimeoutExpired(["git", "clone"], 1),
        FileNotFoundError("git"),
        subprocess.CalledProcessError(128, ["git", "clone"], stderr="not found"),
    ],
)
def test_acquisition_translates_git_failures_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: BaseException,
) -> None:
    created = force_temp_directory(monkeypatch, tmp_path)

    def fail(*_: object, **__: object) -> None:
        raise failure

    monkeypatch.setattr(acquisition.subprocess, "run", fail)

    with pytest.raises(GitHubRepositoryError):
        acquire_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))

    assert created
    assert not created[0].exists()


def test_invalid_sha_is_rejected_and_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    force_temp_directory(monkeypatch, tmp_path)
    fake_git(monkeypatch, commit_sha="not-a-sha")

    with pytest.raises(GitHubRepositoryError, match="invalid commit SHA"):
        acquire_github_repository(GitHubRepositorySource(url=REPOSITORY_URL))

    assert not list(tmp_path.iterdir())


def test_nonexistent_ref_failure_is_cleaned_up(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created = force_temp_directory(monkeypatch, tmp_path)

    def fail_clone(*_: object, **__: object) -> None:
        raise subprocess.CalledProcessError(
            128,
            ["git", "clone"],
            stderr="Remote branch does-not-exist not found",
        )

    monkeypatch.setattr(acquisition.subprocess, "run", fail_clone)

    with pytest.raises(GitHubRepositoryError):
        acquire_github_repository(
            GitHubRepositorySource(
                url=REPOSITORY_URL,
                ref="does-not-exist",
            )
        )

    assert created
    assert not created[0].exists()


def test_invalid_repository_and_ref_are_rejected_before_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def run(*args: object, **kwargs: object) -> None:
        calls.append((args, kwargs))

    monkeypatch.setattr(acquisition.subprocess, "run", run)

    with pytest.raises(GitHubRepositoryError):
        acquire_github_repository(
            GitHubRepositorySource(url="https://gitlab.com/example/project")
        )
    with pytest.raises(GitHubRepositoryError):
        acquire_github_repository(
            GitHubRepositorySource(url=REPOSITORY_URL, ref="--bad-ref")
        )

    assert not calls


def test_snapshot_cleanup_is_explicit_and_idempotent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_git(monkeypatch)

    with acquire_github_repository(
        GitHubRepositorySource(url=REPOSITORY_URL)
    ) as snapshot:
        root_path = snapshot.root_path
        assert root_path.exists()

    assert not root_path.exists()
