from __future__ import annotations

import re
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from .contracts import (
    GitHubRepositoryError,
    GitHubRepositorySource,
    GitHubSnapshot,
)
from .url import parse_github_repository_url

DEFAULT_GIT_TIMEOUT = 120.0
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")


def acquire_github_repository(
    source: GitHubRepositorySource,
    *,
    timeout: float = DEFAULT_GIT_TIMEOUT,
) -> GitHubSnapshot:
    """Acquire a validated GitHub repository into an owned temp directory."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    repository = parse_github_repository_url(source.url)
    _validate_ref(source.ref)
    temporary_directory = tempfile.TemporaryDirectory(prefix="knowledge-hub-github-")
    root_path = Path(temporary_directory.name) / repository.name

    try:
        clone_command = ["git", "clone", "--depth", "1"]
        if source.ref is not None:
            clone_command.extend(["--branch", source.ref])
        clone_command.extend(["--", repository.url, str(root_path)])
        _run_git(clone_command, timeout=timeout)

        _verify_repository(root_path, timeout=timeout)
        resolved_ref = source.ref or _resolve_default_ref(root_path, timeout=timeout)
        commit_sha = _resolve_commit_sha(root_path, timeout=timeout)
        resolved_repository = replace(
            repository,
            ref=resolved_ref,
            commit_sha=commit_sha,
        )
        cleanup_callback = _cleanup_callback(temporary_directory)
        return GitHubSnapshot(
            repository=resolved_repository,
            commit_sha=commit_sha,
            root_path=root_path,
            _cleanup_callback=cleanup_callback,
        )
    except Exception:
        temporary_directory.cleanup()
        raise


def _run_git(
    arguments: Sequence[str], *, timeout: float
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(arguments),
            capture_output=True,
            check=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as error:
        raise GitHubRepositoryError(
            "Git is unavailable; install Git before acquiring repositories"
        ) from error
    except subprocess.TimeoutExpired as error:
        raise GitHubRepositoryError("Git operation timed out") from error
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or error.stdout or "").strip()
        if len(detail) > 300:
            detail = f"{detail[:297]}..."
        message = "Git operation failed"
        if detail:
            message = f"{message}: {detail}"
        raise GitHubRepositoryError(message) from error


def _verify_repository(root_path: Path, *, timeout: float) -> None:
    if not root_path.is_dir() or not (root_path / ".git").exists():
        raise GitHubRepositoryError("Git clone did not produce a valid repository")
    _run_git(
        ["git", "-C", str(root_path), "rev-parse", "--git-dir"],
        timeout=timeout,
    )


def _resolve_default_ref(root_path: Path, *, timeout: float) -> str:
    result = _run_git(
        ["git", "-C", str(root_path), "symbolic-ref", "--short", "HEAD"],
        timeout=timeout,
    )
    ref = result.stdout.strip()
    if not ref:
        raise GitHubRepositoryError("Git did not report the repository default ref")
    return ref


def _resolve_commit_sha(root_path: Path, *, timeout: float) -> str:
    result = _run_git(
        ["git", "-C", str(root_path), "rev-parse", "HEAD"],
        timeout=timeout,
    )
    commit_sha = result.stdout.strip()
    if not _COMMIT_SHA_PATTERN.fullmatch(commit_sha):
        raise GitHubRepositoryError("Git returned an invalid commit SHA")
    return commit_sha.lower()


def _validate_ref(ref: str | None) -> None:
    if ref is not None and ref.startswith("-"):
        raise GitHubRepositoryError("Git ref cannot start with '-'")
    if ref is not None and "\x00" in ref:
        raise GitHubRepositoryError("Git ref contains an invalid null character")


def _cleanup_callback(
    temporary_directory: tempfile.TemporaryDirectory[str],
) -> Callable[[], None]:
    cleaned = False

    def cleanup() -> None:
        nonlocal cleaned
        if not cleaned:
            temporary_directory.cleanup()
            cleaned = True

    return cleanup
