from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self


class GitHubRepositoryError(Exception):
    """Controlled failure for invalid GitHub repository input or state."""


def _require_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise GitHubRepositoryError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class GitHubRepositorySource:
    """User-provided GitHub repository input before acquisition."""

    url: str
    ref: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.url, "url")
        if self.ref is not None:
            _require_text(self.ref, "ref")


@dataclass(frozen=True)
class GitHubRepository:
    """Normalized repository identity, optionally resolved to a commit."""

    owner: str
    name: str
    url: str
    ref: str | None = None
    commit_sha: str | None = None

    def __post_init__(self) -> None:
        _require_text(self.owner, "owner")
        _require_text(self.name, "name")
        _require_text(self.url, "url")
        if self.ref is not None:
            _require_text(self.ref, "ref")
        if self.commit_sha is not None:
            _require_text(self.commit_sha, "commit_sha")


@dataclass(frozen=True)
class GitHubSnapshot:
    """A local snapshot of one resolved GitHub repository state."""

    repository: GitHubRepository
    commit_sha: str
    root_path: Path
    _cleanup_callback: Callable[[], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_text(self.commit_sha, "commit_sha")

    def cleanup(self) -> None:
        """Release the temporary storage owned by this snapshot."""
        if self._cleanup_callback is not None:
            self._cleanup_callback()

    def close(self) -> None:
        """Alias for :meth:`cleanup` for context-manager-style callers."""
        self.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()
