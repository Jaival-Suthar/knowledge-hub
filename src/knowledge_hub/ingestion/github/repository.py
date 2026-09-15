from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from knowledge_hub.ingestion.code import (
    CodeDiscovery,
    CodeFile,
    CodeSource,
    DiscoveryResult,
)

from .acquisition import acquire_github_repository
from .contracts import GitHubRepository, GitHubRepositorySource, GitHubSnapshot


@dataclass(frozen=True)
class GitHubDiscoveryResult:
    """Discovered code together with the snapshot that owns its filesystem."""

    snapshot: GitHubSnapshot
    discovery: DiscoveryResult

    @property
    def files(self) -> tuple[CodeFile, ...]:
        return self.discovery.files

    @property
    def repository(self) -> GitHubRepository:
        return self.snapshot.repository

    def cleanup(self) -> None:
        """Release the temporary repository snapshot."""
        self.snapshot.cleanup()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.cleanup()


def discover_github_repository(
    source: GitHubRepositorySource,
    *,
    discovery: CodeDiscovery | None = None,
) -> GitHubDiscoveryResult:
    """Acquire a GitHub repository and discover its supported code files.

    The returned result owns the acquired snapshot and keeps its temporary
    filesystem alive until ``cleanup()`` or context-manager exit.
    """
    snapshot = acquire_github_repository(source)
    try:
        discovery_engine = discovery if discovery is not None else CodeDiscovery()
        discovery_result = discovery_engine.discover(
            CodeSource.directory(snapshot.root_path)
        )
        return GitHubDiscoveryResult(
            snapshot=snapshot,
            discovery=discovery_result,
        )
    except Exception:
        snapshot.cleanup()
        raise
