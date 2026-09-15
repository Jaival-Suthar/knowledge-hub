"""Contracts for GitHub repository ingestion."""

from .acquisition import DEFAULT_GIT_TIMEOUT, acquire_github_repository
from .contracts import (
    GitHubRepository,
    GitHubRepositoryError,
    GitHubRepositorySource,
    GitHubSnapshot,
)
from .provenance import github_provenance
from .repository import GitHubDiscoveryResult, discover_github_repository
from .url import parse_github_repository_url

__all__ = [
    "DEFAULT_GIT_TIMEOUT",
    "GitHubDiscoveryResult",
    "GitHubRepository",
    "GitHubRepositoryError",
    "GitHubRepositorySource",
    "GitHubSnapshot",
    "acquire_github_repository",
    "discover_github_repository",
    "github_provenance",
    "parse_github_repository_url",
]
