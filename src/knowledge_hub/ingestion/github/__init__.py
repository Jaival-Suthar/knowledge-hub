"""Contracts for GitHub repository ingestion."""

from .contracts import (
    GitHubRepository,
    GitHubRepositoryError,
    GitHubRepositorySource,
    GitHubSnapshot,
)
from .url import parse_github_repository_url

__all__ = [
    "GitHubRepository",
    "GitHubRepositoryError",
    "GitHubRepositorySource",
    "GitHubSnapshot",
    "parse_github_repository_url",
]
