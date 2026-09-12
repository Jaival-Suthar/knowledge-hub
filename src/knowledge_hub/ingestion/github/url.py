from __future__ import annotations

import re
from urllib.parse import urlsplit

from .contracts import GitHubRepository, GitHubRepositoryError

_GITHUB_HOST = "github.com"
_REPOSITORY_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def parse_github_repository_url(url: str) -> GitHubRepository:
    """Validate and normalize an HTTPS GitHub repository URL.

    This function only describes repository identity. It does not resolve refs,
    access the network, or inspect the repository.
    """

    if not isinstance(url, str) or not url.strip():
        raise GitHubRepositoryError("GitHub repository URL must be a non-empty string")

    try:
        parsed = urlsplit(url)
        host = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise GitHubRepositoryError("invalid GitHub repository URL") from error

    if (
        parsed.scheme.lower() != "https"
        or host is None
        or host.lower() != _GITHUB_HOST
        or port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise GitHubRepositoryError("URL must be an HTTPS GitHub repository URL")

    path = parsed.path
    if (
        not path.startswith("/")
        or path.startswith("//")
        or path.endswith("//")
        or "//" in path.rstrip("/")
    ):
        raise GitHubRepositoryError("URL must identify exactly one GitHub repository")

    parts = path.strip("/").split("/")
    if len(parts) != 2:
        raise GitHubRepositoryError("URL must identify exactly one GitHub repository")

    owner, repository = parts
    repository = repository.removesuffix(".git")

    if not (
        repository
        and _REPOSITORY_PART.fullmatch(owner)
        and _REPOSITORY_PART.fullmatch(repository)
    ):
        raise GitHubRepositoryError("URL contains an invalid GitHub repository name")

    return GitHubRepository(
        owner=owner,
        name=repository,
        url=f"https://{_GITHUB_HOST}/{owner}/{repository}",
    )
