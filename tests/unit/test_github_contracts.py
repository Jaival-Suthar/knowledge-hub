from pathlib import Path

import pytest

from knowledge_hub.ingestion.github import (
    GitHubRepository,
    GitHubRepositoryError,
    GitHubRepositorySource,
    GitHubSnapshot,
    parse_github_repository_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/Jaival-Suthar/RippleTalk",
        "https://github.com/Jaival-Suthar/RippleTalk/",
        "https://github.com/Jaival-Suthar/RippleTalk.git",
        "https://github.com/Jaival-Suthar/RippleTalk.git/",
    ],
)
def test_repository_url_forms_normalize_to_one_identity(url: str) -> None:
    repository = parse_github_repository_url(url)

    assert repository.owner == "Jaival-Suthar"
    assert repository.name == "RippleTalk"
    assert repository.url == "https://github.com/Jaival-Suthar/RippleTalk"
    assert repository.ref is None
    assert repository.commit_sha is None


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com",
        "https://github.com/",
        "https://github.com/Jaival-Suthar",
        "https://github.com/Jaival-Suthar/",
        "https://github.com/Jaival-Suthar/RippleTalk/issues",
        "https://github.com/Jaival-Suthar/RippleTalk/pulls",
        "https://github.com/Jaival-Suthar/RippleTalk/pull/12",
        "https://github.com/Jaival-Suthar/RippleTalk/issues/42",
        "https://github.com/Jaival-Suthar/RippleTalk/blob/main/README.md",
        "https://github.com/Jaival-Suthar/RippleTalk/tree/main",
        "https://github.com/Jaival-Suthar/RippleTalk/actions",
        "https://gitlab.com/user/repo",
        "https://bitbucket.org/user/repo",
    ],
)
def test_non_repository_urls_are_rejected(url: str) -> None:
    with pytest.raises(GitHubRepositoryError):
        parse_github_repository_url(url)


def test_source_preserves_explicit_ref_without_resolving_it() -> None:
    source = GitHubRepositorySource(
        url="https://github.com/Jaival-Suthar/RippleTalk",
        ref="main",
    )

    assert source.url == "https://github.com/Jaival-Suthar/RippleTalk"
    assert source.ref == "main"


def test_source_without_ref_remains_unresolved() -> None:
    source = GitHubRepositorySource(url="https://github.com/Jaival-Suthar/RippleTalk")

    assert source.ref is None


@pytest.mark.parametrize(
    ("url", "ref"),
    [
        ("", None),
        ("   ", None),
        ("https://github.com/user/repo", ""),
    ],
)
def test_source_requires_meaningful_text_fields(
    url: str,
    ref: str | None,
) -> None:
    with pytest.raises(GitHubRepositoryError):
        GitHubRepositorySource(url=url, ref=ref)


def test_repository_and_snapshot_can_carry_resolved_commit_state(
    tmp_path: Path,
) -> None:
    repository = GitHubRepository(
        owner="Jaival-Suthar",
        name="RippleTalk",
        url="https://github.com/Jaival-Suthar/RippleTalk",
        ref="main",
        commit_sha="a" * 40,
    )
    snapshot = GitHubSnapshot(
        repository=repository,
        commit_sha=repository.commit_sha,
        root_path=tmp_path,
    )

    assert snapshot.repository is repository
    assert snapshot.commit_sha == "a" * 40
    assert snapshot.root_path == tmp_path
