from __future__ import annotations

import os
from pathlib import PurePosixPath

import pytest

from knowledge_hub.ingestion.code import CodeFile
from knowledge_hub.ingestion.github import (
    GitHubRepositorySource,
    discover_github_repository,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("KNOWLEDGE_HUB_RUN_NETWORK_TESTS") != "1",
    reason="set KNOWLEDGE_HUB_RUN_NETWORK_TESTS=1 to run network integration tests",
)


def test_real_github_repository_flows_through_code_discovery() -> None:
    result = discover_github_repository(
        GitHubRepositorySource(url="https://github.com/pallets/markupsafe")
    )

    try:
        assert result.repository.owner == "pallets"
        assert result.repository.name == "markupsafe"
        assert result.snapshot.root_path.is_dir()
        assert result.files
        assert all(isinstance(code_file, CodeFile) for code_file in result.files)
        assert all(code_file.relative_path for code_file in result.files)
        assert all(
            not str(code_file.relative_path).startswith(str(result.snapshot.root_path))
            for code_file in result.files
        )
        assert all(
            ".git" not in PurePosixPath(code_file.relative_path).parts
            for code_file in result.files
        )
    finally:
        result.cleanup()

    assert not result.snapshot.root_path.exists()
