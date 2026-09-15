from __future__ import annotations

from knowledge_hub.models import Provenance

from .contracts import GitHubSnapshot


def github_provenance(snapshot: GitHubSnapshot) -> Provenance:
    """Convert resolved GitHub snapshot identity into generic provenance."""
    repository = snapshot.repository
    extra = {
        "owner": repository.owner,
        "repository_url": repository.url,
    }
    if repository.ref is not None:
        extra["ref"] = repository.ref

    return Provenance(
        source_uri=repository.url,
        repository=f"{repository.owner}/{repository.name}",
        branch=repository.ref,
        commit_sha=snapshot.commit_sha,
        extra=extra,
    )
