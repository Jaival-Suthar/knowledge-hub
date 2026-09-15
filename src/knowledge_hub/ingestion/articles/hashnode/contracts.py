"""Hashnode-specific raw acquisition contracts."""

from __future__ import annotations

from dataclasses import dataclass


class HashnodeAcquisitionError(Exception):
    """Base error for Hashnode article acquisition failures."""


class HashnodeValidationError(HashnodeAcquisitionError):
    """The requested URL is not a supported Hashnode article URL."""


class HashnodeRequestError(HashnodeAcquisitionError):
    """The Hashnode API request could not be completed successfully."""


class HashnodeNotFoundError(HashnodeAcquisitionError):
    """The requested Hashnode article does not exist."""


class HashnodeResponseError(HashnodeAcquisitionError):
    """The Hashnode API returned an unusable response."""


@dataclass(frozen=True)
class HashnodeArticlePayload:
    """Raw article data returned by Hashnode acquisition.

    This is deliberately distinct from the source-neutral ``Article`` model.
    ``content`` is the provider's original article source and is not parsed or
    normalized by the acquisition layer.
    """

    source_uri: str
    content: str
    article_id: str | None = None
    title: str | None = None
    author: str | None = None
    published_at: str | None = None
    updated_at: str | None = None
    description: str | None = None
    tags: tuple[str, ...] = ()
    series: str | None = None
    slug: str | None = None
    publication_host: str | None = None

    @property
    def url(self) -> str:
        """Compatibility alias for consumers that call the source URL ``url``."""
        return self.source_uri
