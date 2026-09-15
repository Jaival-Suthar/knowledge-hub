"""Bounded structured acquisition of individual Hashnode articles."""

from __future__ import annotations

from urllib.parse import ParseResult, urlparse

import httpx

from .contracts import (
    HashnodeArticlePayload,
    HashnodeNotFoundError,
    HashnodeRequestError,
    HashnodeResponseError,
    HashnodeValidationError,
)

DEFAULT_HASHNODE_TIMEOUT = 30.0

_HASHNODE_HOST_SUFFIX = ".hashnode.dev"


class HashnodeAcquirer:
    """Acquire one raw Hashnode article through its Markdown endpoint."""

    def __init__(
        self,
        *,
        timeout: float = DEFAULT_HASHNODE_TIMEOUT,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.timeout = timeout

    def acquire(self, article_url: str) -> HashnodeArticlePayload:
        """Fetch and validate one Hashnode article without parsing its content."""
        parsed_url = _parse_article_url(article_url)
        host = parsed_url.hostname
        if host is None:
            raise HashnodeValidationError("Hashnode article URL has no hostname")
        slug = parsed_url.path.rstrip("/").rsplit("/", 1)[-1]
        markdown_url = _markdown_url(parsed_url)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.get(markdown_url)
        except httpx.TimeoutException as error:
            raise HashnodeRequestError("Hashnode request timed out") from error
        except httpx.RequestError as error:
            raise HashnodeRequestError("Hashnode request failed") from error

        if response.status_code == 404:
            raise HashnodeNotFoundError(
                f"Hashnode article was not found: {article_url}"
            )
        if not 200 <= response.status_code < 300:
            raise HashnodeRequestError(
                f"Hashnode Markdown endpoint returned HTTP {response.status_code}"
            )

        content_type = response.headers.get("content-type", "")
        if content_type.split(";", 1)[0].strip().lower() != "text/markdown":
            raise HashnodeResponseError(
                "Hashnode Markdown endpoint returned a non-Markdown response"
            )

        return HashnodeArticlePayload(
            source_uri=article_url,
            content=response.text,
            slug=slug,
            publication_host=host,
        )


def _parse_article_url(article_url: str) -> ParseResult:
    if not isinstance(article_url, str) or not article_url.strip():
        raise HashnodeValidationError("article URL must be a non-empty string")

    parsed = urlparse(article_url)
    hostname = parsed.hostname
    if (
        parsed.scheme.lower() != "https"
        or hostname is None
        or not _is_hashnode_host(hostname)
        or not parsed.path.strip("/")
        or parsed.fragment
    ):
        raise HashnodeValidationError(
            "article URL must be an HTTPS Hashnode publication article URL"
        )
    return parsed


def _is_hashnode_host(hostname: str) -> bool:
    normalized = hostname.lower().rstrip(".")
    return (
        normalized.endswith(_HASHNODE_HOST_SUFFIX)
        and normalized != (_HASHNODE_HOST_SUFFIX[1:])
    )


def _markdown_url(parsed_url: ParseResult) -> str:
    """Build the public Markdown endpoint from a validated article URL."""
    path = f"{parsed_url.path.rstrip('/')}.md"
    return parsed_url._replace(path=path, query="", fragment="").geturl()
