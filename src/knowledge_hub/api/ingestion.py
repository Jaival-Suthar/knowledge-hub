"""API adapters that orchestrate the repository's existing ingestion flows."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from knowledge_hub.chunking.markdown import MarkdownChunker
from knowledge_hub.chunking.pdf import PdfChunker
from knowledge_hub.ingestion.adapters.markdown import MarkdownAdapter
from knowledge_hub.ingestion.adapters.pdf import PdfAdapter
from knowledge_hub.ingestion.articles import (
    ArticleChunker,
    ArticleParser,
    canonicalize_article,
)
from knowledge_hub.ingestion.articles.hashnode import HashnodeAcquirer
from knowledge_hub.ingestion.articles.hashnode.contracts import (
    HashnodeNotFoundError,
    HashnodeRequestError,
    HashnodeResponseError,
    HashnodeValidationError,
)
from knowledge_hub.ingestion.code import (
    CodeChunker,
    CodeIngestionError,
    CodeSource,
    ParserRegistry,
    SecureZipIngestor,
    ZipIngestionError,
    canonicalize_code,
)
from knowledge_hub.ingestion.code.discovery import DiscoveryResult
from knowledge_hub.ingestion.github import (
    GitHubRepositoryError,
    ingest_github_repository,
)
from knowledge_hub.models import Chunk, Document, Provenance, SourceType


class IngestionServiceError(RuntimeError):
    """Controlled failure raised by the API ingestion adapter."""


class IngestionResult:
    """Canonical results produced by one existing ingestion pipeline."""

    def __init__(
        self,
        source_type: SourceType,
        documents: Iterable[Document],
        chunks: Iterable[Chunk],
    ) -> None:
        self.source_type = source_type
        self.documents = tuple(documents)
        self.chunks = tuple(chunks)


class KnowledgeIngestionService:
    """Dispatch API inputs to existing source-specific ingestion components."""

    def ingest(
        self,
        source: str | Path,
        source_type: SourceType | None = None,
    ) -> IngestionResult:
        try:
            source_value = str(source)
            if _is_hashnode_url(source_value):
                return self._ingest_hashnode(source_value)
            if _is_github_url(source_value):
                return self._ingest_github(source_value)

            path = Path(source_value)
            detected_type = source_type or _file_source_type(path)
            if detected_type is SourceType.MARKDOWN:
                document = MarkdownAdapter().extract(path)
                return IngestionResult(
                    SourceType.MARKDOWN,
                    [document],
                    MarkdownChunker().chunk(document),
                )
            if detected_type is SourceType.PDF:
                document = PdfAdapter().extract(path)
                return IngestionResult(
                    SourceType.PDF,
                    [document],
                    PdfChunker().chunk(document),
                )
            if detected_type is SourceType.CODE and path.suffix.lower() == ".zip":
                return self._ingest_zip(path)
            raise IngestionServiceError(
                "source must be a Hashnode URL, GitHub URL, Markdown file, PDF, or ZIP"
            )
        except (
            CodeIngestionError,
            GitHubRepositoryError,
            HashnodeNotFoundError,
            HashnodeRequestError,
            HashnodeResponseError,
            HashnodeValidationError,
            ZipIngestionError,
        ) as exc:
            raise IngestionServiceError(str(exc)) from exc

    def _ingest_hashnode(self, source: str) -> IngestionResult:
        payload = HashnodeAcquirer().acquire(source)
        article = ArticleParser().parse(payload)
        document, chunks = canonicalize_article(
            article, ArticleChunker().chunk(article)
        )
        return IngestionResult(SourceType.ARTICLE, [document], chunks)

    def _ingest_github(self, source: str) -> IngestionResult:
        result = ingest_github_repository(source)
        return IngestionResult(
            SourceType.GITHUB,
            result.canonical_documents,
            result.canonical_chunks,
        )

    def _ingest_zip(self, source: Path) -> IngestionResult:
        discovered = SecureZipIngestor().ingest(CodeSource.zip(source))
        provenance = Provenance(source_uri=source.resolve().as_uri())
        documents, chunks = _canonicalize_code_discovery(discovered, provenance)
        return IngestionResult(SourceType.CODE, documents, chunks)


def _canonicalize_code_discovery(
    discovered: DiscoveryResult,
    provenance: Provenance,
) -> tuple[tuple[Document, ...], tuple[Chunk, ...]]:
    registry = ParserRegistry.default()
    chunker = CodeChunker()
    documents: list[Document] = []
    chunks: list[Chunk] = []
    for code_file in discovered.files:
        parsed = registry.get(code_file.language).parse(code_file)
        semantic_chunks = chunker.chunk(code_file, parsed)
        document, canonical_chunks = canonicalize_code(
            code_file,
            semantic_chunks,
            source_type=SourceType.CODE,
            provenance=provenance,
        )
        documents.append(document)
        chunks.extend(canonical_chunks)
    return tuple(documents), tuple(chunks)


def _file_source_type(path: Path) -> SourceType | None:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return SourceType.MARKDOWN
    if suffix == ".pdf":
        return SourceType.PDF
    if suffix == ".zip":
        return SourceType.CODE
    return None


def _is_hashnode_url(source: str) -> bool:
    return source.lower().startswith("https://") and ".hashnode.dev/" in source.lower()


def _is_github_url(source: str) -> bool:
    return source.lower().startswith(("https://github.com/", "http://github.com/"))
