from __future__ import annotations

from dataclasses import dataclass

from knowledge_hub.ingestion.code import (
    CodeChunker,
    CodeDiscovery,
    CodeFile,
    CodeParseError,
    CodeParseResult,
    CodeSymbol,
    DiscoveryResult,
    ParserRegistry,
    SemanticCodeChunk,
    canonicalize_code,
)
from knowledge_hub.models import Chunk, Document, SourceType

from .contracts import GitHubRepository, GitHubRepositorySource
from .provenance import github_provenance
from .repository import discover_github_repository


@dataclass(frozen=True)
class GitHubFileIngestionResult:
    """The parser-to-canonical result for one discovered code file."""

    code_file: CodeFile
    parse_result: CodeParseResult
    semantic_chunks: tuple[SemanticCodeChunk, ...]
    document: Document
    canonical_chunks: tuple[Chunk, ...]


@dataclass(frozen=True)
class GitHubIngestionResult:
    """Materialized canonical knowledge produced from one GitHub snapshot."""

    repository: GitHubRepository
    discovery: DiscoveryResult
    file_results: tuple[GitHubFileIngestionResult, ...]

    @property
    def files(self) -> tuple[CodeFile, ...]:
        return self.discovery.files

    @property
    def parsed_files(self) -> tuple[CodeFile, ...]:
        return tuple(file_result.code_file for file_result in self.file_results)

    @property
    def semantic_chunks(self) -> tuple[SemanticCodeChunk, ...]:
        return tuple(
            semantic_chunk
            for file_result in self.file_results
            for semantic_chunk in file_result.semantic_chunks
        )

    @property
    def documents(self) -> tuple[Document, ...]:
        return tuple(file_result.document for file_result in self.file_results)

    @property
    def canonical_documents(self) -> tuple[Document, ...]:
        return self.documents

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(
            chunk
            for file_result in self.file_results
            for chunk in file_result.canonical_chunks
        )

    @property
    def canonical_chunks(self) -> tuple[Chunk, ...]:
        return self.chunks

    @property
    def symbols(self) -> tuple[CodeSymbol, ...]:
        return tuple(
            symbol
            for file_result in self.file_results
            for symbol in file_result.parse_result.symbols
        )

    @property
    def parse_errors(self) -> tuple[CodeParseError, ...]:
        return tuple(
            error
            for file_result in self.file_results
            for error in file_result.parse_result.errors
        )

    @property
    def ref(self) -> str | None:
        return self.repository.ref

    @property
    def commit_sha(self) -> str | None:
        return self.repository.commit_sha


def ingest_github_repository(
    url: str,
    ref: str | None = None,
    *,
    discovery: CodeDiscovery | None = None,
    parser_registry: ParserRegistry | None = None,
    chunker: CodeChunker | None = None,
) -> GitHubIngestionResult:
    """Ingest supported code from GitHub into canonical models."""
    source = GitHubRepositorySource(url=url, ref=ref)
    discovery_result = discover_github_repository(source, discovery=discovery)

    try:
        registry = (
            parser_registry if parser_registry is not None else ParserRegistry.default()
        )
        code_chunker = chunker if chunker is not None else CodeChunker()
        provenance = github_provenance(discovery_result.snapshot)
        file_results: list[GitHubFileIngestionResult] = []

        for code_file in discovery_result.files:
            parse_result = registry.get(code_file.language).parse(code_file)
            semantic_chunks = code_chunker.chunk(code_file, parse_result)
            document, canonical_chunks = canonicalize_code(
                code_file,
                semantic_chunks,
                source_type=SourceType.GITHUB,
                provenance=provenance,
            )
            file_results.append(
                GitHubFileIngestionResult(
                    code_file=code_file,
                    parse_result=parse_result,
                    semantic_chunks=semantic_chunks,
                    document=document,
                    canonical_chunks=canonical_chunks,
                )
            )

        return GitHubIngestionResult(
            repository=discovery_result.repository,
            discovery=discovery_result.discovery,
            file_results=tuple(file_results),
        )
    finally:
        discovery_result.cleanup()
