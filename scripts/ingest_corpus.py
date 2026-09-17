from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from knowledge_hub.ingestion.adapters.markdown import MarkdownAdapter
from knowledge_hub.ingestion.adapters.pdf import PdfAdapter
from knowledge_hub.ingestion.articles import (
    ArticleChunker,
    ArticleParser,
    canonicalize_article,
)
from knowledge_hub.ingestion.articles.hashnode import HashnodeAcquirer
from knowledge_hub.ingestion.code.canonical import canonicalize_code
from knowledge_hub.ingestion.code.chunking import CodeChunker
from knowledge_hub.ingestion.code.contracts import CodeSource
from knowledge_hub.ingestion.code.discovery import CodeDiscovery
from knowledge_hub.ingestion.code.parsers.registry import ParserRegistry
from knowledge_hub.ingestion.code.zip_ingestion import SecureZipIngestor
from knowledge_hub.ingestion.github import ingest_github_repository

RAW_DIR = Path("data/raw")

GITHUB_URL = "https://github.com/Jaival-Suthar/RippleTalk"

ARTICLE_URLS = [
    "https://jaivalsuthar.hashnode.dev/agentguard-the-trust-boundary-for-agentic-execution",
    "https://jaivalsuthar.hashnode.dev/prefetching-at-scale-why-instagram-works-without-internet-through-predictive-caching",
    "https://jaivalsuthar.hashnode.dev/what-google-docs-taught-us-about-building-the-impossible",
]

TEST_FIXTURE_ZIPS = {
    "01_nested_mixed_codebase.zip",
    "02_oversized_code.zip",
    "03_malformed_sources.zip",
    "04_duplicates.zip",
    "05_security_zip.zip",
}

CODE_EXTENSIONS = {
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".py",
    ".c",
    ".h",
    ".cpp",
    ".cc",
    ".cxx",
    ".hpp",
}


@dataclass(frozen=True)
class CorpusIngestionResult:
    """Canonical material and source-specific validation details."""

    documents: tuple[object, ...]
    chunks: tuple[object, ...]
    github: object
    articles: tuple[tuple[str, int], ...]


def canonicalize_discovered_files(
    discovered: object,
    parser_registry: ParserRegistry,
    code_chunker: CodeChunker,
) -> tuple[list[object], list[object]]:
    """Parse, chunk, and canonicalize discovered code files."""
    documents: list[object] = []
    chunks: list[object] = []

    for code_file in discovered.files:
        parser = parser_registry.get(code_file.language)

        semantic_chunks = parser.parse(code_file)

        canonical_chunks = code_chunker.chunk(
            code_file,
            semantic_chunks,
        )

        document, chunks_for_file = canonicalize_code(
            code_file=code_file,
            semantic_chunks=canonical_chunks,
        )

        documents.append(document)
        chunks.extend(chunks_for_file)

    return documents, chunks


def ingest_code_source(
    source: CodeSource,
    discovery: CodeDiscovery,
    parser_registry: ParserRegistry,
    code_chunker: CodeChunker,
) -> tuple[list[object], list[object]]:
    """Discover, parse, chunk, and canonicalize a code source."""
    discovered = discovery.discover(source)

    return canonicalize_discovered_files(
        discovered,
        parser_registry,
        code_chunker,
    )


def ingest_local_corpus(
    raw_dir: Path,
) -> tuple[list[object], list[object]]:
    """Ingest supported local corpus sources."""
    pdf_adapter = PdfAdapter()
    markdown_adapter = MarkdownAdapter()

    discovery = CodeDiscovery()
    parser_registry = ParserRegistry.default()
    code_chunker = CodeChunker()
    zip_ingestor = SecureZipIngestor()

    documents: list[object] = []
    chunks: list[object] = []

    for path in sorted(raw_dir.iterdir()):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()

        # ---------------------------------------------------------------
        # PDF
        # ---------------------------------------------------------------
        if suffix == ".pdf":
            print(f"  PDF       {path.name}")

            document = pdf_adapter.extract(path)
            documents.append(document)
            continue

        # ---------------------------------------------------------------
        # Markdown
        # ---------------------------------------------------------------
        if suffix in {".md", ".markdown"}:
            print(f"  MARKDOWN  {path.name}")

            document = markdown_adapter.extract(path)
            documents.append(document)
            continue

        # ---------------------------------------------------------------
        # ZIP
        # ---------------------------------------------------------------
        if suffix == ".zip":
            if path.name in TEST_FIXTURE_ZIPS:
                print(f"  SKIP      {path.name} [test fixture]")
                continue

            print(f"  ZIP       {path.name}")

            discovered = zip_ingestor.ingest(CodeSource.zip(path))

            source_documents, source_chunks = canonicalize_discovered_files(
                discovered,
                parser_registry,
                code_chunker,
            )

            documents.extend(source_documents)
            chunks.extend(source_chunks)
            continue

        # ---------------------------------------------------------------
        # Direct source-code files
        # ---------------------------------------------------------------
        if suffix in CODE_EXTENSIONS:
            print(f"  CODE      {path.name}")

            source_documents, source_chunks = ingest_code_source(
                CodeSource.file(path),
                discovery,
                parser_registry,
                code_chunker,
            )

            documents.extend(source_documents)
            chunks.extend(source_chunks)
            continue

        # ---------------------------------------------------------------
        # Unsupported local files
        # ---------------------------------------------------------------
        print(f"  SKIP      {path.name} [unsupported local source]")

    return documents, chunks


def ingest_article(
    article_url: str,
    *,
    acquirer: HashnodeAcquirer | None = None,
) -> tuple[object, tuple[object, ...]]:
    """Acquire and canonicalize one Hashnode article through existing APIs."""
    payload = (acquirer or HashnodeAcquirer()).acquire(article_url)
    article = ArticleParser().parse(payload)
    article_chunks = ArticleChunker().chunk(article)
    return canonicalize_article(article, article_chunks)


def ingest_corpus(raw_dir: Path) -> CorpusIngestionResult:
    """Ingest local, GitHub, and configured Hashnode sources."""
    local_documents, local_chunks = ingest_local_corpus(raw_dir)
    github_result = ingest_github_repository(GITHUB_URL)

    documents = [*local_documents, *github_result.documents]
    chunks = [*local_chunks, *github_result.chunks]
    article_counts: list[tuple[str, int]] = []

    for article_url in ARTICLE_URLS:
        document, article_chunks = ingest_article(article_url)
        documents.append(document)
        chunks.extend(article_chunks)
        article_counts.append((article_url, len(article_chunks)))

    return CorpusIngestionResult(
        documents=tuple(documents),
        chunks=tuple(chunks),
        github=github_result,
        articles=tuple(article_counts),
    )


def print_summary(
    documents: list[object],
    chunks: list[object],
) -> None:
    """Print ingestion summary."""
    print()
    print("=" * 60)
    print("LOCAL CORPUS INGESTION SUMMARY")
    print("=" * 60)
    print(f"Documents: {len(documents)}")
    print(f"Chunks:    {len(chunks)}")

    if documents:
        source_counts: dict[str, int] = {}

        for document in documents:
            source_type = (
                document.source_type.value
                if hasattr(document.source_type, "value")
                else str(document.source_type)
            )
            source_counts[source_type] = source_counts.get(source_type, 0) + 1

        print()
        print("Documents by source type:")

        for source_type, count in sorted(source_counts.items()):
            print(f"  {source_type:<15} {count}")

    if chunks:
        chunk_source_counts: dict[str, int] = {}

        for chunk in chunks:
            source = chunk.source_type
            source_type = source.value if hasattr(source, "value") else str(source)
            chunk_source_counts[source_type] = (
                chunk_source_counts.get(source_type, 0) + 1
            )

        print()
        print("Chunks by source type:")

        for source_type, count in sorted(chunk_source_counts.items()):
            print(f"  {source_type:<15} {count}")


def print_external_summary(result: CorpusIngestionResult) -> None:
    """Print deterministic details from external ingestion results."""
    github = result.github
    print()
    print("GitHub:")
    print(f"  repository: {github.repository.url}")
    print(f"  ref:        {github.ref or '<default>'}")
    print(f"  commit SHA: {github.commit_sha}")
    print(f"  files:      {len(github.files)}")
    print(f"  chunks:     {len(github.chunks)}")

    print()
    print("Hashnode articles:")
    for article_url, chunk_count in result.articles:
        print(f"  {article_url}: {chunk_count} chunks")


def main() -> None:
    print("Knowledge Hub — Multi-source corpus ingestion")
    print("=" * 60)
    print(f"Raw corpus: {RAW_DIR}")
    print(f"GitHub:     {GITHUB_URL}")
    print(f"Articles:   {len(ARTICLE_URLS)}")
    print()

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Raw corpus directory does not exist: {RAW_DIR}")

    result = ingest_corpus(RAW_DIR)
    print_summary(list(result.documents), list(result.chunks))
    print_external_summary(result)


if __name__ == "__main__":
    main()
