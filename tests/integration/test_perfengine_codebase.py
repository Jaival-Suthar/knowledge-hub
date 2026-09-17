from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from knowledge_hub.ingestion.code import (
    CodeChunker,
    CodeSource,
    ParserRegistry,
    SecureZipIngestor,
    canonicalize_code,
)
from knowledge_hub.models import SourceType

PERFENGINE_ARCHIVE = Path("data/raw/perfengine.zip")


def ingest_perfengine():
    assert PERFENGINE_ARCHIVE.is_file(), PERFENGINE_ARCHIVE
    return SecureZipIngestor().ingest(CodeSource.zip(PERFENGINE_ARCHIVE))


def test_perfengine_archive_discovery_preserves_real_codebase_invariants() -> None:
    result = ingest_perfengine()

    assert result.files
    assert {file.language.value for file in result.files} == {
        "typescript",
        "javascript",
    }
    assert any(
        file.relative_path == "perfengine/bin/perfengine.js" for file in result.files
    )
    assert any(file.relative_path.endswith("/valid.ts") for file in result.files)
    assert all("\\" not in file.relative_path for file in result.files)
    assert all(
        file.content_hash == sha256(file.content.encode("utf-8")).hexdigest()
        for file in result.files
    )

    stats = result.statistics
    assert stats.supported_count == len(result.files)
    assert stats.total_files_encountered == (
        len(result.files) + len(result.unsupported) + len(result.skipped)
    )
    assert stats.excluded_count == len(result.excluded)
    assert stats.unsupported_count == len(result.unsupported)
    assert stats.skipped_count == len(result.skipped)
    assert sum(stats.language_counts.values()) == len(result.files)
    assert result.excluded
    assert result.unsupported


def test_perfengine_files_survive_parser_chunker_and_canonical_mapping() -> None:
    discovered = ingest_perfengine()
    registry = ParserRegistry.default()
    chunker = CodeChunker()

    parsed_files = 0
    parse_errors = 0
    files_with_symbols = 0
    canonical_documents = []
    canonical_chunks = []

    for code_file in discovered.files:
        parse_result = registry.get(code_file.language).parse(code_file)
        parse_errors += len(parse_result.errors)
        semantic_chunks = chunker.chunk(code_file, parse_result)
        document, chunks = canonicalize_code(code_file, semantic_chunks)

        parsed_files += 1
        files_with_symbols += bool(parse_result.symbols)
        canonical_documents.append(document)
        canonical_chunks.extend(chunks)

        assert document.source_type is SourceType.CODE
        assert document.content == code_file.content
        assert document.title == code_file.relative_path
        assert all(chunk.document_id == document.document_id for chunk in chunks)
        assert all(chunk.source_type is SourceType.CODE for chunk in chunks)
        assert all(chunk.language == code_file.language.value for chunk in chunks)
        assert all(chunk.content for chunk in chunks)

    assert parsed_files == len(discovered.files)
    assert files_with_symbols > 0
    assert parse_errors == 0
    assert canonical_documents
    assert canonical_chunks
    assert len({document.document_id for document in canonical_documents}) == len(
        canonical_documents
    )


def test_perfengine_code_discovery_can_be_repeated_deterministically() -> None:
    first = ingest_perfengine()
    second = ingest_perfengine()

    assert first.statistics == second.statistics
    assert [file.relative_path for file in first.files] == [
        file.relative_path for file in second.files
    ]
    assert [(file.content_hash, file.language) for file in first.files] == [
        (file.content_hash, file.language) for file in second.files
    ]


def test_real_qualified_symbols_survive_to_canonical_chunks() -> None:
    discovered = ingest_perfengine()
    registry = ParserRegistry.default()
    chunker = CodeChunker()
    observed: set[tuple[str, str]] = set()

    for code_file in discovered.files:
        parsed = registry.get(code_file.language).parse(code_file)
        _, chunks = canonicalize_code(code_file, chunker.chunk(code_file, parsed))
        observed.update(
            (chunk.source_uri, chunk.metadata["qualified_name"]) for chunk in chunks
        )

    assert (
        "perfengine/fixtures/node-service/index.js",
        "processItems",
    ) in observed
    assert (
        "perfengine/fixtures/rules/core/duplicate-logic-block/invalid.ts",
        "first",
    ) in observed
    assert (
        "perfengine/fixtures/rules/core/duplicate-logic-block/invalid.ts",
        "second",
    ) in observed
