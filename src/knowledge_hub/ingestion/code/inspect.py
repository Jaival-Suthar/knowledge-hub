from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonicalize_code
from .chunking import CodeChunker, SemanticCodeChunk
from .contracts import CodeFile, CodeIngestionError, CodeParseResult, CodeSource
from .discovery import CodeDiscovery, DiscoveryResult
from .parsers import ParserRegistry
from .zip_ingestion import SecureZipIngestor

PREVIEW_LIMIT = 140
SAMPLE_FILE_LIMIT = 3
SAMPLE_SYMBOL_LIMIT = 8
SAMPLE_CHUNK_LIMIT = 8


@dataclass(frozen=True)
class FileInspection:
    code_file: CodeFile
    parse_result: CodeParseResult
    semantic_chunks: tuple[SemanticCodeChunk, ...]
    document: object
    canonical_chunks: tuple[object, ...]


def inspect_source(path: Path, relative_file: str | None = None) -> str:
    result, source_kind = _discover(path)
    files = list(result.files)
    if relative_file is not None:
        files = [file for file in files if file.relative_path == relative_file]
        if not files:
            raise ValueError(f"discovered file not found: {relative_file}")

    inspections = _inspect_files(files)
    return _render(path, source_kind, result, inspections)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect code discovery, parsing, semantic chunking, and canonicalization."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--file", dest="relative_file", help="inspect one discovered relative path"
    )
    args = parser.parse_args(argv)

    try:
        print(inspect_source(args.source, args.relative_file))
    except (CodeIngestionError, OSError, ValueError) as error:
        parser.error(str(error))
    return 0


def _discover(path: Path) -> tuple[DiscoveryResult, str]:
    if path.suffix.lower() == ".zip":
        return SecureZipIngestor().ingest(CodeSource.zip(path)), "zip"
    if path.is_dir():
        return CodeDiscovery().discover(CodeSource.directory(path)), "directory"
    raise ValueError(f"source must be an existing directory or .zip file: {path}")


def _inspect_files(files: list[CodeFile]) -> list[FileInspection]:
    registry = ParserRegistry.default()
    chunker = CodeChunker()
    inspections: list[FileInspection] = []
    for code_file in files:
        parse_result = registry.get(code_file.language).parse(code_file)
        semantic_chunks = chunker.chunk(code_file, parse_result)
        document, canonical_chunks = canonicalize_code(code_file, semantic_chunks)
        inspections.append(
            FileInspection(
                code_file,
                parse_result,
                semantic_chunks,
                document,
                canonical_chunks,
            )
        )
    return inspections


def _render(
    path: Path,
    source_kind: str,
    result: DiscoveryResult,
    inspections: list[FileInspection],
) -> str:
    parsed_symbols = sum(len(item.parse_result.symbols) for item in inspections)
    parse_errors = sum(len(item.parse_result.errors) for item in inspections)
    semantic_chunks = [chunk for item in inspections for chunk in item.semantic_chunks]
    canonical_chunks = [
        chunk for item in inspections for chunk in item.canonical_chunks
    ]
    documents = [item.document for item in inspections]
    lines = [
        "SOURCE",
        f"  input: {path}",
        f"  kind: {source_kind}",
        "",
        "DISCOVERY",
        f"  supported files: {result.statistics.supported_count}",
        f"  languages: {_language_counts(result)}",
        f"  unsupported: {result.statistics.unsupported_count}",
        f"  excluded: {result.statistics.excluded_count}",
        f"  skipped: {result.statistics.skipped_count}",
        "",
        "PARSING",
        f"  parsed files: {len(inspections)}",
        f"  symbols: {parsed_symbols}",
        f"  parser/recoverable errors: {parse_errors}",
        "",
        "CHUNKING",
        f"  semantic chunks: {len(semantic_chunks)}",
        f"  split chunks (part_index > 0): {sum(chunk.part_index > 0 for chunk in semantic_chunks)}",
        "",
        "CANONICALIZATION",
        f"  documents: {len(documents)}",
        f"  canonical chunks: {len(canonical_chunks)}",
        f"  unique document IDs: {len({doc.document_id for doc in documents})}",
        f"  unique chunk IDs: {len({chunk.chunk_id for chunk in canonical_chunks})}",
        "",
        "REPRESENTATIVE FILES",
    ]
    lines.extend(
        f"  {item.code_file.relative_path} [{item.code_file.language.value}]"
        for item in inspections[:SAMPLE_FILE_LIMIT]
    )
    lines.extend(["", "REPRESENTATIVE SYMBOLS"])
    symbols = [symbol for item in inspections for symbol in item.parse_result.symbols]
    lines.extend(
        f"  {symbol.qualified_name} [{_symbol_type(symbol.symbol_type)}] "
        f"{symbol.path}:{symbol.start_line}-{symbol.end_line}"
        for symbol in symbols[:SAMPLE_SYMBOL_LIMIT]
    )
    lines.extend(["", "REPRESENTATIVE CHUNKS"])
    for chunk in canonical_chunks[:SAMPLE_CHUNK_LIMIT]:
        lines.extend(
            [
                f"  chunk_id: {chunk.chunk_id}",
                f"  qualified_name: {chunk.metadata.get('qualified_name', '')}",
                f"  structural_type: {chunk.structural_type}",
                f"  language: {chunk.language}",
                f"  source_uri: {chunk.source_uri}",
                f"  location: {chunk.location}",
                f"  document_id: {chunk.document_id}",
                f"  content: {_preview(chunk.content)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip()


def _language_counts(result: DiscoveryResult) -> str:
    return (
        ", ".join(
            f"{language.value}={count}"
            for language, count in sorted(
                result.statistics.language_counts.items(),
                key=lambda item: item[0].value,
            )
        )
        or "none"
    )


def _symbol_type(symbol_type: object) -> str:
    return str(getattr(symbol_type, "value", symbol_type))


def _preview(content: str) -> str:
    compact = " ".join(content.split())
    if len(compact) <= PREVIEW_LIMIT:
        return compact
    return f"{compact[: PREVIEW_LIMIT - 3]}..."


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
