from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from knowledge_hub.ingestion.code import (
    CodeFile,
    CodeSymbol,
    CodeSymbolType,
    Language,
    SemanticCodeChunk,
    canonicalize_code,
)
from knowledge_hub.ingestion.github import (
    GitHubRepository,
    GitHubSnapshot,
    github_provenance,
)
from knowledge_hub.models import Chunk, Document, SourceType

REPOSITORY_URL = "https://github.com/Jaival-Suthar/RippleTalk"
COMMIT_SHA = "a" * 40
RELATIVE_PATH = "src/services/message.service.ts"
SYMBOL_NAME = "sendMessage"
QUALIFIED_SYMBOL_NAME = "MessageService.sendMessage"
START_LINE = 42
END_LINE = 81


def make_code_artifact(
    content: str = "sendMessage(message: string) { return message; }\n",
) -> CodeFile:
    return CodeFile(
        path=Path(RELATIVE_PATH),
        relative_path=RELATIVE_PATH,
        language=Language.TYPESCRIPT,
        content=content,
        size=len(content),
        content_hash=sha256(content.encode("utf-8")).hexdigest(),
    )


def make_semantic_chunk(
    content: str = "sendMessage(message: string) { return message; }\n",
) -> SemanticCodeChunk:
    symbol = CodeSymbol(
        name=SYMBOL_NAME,
        qualified_name=QUALIFIED_SYMBOL_NAME,
        symbol_type=CodeSymbolType.METHOD,
        language=Language.TYPESCRIPT,
        path=RELATIVE_PATH,
        start_line=START_LINE,
        end_line=END_LINE,
        content=content,
        parent="MessageService",
    )

    chunk_id = sha256(
        f"{RELATIVE_PATH}\0{QUALIFIED_SYMBOL_NAME}\0{content}".encode()
    ).hexdigest()

    return SemanticCodeChunk(
        symbol=symbol,
        content=content,
        chunk_id=chunk_id,
        start_line=START_LINE,
        end_line=END_LINE,
        part_index=0,
    )


def make_snapshot() -> GitHubSnapshot:
    repository = GitHubRepository(
        owner="Jaival-Suthar",
        name="RippleTalk",
        url=REPOSITORY_URL,
        ref="main",
        commit_sha=COMMIT_SHA,
    )

    return GitHubSnapshot(
        repository=repository,
        commit_sha=COMMIT_SHA,
        root_path=Path("snapshot"),
    )


def canonicalize_github_artifact(
    content: str = "sendMessage(message: string) { return message; }\n",
) -> tuple[Document, tuple[Chunk, ...]]:
    code_file = make_code_artifact(content)
    semantic_chunk = make_semantic_chunk(content)

    return canonicalize_code(
        code_file,
        (semantic_chunk,),
        source_type=SourceType.GITHUB,
        provenance=github_provenance(make_snapshot()),
    )


def test_github_provenance_survives_canonical_code_mapping() -> None:
    document, chunks = canonicalize_github_artifact()

    assert len(chunks) == 1
    chunk = chunks[0]

    assert document.source_type is SourceType.GITHUB
    assert document.source_uri == REPOSITORY_URL
    assert document.metadata["relative_path"] == RELATIVE_PATH
    assert document.metadata["language"] == "typescript"
    assert document.metadata["content_hash"]
    assert document.metadata["repository"] == "Jaival-Suthar/RippleTalk"
    assert document.metadata["ref"] == "main"
    assert document.metadata["commit_sha"] == COMMIT_SHA
    assert document.metadata["repository_url"] == REPOSITORY_URL

    assert document.provenance.source_uri == REPOSITORY_URL
    assert document.provenance.repository == "Jaival-Suthar/RippleTalk"
    assert document.provenance.branch == "main"
    assert document.provenance.commit_sha == COMMIT_SHA
    assert document.provenance.extra["owner"] == "Jaival-Suthar"

    assert chunk.source_type is SourceType.GITHUB
    assert chunk.source_uri == REPOSITORY_URL
    assert chunk.language == "typescript"
    assert chunk.structural_type == "method"
    assert chunk.provenance.path == RELATIVE_PATH
    assert chunk.provenance.symbol == QUALIFIED_SYMBOL_NAME
    assert chunk.provenance.line_start == START_LINE
    assert chunk.provenance.line_end == END_LINE

    assert chunk.metadata["repository"] == "Jaival-Suthar/RippleTalk"
    assert chunk.metadata["ref"] == "main"
    assert chunk.metadata["commit_sha"] == COMMIT_SHA
    assert chunk.metadata["owner"] == "Jaival-Suthar"
    assert chunk.metadata["repository_url"] == REPOSITORY_URL

    assert chunk.content_hash == sha256(chunk.content.encode("utf-8")).hexdigest()


def test_github_canonicalization_is_deterministic() -> None:
    first_document, first_chunks = canonicalize_github_artifact()
    second_document, second_chunks = canonicalize_github_artifact()

    assert first_document.document_id == second_document.document_id
    assert first_chunks[0].chunk_id == second_chunks[0].chunk_id
    assert first_chunks[0].content_hash == second_chunks[0].content_hash


def test_changed_content_gets_a_different_canonical_identity() -> None:
    first_document, first_chunks = canonicalize_github_artifact()

    changed_document, changed_chunks = canonicalize_github_artifact(
        "sendMessage(message: string) { return message.trim(); }\n"
    )

    assert first_document.document_id != changed_document.document_id
    assert first_chunks[0].chunk_id != changed_chunks[0].chunk_id
    assert first_chunks[0].content_hash != changed_chunks[0].content_hash
