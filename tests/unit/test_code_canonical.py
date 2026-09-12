from hashlib import sha256
from pathlib import Path

from knowledge_hub.ingestion.code import (
    CodeFile,
    CodeSymbol,
    CodeSymbolType,
    Language,
    SemanticCodeChunk,
    canonicalize_code,
    document_id_for,
    to_canonical_chunks,
    to_canonical_document,
)
from knowledge_hub.models import SourceType


def fixture_code_file(path: str = "src/auth/service.py") -> CodeFile:
    content = "class Auth:\n    def login(self):\n        return True\n"
    return CodeFile(
        path=Path(path),
        relative_path=path,
        language=Language.PYTHON,
        content=content,
        size=len(content),
        content_hash=sha256(content.encode()).hexdigest(),
    )


def fixture_chunks(code_file: CodeFile) -> tuple[SemanticCodeChunk, ...]:
    class_symbol = CodeSymbol(
        name="Auth",
        qualified_name="Auth",
        symbol_type=CodeSymbolType.CLASS,
        language=Language.PYTHON,
        path=code_file.relative_path,
        start_line=1,
        end_line=3,
        content=code_file.content,
    )
    method_symbol = CodeSymbol(
        name="login",
        qualified_name="Auth.login",
        symbol_type=CodeSymbolType.METHOD,
        language=Language.PYTHON,
        path=code_file.relative_path,
        start_line=2,
        end_line=3,
        content="    def login(self):\n        return True\n",
        parent="Auth",
    )
    return (
        SemanticCodeChunk(class_symbol, class_symbol.content, "class-id", 1, 3),
        SemanticCodeChunk(
            method_symbol,
            method_symbol.content,
            "method-id",
            2,
            3,
            part_index=1,
        ),
    )


def test_code_file_maps_to_one_document_and_chunks_share_its_id() -> None:
    code_file = fixture_code_file()
    document, chunks = canonicalize_code(code_file, fixture_chunks(code_file))
    assert document.source_type is SourceType.CODE
    assert document.title == code_file.relative_path
    assert document.content == code_file.content
    assert len(chunks) == 2
    assert {chunk.document_id for chunk in chunks} == {document.document_id}


def test_document_id_is_deterministic_and_path_sensitive() -> None:
    first = fixture_code_file()
    same = fixture_code_file()
    other = fixture_code_file("src/other/service.py")
    assert document_id_for(first) == document_id_for(same)
    assert document_id_for(first) != document_id_for(other)


def test_chunk_fields_and_metadata_are_preserved() -> None:
    code_file = fixture_code_file()
    _document, chunks = canonicalize_code(code_file, fixture_chunks(code_file))
    class_chunk, method_chunk = chunks
    assert class_chunk.chunk_id == "class-id"
    assert class_chunk.content == code_file.content
    assert class_chunk.language == "python"
    assert class_chunk.structural_type == "class"
    assert class_chunk.parent_structure is None
    assert class_chunk.location == "src/auth/service.py:lines:1-3"
    assert class_chunk.provenance.symbol == "Auth"
    assert class_chunk.provenance.line_start == 1
    assert class_chunk.provenance.line_end == 3
    assert method_chunk.parent_structure == "Auth"
    assert method_chunk.location == "src/auth/service.py:lines:2-3:part:1"
    assert method_chunk.metadata["qualified_name"] == "Auth.login"
    assert method_chunk.metadata["part_index"] == "1"
    assert (
        method_chunk.content_hash
        == sha256(method_chunk.content.encode("utf-8")).hexdigest()
    )


def test_conversion_is_repeatable_and_does_not_mutate_inputs() -> None:
    code_file = fixture_code_file()
    semantic_chunks = fixture_chunks(code_file)
    before_file = code_file
    before_chunks = semantic_chunks
    first = canonicalize_code(code_file, semantic_chunks)
    second = canonicalize_code(code_file, semantic_chunks)
    assert first == second
    assert code_file == before_file
    assert semantic_chunks == before_chunks


def test_empty_semantic_chunks_produce_no_canonical_chunks() -> None:
    code_file = fixture_code_file()
    document = to_canonical_document(code_file)
    assert to_canonical_chunks(code_file, document, []) == ()


def test_generator_semantic_chunks_are_consumed_once() -> None:
    code_file = fixture_code_file()
    document = to_canonical_document(code_file)
    semantic_chunks = fixture_chunks(code_file)
    generated = (chunk for chunk in semantic_chunks)
    result = to_canonical_chunks(code_file, document, generated)
    assert [chunk.chunk_id for chunk in result] == ["class-id", "method-id"]


def test_direct_chunk_conversion_preserves_document_identity() -> None:
    code_file = fixture_code_file()
    document = to_canonical_document(code_file)
    chunks = to_canonical_chunks(code_file, document, fixture_chunks(code_file))
    assert len(chunks) == 2
    assert all(chunk.document_id == document.document_id for chunk in chunks)


def test_part_index_zero_is_explicit_and_has_no_part_location_suffix() -> None:
    code_file = fixture_code_file()
    document = to_canonical_document(code_file)
    chunk = to_canonical_chunks(code_file, document, fixture_chunks(code_file))[0]
    assert chunk.metadata["part_index"] == "0"
    assert chunk.location == "src/auth/service.py:lines:1-3"


def test_all_existing_symbol_types_map_to_structural_type() -> None:
    code_file = fixture_code_file()
    document = to_canonical_document(code_file)
    symbols = tuple(
        SemanticCodeChunk(
            CodeSymbol(
                name=symbol_type.value,
                qualified_name=symbol_type.value,
                symbol_type=symbol_type,
                language=Language.PYTHON,
                path=code_file.relative_path,
                start_line=1,
                end_line=1,
                content=symbol_type.value,
            ),
            symbol_type.value,
            f"{symbol_type.value}-id",
            1,
            1,
        )
        for symbol_type in CodeSymbolType
    )
    chunks = to_canonical_chunks(code_file, document, symbols)
    assert [chunk.structural_type for chunk in chunks] == [
        symbol_type.value for symbol_type in CodeSymbolType
    ]


def test_all_five_languages_are_preserved() -> None:
    for language in Language:
        code_file = CodeFile(
            path=Path(f"fixture.{language.value}"),
            relative_path=f"fixture.{language.value}",
            language=language,
            content="source",
            size=6,
            content_hash="hash",
        )
        document = to_canonical_document(code_file)
        symbol = CodeSymbol(
            name="item",
            qualified_name="item",
            symbol_type=CodeSymbolType.FUNCTION,
            language=language,
            path=code_file.relative_path,
            start_line=1,
            end_line=1,
            content="source",
        )
        chunk = to_canonical_chunks(
            code_file,
            document,
            (SemanticCodeChunk(symbol, "source", "id", 1, 1),),
        )[0]
        assert document.metadata["language"] == language.value
        assert chunk.language == language.value


def test_content_changes_change_document_id() -> None:
    first = fixture_code_file()
    changed = CodeFile(
        path=first.path,
        relative_path=first.relative_path,
        language=first.language,
        content="different source",
        size=15,
        content_hash=sha256(b"different source").hexdigest(),
    )
    assert document_id_for(first) != document_id_for(changed)


def test_same_path_and_content_hash_produce_same_document_id() -> None:
    first = fixture_code_file()
    equivalent = CodeFile(
        path=Path("elsewhere/service.py"),
        relative_path=first.relative_path,
        language=Language.PYTHON,
        content="different representation",
        size=25,
        content_hash=first.content_hash,
    )
    assert document_id_for(first) == document_id_for(equivalent)


def test_document_metadata_and_structure_are_complete() -> None:
    code_file = fixture_code_file()
    document = to_canonical_document(code_file)
    assert document.metadata == {
        "relative_path": "src/auth/service.py",
        "language": "python",
        "content_hash": code_file.content_hash,
    }
    assert document.structure == {
        "relative_path": "src/auth/service.py",
        "language": "python",
    }
    assert document.provenance.path == code_file.relative_path
    assert document.provenance.source_uri == code_file.relative_path
