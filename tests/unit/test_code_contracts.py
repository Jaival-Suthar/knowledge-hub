from pathlib import Path

from knowledge_hub.ingestion.code import (
    CodeFile,
    CodeParseResult,
    CodeSource,
    CodeSourceKind,
    CodeSymbol,
    CodeSymbolType,
    Language,
    LanguageParser,
)


def test_code_sources_support_file_directory_and_zip() -> None:
    assert CodeSource.file("src/main.py") == CodeSource(
        CodeSourceKind.FILE, Path("src/main.py")
    )
    assert CodeSource.directory("src") == CodeSource(
        CodeSourceKind.DIRECTORY, Path("src")
    )
    assert CodeSource.zip("source.zip") == CodeSource(
        CodeSourceKind.ZIP, Path("source.zip")
    )


def test_supported_languages_are_explicit() -> None:
    assert {language.value for language in Language} == {
        "typescript",
        "javascript",
        "python",
        "c",
        "cpp",
    }


def test_code_file_represents_normalized_source_data() -> None:
    code_file = CodeFile(
        path=Path("C:/repo/src/main.py"),
        relative_path="src/main.py",
        language=Language.PYTHON,
        content="def main():\n    pass\n",
        size=23,
        content_hash="abc123",
    )
    assert code_file.relative_path == "src/main.py"
    assert code_file.language is Language.PYTHON
    assert code_file.content.startswith("def main")
    assert code_file.size == 23
    assert code_file.content_hash == "abc123"


def test_code_symbol_represents_semantic_relationships() -> None:
    symbol = CodeSymbol(
        name="run",
        qualified_name="Worker.run",
        symbol_type=CodeSymbolType.METHOD,
        language=Language.TYPESCRIPT,
        path="src/worker.ts",
        start_line=8,
        end_line=14,
        parent="Worker",
        content="run(): void {}",
    )
    assert symbol.symbol_type is CodeSymbolType.METHOD
    assert symbol.qualified_name == "Worker.run"
    assert symbol.parent == "Worker"
    assert (symbol.start_line, symbol.end_line) == (8, 14)


def test_language_parser_can_be_implemented_without_ast_dependencies() -> None:
    class FakeParser:
        language = Language.PYTHON

        def parse(self, code_file: CodeFile) -> CodeParseResult:
            return CodeParseResult(
                symbols=(
                    CodeSymbol(
                        name="main",
                        qualified_name="main",
                        symbol_type=CodeSymbolType.FUNCTION,
                        language=code_file.language,
                        path=code_file.relative_path,
                        start_line=1,
                        end_line=2,
                        content=code_file.content,
                    ),
                )
            )

    parser: LanguageParser = FakeParser()
    code_file = CodeFile(
        Path("main.py"), "main.py", Language.PYTHON, "def main():\n", 12, "hash"
    )
    result = parser.parse(code_file)
    assert result.symbols[0].name == "main"
    assert result.errors == ()
