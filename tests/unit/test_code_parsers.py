from pathlib import Path

import pytest

from knowledge_hub.ingestion.code import (
    CodeFile,
    CodeSymbolType,
    Language,
    ParserRegistry,
    UnsupportedLanguageError,
)


def code_file(language: Language, content: str) -> CodeFile:
    return CodeFile(
        Path("fixture"), "fixture.source", language, content, len(content), "hash"
    )


@pytest.mark.parametrize(
    ("language", "content"),
    [
        (Language.TYPESCRIPT, "class Auth { login() {} }\nfunction boot() {}"),
        (Language.JAVASCRIPT, "class Auth { login() {} }\nfunction boot() {}"),
        (Language.PYTHON, "class Auth:\n    def login(self):\n        pass\n"),
        (Language.C, "int boot() { return 1; }"),
        (Language.CPP, "namespace app { class Auth {}; enum Mode { Fast }; }"),
    ],
)
def test_registry_parses_supported_languages(language: Language, content: str) -> None:
    parser = ParserRegistry.default().get(language)
    result = parser.parse(code_file(language, content))
    assert result.errors == ()
    assert result.symbols


def test_typescript_symbols_have_locations_and_relationships() -> None:
    content = "class Auth {\n  login() {}\n}\nfunction boot() {}\n"
    result = (
        ParserRegistry.default()
        .get(Language.TYPESCRIPT)
        .parse(code_file(Language.TYPESCRIPT, content))
    )
    assert [(symbol.name, symbol.symbol_type) for symbol in result.symbols] == [
        ("Auth", CodeSymbolType.CLASS),
        ("login", CodeSymbolType.METHOD),
        ("boot", CodeSymbolType.FUNCTION),
    ]
    method = result.symbols[1]
    assert method.parent == "Auth"
    assert method.qualified_name == "Auth.login"
    assert (method.start_line, method.end_line) == (2, 2)


def test_python_symbols_include_methods_and_async_functions() -> None:
    content = "class Auth:\n    def login(self):\n        pass\n\nasync def boot():\n    pass\n"
    result = (
        ParserRegistry.default()
        .get(Language.PYTHON)
        .parse(code_file(Language.PYTHON, content))
    )
    assert [
        (symbol.qualified_name, symbol.symbol_type) for symbol in result.symbols
    ] == [
        ("Auth", CodeSymbolType.CLASS),
        ("Auth.login", CodeSymbolType.METHOD),
        ("boot", CodeSymbolType.ASYNC_FUNCTION),
    ]


def test_cpp_symbols_include_namespace_class_enum_and_function() -> None:
    content = "namespace app {\nclass Auth { public: void login() {} };\nenum Mode { Fast };\n}\n"
    result = (
        ParserRegistry.default()
        .get(Language.CPP)
        .parse(code_file(Language.CPP, content))
    )
    assert [
        (symbol.qualified_name, symbol.symbol_type) for symbol in result.symbols
    ] == [
        ("app", CodeSymbolType.NAMESPACE),
        ("app.Auth", CodeSymbolType.CLASS),
        ("app.Auth.login", CodeSymbolType.METHOD),
        ("app.Mode", CodeSymbolType.ENUM),
    ]


@pytest.mark.parametrize(
    ("language", "content"),
    [
        (Language.TYPESCRIPT, "function broken( {"),
        (Language.PYTHON, "def broken(\n"),
    ],
)
def test_malformed_source_returns_errors_without_raising(
    language: Language, content: str
) -> None:
    result = ParserRegistry.default().get(language).parse(code_file(language, content))
    assert result.errors


def test_empty_source_is_deterministic() -> None:
    parser = ParserRegistry.default().get(Language.JAVASCRIPT)
    result = parser.parse(code_file(Language.JAVASCRIPT, ""))
    assert result.symbols == ()
    assert result.errors == ()


def test_registry_rejects_unregistered_language() -> None:
    registry = ParserRegistry({})
    with pytest.raises(UnsupportedLanguageError, match="no parser registered"):
        registry.get(Language.PYTHON)
