from pathlib import Path

from knowledge_hub.ingestion.code import (
    CodeChunker,
    CodeFile,
    CodeSymbolType,
    Language,
    ParserRegistry,
)


def parse(language: Language, content: str):
    code_file = CodeFile(
        Path("fixture.source"),
        "fixture.source",
        language,
        content,
        len(content),
        "hash",
    )
    return code_file, ParserRegistry.default().get(language).parse(code_file)


def test_typescript_class_and_methods_are_semantic_chunks() -> None:
    content = "class AuthService {\n  login() {}\n  refreshToken() {}\n}\n"
    code_file, result = parse(Language.TYPESCRIPT, content)

    chunks = CodeChunker().chunk(code_file, result)

    assert [chunk.qualified_name for chunk in chunks] == [
        "AuthService",
        "AuthService.login",
        "AuthService.refreshToken",
    ]
    assert chunks[0].symbol_type is CodeSymbolType.CLASS
    assert chunks[1].parent == "AuthService"
    assert chunks[1].content == result.symbols[1].content
    assert (chunks[1].start_line, chunks[1].end_line) == (2, 2)


def test_python_functions_and_methods_keep_parent_context() -> None:
    content = (
        "class User:\n"
        "    def validate(self):\n"
        "        return True\n"
        "\n"
        "def build_user():\n"
        "    return User()\n"
    )
    code_file, result = parse(Language.PYTHON, content)

    chunks = CodeChunker().chunk(code_file, result)

    assert [chunk.qualified_name for chunk in chunks] == [
        "User",
        "User.validate",
        "build_user",
    ]
    assert chunks[1].parent == "User"
    assert chunks[2].parent is None


def test_javascript_and_cpp_symbols_are_chunkable() -> None:
    js_file, js_result = parse(
        Language.JAVASCRIPT,
        "class Service { run() {} }\nfunction boot() {}",
    )
    cpp_file, cpp_result = parse(
        Language.CPP,
        "namespace app {\nclass Service { public: void run() {} };\n}\n",
    )

    js_chunks = CodeChunker().chunk(js_file, js_result)
    cpp_chunks = CodeChunker().chunk(cpp_file, cpp_result)

    assert {chunk.qualified_name for chunk in js_chunks} >= {
        "Service",
        "Service.run",
        "boot",
    }
    assert {chunk.qualified_name for chunk in cpp_chunks} >= {
        "app",
        "app.Service",
        "app.Service.run",
    }


def test_oversized_class_is_subdivided_into_methods() -> None:
    content = (
        "class LargeService {\n"
        "  methodA() { return 'aaaaaaaaaaaaaaaa'; }\n"
        "  methodB() { return 'bbbbbbbbbbbbbbbb'; }\n"
        "  methodC() { return 'cccccccccccccccc'; }\n"
        "}\n"
    )
    code_file, result = parse(Language.TYPESCRIPT, content)

    chunks = CodeChunker(max_chunk_chars=50).chunk(code_file, result)

    assert [chunk.qualified_name for chunk in chunks] == [
        "LargeService.methodA",
        "LargeService.methodB",
        "LargeService.methodC",
    ]
    assert all(chunk.parent == "LargeService" for chunk in chunks)
    assert all(
        chunk.content == symbol.content
        for chunk, symbol in zip(chunks, result.symbols[1:])
    )


def test_oversized_nested_function_uses_nested_symbol() -> None:
    content = (
        "def outer():\n    def inner():\n        return 'small'\n    return inner()\n"
    )
    code_file, result = parse(Language.PYTHON, content)

    chunks = CodeChunker(max_chunk_chars=45).chunk(code_file, result)

    assert [chunk.qualified_name for chunk in chunks] == ["outer.inner"]
    assert chunks[0].parent == "outer"


def test_oversized_leaf_has_deterministic_exact_fallback() -> None:
    content = "function huge() {\n" + ("  const value = 'abcdefghij';\n" * 8) + "}\n"
    code_file, result = parse(Language.JAVASCRIPT, content)

    chunks = CodeChunker(max_chunk_chars=50).chunk(code_file, result)

    assert len(chunks) > 1
    assert [chunk.part_index for chunk in chunks] == list(range(len(chunks)))
    assert "".join(chunk.content for chunk in chunks) == result.symbols[0].content
    assert all(chunk.qualified_name == "huge" for chunk in chunks)
    assert all(chunk.parent is None for chunk in chunks)


def test_empty_malformed_and_repeated_inputs_are_safe_and_deterministic() -> None:
    empty_file, empty_result = parse(Language.PYTHON, "")
    malformed_file, malformed_result = parse(Language.PYTHON, "def broken(\n")

    chunker = CodeChunker()

    assert chunker.chunk(empty_file, empty_result) == ()
    assert chunker.chunk(malformed_file, malformed_result) == ()

    first = chunker.chunk(*parse(Language.TYPESCRIPT, "function boot() {}"))
    second = chunker.chunk(*parse(Language.TYPESCRIPT, "function boot() {}"))

    assert first == second
