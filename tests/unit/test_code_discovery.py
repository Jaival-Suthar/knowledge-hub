from pathlib import Path

from knowledge_hub.ingestion.code import (
    CodeDiscovery,
    CodeSource,
    DiscoveryReason,
    Language,
    classify_language,
)


def write_file(root: Path, relative: str, content: str = "source\n") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_individual_file_is_classified_and_preserves_relative_path(
    tmp_path: Path,
) -> None:
    path = write_file(tmp_path, "service.ts", "export const service = true;\n")
    result = CodeDiscovery().discover(CodeSource.file(path))
    assert len(result.files) == 1
    assert result.files[0].relative_path == "service.ts"
    assert result.files[0].language is Language.TYPESCRIPT


def test_directory_discovery_is_recursive_and_maps_extensions(tmp_path: Path) -> None:
    write_file(tmp_path, "src/auth/service.ts")
    write_file(tmp_path, "src/utils/parser.js")
    write_file(tmp_path, "src/models/user.py")
    write_file(tmp_path, "cpp/engine.cpp")
    write_file(tmp_path, "tests/auth.test.ts")

    result = CodeDiscovery().discover(CodeSource.directory(tmp_path))
    assert {file.relative_path for file in result.files} == {
        "src/auth/service.ts",
        "src/utils/parser.js",
        "src/models/user.py",
        "cpp/engine.cpp",
        "tests/auth.test.ts",
    }
    assert result.statistics.language_counts == {
        Language.TYPESCRIPT: 2,
        Language.JAVASCRIPT: 1,
        Language.PYTHON: 1,
        Language.CPP: 1,
    }


def test_nested_exclusions_and_generated_files_are_traceable(tmp_path: Path) -> None:
    write_file(tmp_path, "node_modules/fake.ts")
    write_file(tmp_path, "dist/generated.js")
    write_file(tmp_path, ".git/ignored.ts")
    write_file(tmp_path, ".cache/cache.ts")
    write_file(tmp_path, "coverage/report.js")
    write_file(tmp_path, "generated.map")
    write_file(tmp_path, ".env")
    write_file(tmp_path, "package-lock.json")

    result = CodeDiscovery().discover(CodeSource.directory(tmp_path))
    assert not result.files
    assert {issue.reason for issue in result.excluded} >= {
        DiscoveryReason.EXCLUDED_DIRECTORY,
        DiscoveryReason.GENERATED_FILE,
        DiscoveryReason.LOCKFILE,
    }


def test_unsupported_extensions_are_separate_from_code_files(tmp_path: Path) -> None:
    write_file(tmp_path, "notes.xyz")
    result = CodeDiscovery().discover(CodeSource.directory(tmp_path))
    assert not result.files
    assert result.unsupported[0].reason is DiscoveryReason.UNSUPPORTED_EXTENSION


def test_size_limit_reports_skipped_file(tmp_path: Path) -> None:
    write_file(tmp_path, "large.py", "123456789")
    result = CodeDiscovery(max_file_size=4).discover(CodeSource.directory(tmp_path))
    assert not result.files
    assert result.skipped[0].reason is DiscoveryReason.FILE_SIZE_LIMIT
    assert result.statistics.skipped_count == 1


def test_all_supported_extensions_are_classified() -> None:
    assert classify_language("a.ts") is Language.TYPESCRIPT
    assert classify_language("a.tsx") is Language.TYPESCRIPT
    assert classify_language("a.js") is Language.JAVASCRIPT
    assert classify_language("a.jsx") is Language.JAVASCRIPT
    assert classify_language("a.mjs") is Language.JAVASCRIPT
    assert classify_language("a.cjs") is Language.JAVASCRIPT
    assert classify_language("a.py") is Language.PYTHON
    assert classify_language("a.c") is Language.C
    assert classify_language("a.h") is Language.C
    assert classify_language("a.cc") is Language.CPP
    assert classify_language("a.cpp") is Language.CPP
    assert classify_language("a.cxx") is Language.CPP
    assert classify_language("a.hpp") is Language.CPP


def test_zip_source_is_explicitly_not_supported(tmp_path: Path) -> None:
    result = CodeDiscovery().discover(CodeSource.zip(tmp_path / "source.zip"))
    assert result.skipped[0].reason is DiscoveryReason.ZIP_NOT_SUPPORTED
