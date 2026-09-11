from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class CodeSourceKind(StrEnum):
    FILE = "file"
    DIRECTORY = "directory"
    ZIP = "zip"


@dataclass(frozen=True)
class CodeSource:
    """A code-ingestion source before discovery or extraction."""

    kind: CodeSourceKind
    path: Path

    @classmethod
    def file(cls, path: str | Path) -> CodeSource:
        return cls(CodeSourceKind.FILE, Path(path))

    @classmethod
    def directory(cls, path: str | Path) -> CodeSource:
        return cls(CodeSourceKind.DIRECTORY, Path(path))

    @classmethod
    def zip(cls, path: str | Path) -> CodeSource:
        return cls(CodeSourceKind.ZIP, Path(path))


class Language(StrEnum):
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    PYTHON = "python"
    C = "c"
    CPP = "cpp"


class CodeSymbolType(StrEnum):
    CLASS = "class"
    METHOD = "method"
    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    INTERFACE = "interface"
    TYPE = "type"
    ENUM = "enum"
    STRUCT = "struct"
    NAMESPACE = "namespace"


@dataclass(frozen=True)
class CodeFile:
    """Normalized source-file data used by a language parser."""

    path: Path
    relative_path: str
    language: Language
    content: str
    size: int
    content_hash: str


@dataclass(frozen=True)
class CodeSymbol:
    """Parser-independent semantic information about one code symbol."""

    name: str
    qualified_name: str
    symbol_type: CodeSymbolType | str
    language: Language
    path: str
    start_line: int
    end_line: int
    content: str
    parent: str | None = None


@dataclass(frozen=True)
class CodeParseError:
    message: str
    path: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class CodeParseResult:
    symbols: tuple[CodeSymbol, ...] = ()
    errors: tuple[CodeParseError, ...] = ()


class LanguageParser(Protocol):
    """Parser contract independent of any AST implementation."""

    language: Language

    def parse(self, code_file: CodeFile) -> CodeParseResult:
        """Parse one normalized file into semantic symbols and parse errors."""
        ...


class CodeIngestionError(Exception):
    """Base error for future code-source validation and parsing operations."""


class SourceValidationError(CodeIngestionError):
    pass


class UnsupportedLanguageError(CodeIngestionError):
    pass


class ParseError(CodeIngestionError):
    pass
