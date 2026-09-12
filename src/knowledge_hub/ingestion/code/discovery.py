from __future__ import annotations

import os
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path

from .contracts import (
    CodeFile,
    CodeSource,
    CodeSourceKind,
    Language,
    SourceValidationError,
)

SUPPORTED_EXTENSIONS: dict[str, Language] = {
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    ".py": Language.PYTHON,
    ".c": Language.C,
    ".h": Language.C,
    ".cc": Language.CPP,
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".hpp": Language.CPP,
}

EXCLUDED_DIRECTORIES = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "__pycache__",
        "dist",
        "build",
        "coverage",
        ".cache",
    }
)
EXCLUDED_FILENAMES = frozenset(
    {".env", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
)
EXCLUDED_SUFFIXES = frozenset({".map"})
DEFAULT_MAX_FILE_SIZE = 10 * 1024 * 1024


class DiscoveryReason(StrEnum):
    EXCLUDED_DIRECTORY = "excluded_directory"
    GENERATED_FILE = "generated_file"
    LOCKFILE = "lockfile"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    FILE_SIZE_LIMIT = "file_size_limit"
    UNREADABLE_FILE = "unreadable_file"
    INVALID_SOURCE = "invalid_source"
    ZIP_NOT_SUPPORTED = "zip_not_supported"


@dataclass(frozen=True)
class DiscoveryIssue:
    path: Path
    reason: DiscoveryReason
    detail: str | None = None


@dataclass(frozen=True)
class DiscoveryStatistics:
    total_files_encountered: int
    supported_count: int
    excluded_count: int
    unsupported_count: int
    skipped_count: int
    total_bytes: int
    language_counts: dict[Language, int]


@dataclass(frozen=True)
class DiscoveryResult:
    files: tuple[CodeFile, ...]
    excluded: tuple[DiscoveryIssue, ...]
    unsupported: tuple[DiscoveryIssue, ...]
    skipped: tuple[DiscoveryIssue, ...]
    statistics: DiscoveryStatistics


def classify_language(path: str | Path) -> Language | None:
    return SUPPORTED_EXTENSIONS.get(Path(path).suffix.lower())


class CodeDiscovery:
    """Safely discover supported source files without parsing their contents."""

    def __init__(self, max_file_size: int = DEFAULT_MAX_FILE_SIZE) -> None:
        if max_file_size <= 0:
            raise ValueError("max_file_size must be positive")
        self.max_file_size = max_file_size

    def discover(self, source: CodeSource) -> DiscoveryResult:
        root = source.path.resolve()
        if source.kind is CodeSourceKind.ZIP:
            return self._result(
                [], [], [], [DiscoveryIssue(root, DiscoveryReason.ZIP_NOT_SUPPORTED)]
            )
        if not root.exists():
            raise SourceValidationError(f"source does not exist: {source.path}")
        if source.kind is CodeSourceKind.FILE:
            if not root.is_file():
                raise SourceValidationError(f"source is not a file: {source.path}")
            return self._discover_paths(root.parent, [root])
        if not root.is_dir():
            raise SourceValidationError(f"source is not a directory: {source.path}")
        return self._discover_directory(root)

    def _discover_directory(self, root: Path) -> DiscoveryResult:
        paths: list[Path] = []
        excluded: list[DiscoveryIssue] = []
        for current, directories, filenames in os.walk(
            root, topdown=True, followlinks=False
        ):
            current_path = Path(current)
            kept_directories: list[str] = []
            for directory in sorted(directories):
                directory_path = current_path / directory
                if directory in EXCLUDED_DIRECTORIES:
                    excluded.append(
                        DiscoveryIssue(
                            directory_path.relative_to(root),
                            DiscoveryReason.EXCLUDED_DIRECTORY,
                        )
                    )
                else:
                    kept_directories.append(directory)
            directories[:] = kept_directories
            paths.extend(current_path / filename for filename in sorted(filenames))
        result = self._discover_paths(root, paths)
        return DiscoveryResult(
            result.files,
            (*excluded, *result.excluded),
            result.unsupported,
            result.skipped,
            DiscoveryStatistics(
                result.statistics.total_files_encountered,
                result.statistics.supported_count,
                len(excluded) + result.statistics.excluded_count,
                result.statistics.unsupported_count,
                result.statistics.skipped_count,
                result.statistics.total_bytes,
                result.statistics.language_counts,
            ),
        )

    def _discover_paths(self, root: Path, paths: list[Path]) -> DiscoveryResult:
        files: list[CodeFile] = []
        excluded: list[DiscoveryIssue] = []
        unsupported: list[DiscoveryIssue] = []
        skipped: list[DiscoveryIssue] = []
        total_bytes = 0
        language_counts: Counter[Language] = Counter()

        for path in sorted(paths):
            relative_path = path.relative_to(root)
            if any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts[:-1]):
                excluded.append(
                    DiscoveryIssue(relative_path, DiscoveryReason.EXCLUDED_DIRECTORY)
                )
                continue
            if path.name in EXCLUDED_FILENAMES:
                excluded.append(
                    DiscoveryIssue(relative_path, self._filename_reason(path.name))
                )
                continue
            if path.suffix.lower() in EXCLUDED_SUFFIXES:
                excluded.append(
                    DiscoveryIssue(relative_path, DiscoveryReason.GENERATED_FILE)
                )
                continue
            language = classify_language(path)
            if language is None:
                unsupported.append(
                    DiscoveryIssue(relative_path, DiscoveryReason.UNSUPPORTED_EXTENSION)
                )
                continue
            try:
                size = path.stat().st_size
            except OSError as error:
                skipped.append(
                    DiscoveryIssue(
                        relative_path, DiscoveryReason.UNREADABLE_FILE, str(error)
                    )
                )
                continue
            if size > self.max_file_size:
                skipped.append(
                    DiscoveryIssue(
                        relative_path, DiscoveryReason.FILE_SIZE_LIMIT, str(size)
                    )
                )
                continue
            try:
                raw = path.read_bytes()
                content = raw.decode("utf-8")
            except (OSError, UnicodeDecodeError) as error:
                skipped.append(
                    DiscoveryIssue(
                        relative_path, DiscoveryReason.UNREADABLE_FILE, str(error)
                    )
                )
                continue
            files.append(
                CodeFile(
                    path,
                    relative_path.as_posix(),
                    language,
                    content,
                    size,
                    sha256(raw).hexdigest(),
                )
            )
            total_bytes += size
            language_counts[language] += 1

        return self._result(
            files,
            excluded,
            unsupported,
            skipped,
            total_bytes,
            language_counts,
            len(paths),
        )

    @staticmethod
    def _filename_reason(name: str) -> DiscoveryReason:
        return (
            DiscoveryReason.LOCKFILE
            if name.endswith("lock") or "lock" in name
            else DiscoveryReason.GENERATED_FILE
        )

    @staticmethod
    def _result(
        files: list[CodeFile],
        excluded: list[DiscoveryIssue],
        unsupported: list[DiscoveryIssue],
        skipped: list[DiscoveryIssue],
        total_bytes: int = 0,
        language_counts: Counter[Language] | None = None,
        total_files: int | None = None,
    ) -> DiscoveryResult:
        counts = language_counts or Counter()
        statistics = DiscoveryStatistics(
            total_files if total_files is not None else len(files),
            len(files),
            len(excluded),
            len(unsupported),
            len(skipped),
            total_bytes,
            dict(counts),
        )
        return DiscoveryResult(
            tuple(files),
            tuple(excluded),
            tuple(unsupported),
            tuple(skipped),
            statistics,
        )
