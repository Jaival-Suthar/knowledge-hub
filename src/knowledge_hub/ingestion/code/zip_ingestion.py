from __future__ import annotations

import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

from .contracts import CodeSource, CodeSourceKind, ZipIngestionError
from .discovery import CodeDiscovery, DiscoveryResult


@dataclass(frozen=True)
class ZipLimits:
    """Resource limits applied while safely ingesting a ZIP archive."""

    max_extracted_bytes: int = 100 * 1024 * 1024
    max_file_count: int = 10_000
    max_file_size: int = 10 * 1024 * 1024

    def __post_init__(self) -> None:
        if any(
            value <= 0
            for value in (
                self.max_extracted_bytes,
                self.max_file_count,
                self.max_file_size,
            )
        ):
            raise ValueError("ZIP limits must be positive")


class SecureZipIngestor:
    """Safely extract a ZIP codebase and delegate discovery to CodeDiscovery."""

    def __init__(
        self,
        limits: ZipLimits | None = None,
        discovery: CodeDiscovery | None = None,
    ) -> None:
        self.limits = limits or ZipLimits()
        self.discovery = discovery or CodeDiscovery()

    def ingest(self, source: CodeSource) -> DiscoveryResult:
        """Validate, safely extract, and discover a ZIP codebase."""
        if source.kind is not CodeSourceKind.ZIP:
            raise ZipIngestionError("SecureZipIngestor requires a ZIP CodeSource")

        archive = source.path.resolve()

        if not archive.is_file():
            raise ZipIngestionError(f"ZIP source is not a file: {source.path}")

        try:
            with (
                ZipFile(archive) as zip_file,
                tempfile.TemporaryDirectory(prefix="knowledge-hub-zip-") as temp,
            ):
                root = Path(temp).resolve()

                members = self._validate_members(zip_file, root)
                self._extract(zip_file, members, root)

                return self.discovery.discover(CodeSource.directory(root))

        except BadZipFile as error:
            raise ZipIngestionError(f"invalid ZIP archive: {archive}") from error
        except OSError as error:
            raise ZipIngestionError(f"unable to read ZIP archive: {archive}") from error

    def _validate_members(
        self,
        zip_file: ZipFile,
        root: Path,
    ) -> list[tuple[ZipInfo, PurePosixPath]]:
        """Validate every ZIP member before anything is extracted."""
        members: list[tuple[ZipInfo, PurePosixPath]] = []

        declared_total_bytes = 0
        file_count = 0

        for info in zip_file.infolist():
            normalized = self._normalize_member_name(info.filename)

            if normalized is None:
                raise ZipIngestionError(f"unsafe ZIP member path: {info.filename}")

            if self._is_symlink(info):
                raise ZipIngestionError(f"unsafe symlink member: {info.filename}")

            if info.flag_bits & 0x1:
                raise ZipIngestionError(
                    f"encrypted ZIP member is unsupported: {info.filename}"
                )

            destination = root.joinpath(*normalized.parts).resolve()

            try:
                destination.relative_to(root)
            except ValueError as error:
                raise ZipIngestionError(
                    f"ZIP member escapes extraction root: {info.filename}"
                ) from error

            if info.is_dir() or info.filename.endswith(("/", "\\")):
                continue

            file_count += 1

            if file_count > self.limits.max_file_count:
                raise ZipIngestionError("ZIP file count limit exceeded")

            if info.file_size > self.limits.max_file_size:
                raise ZipIngestionError(
                    f"ZIP member size limit exceeded: {info.filename}"
                )

            declared_total_bytes += info.file_size

            if declared_total_bytes > self.limits.max_extracted_bytes:
                raise ZipIngestionError("ZIP extracted byte limit exceeded")

            members.append((info, normalized))

        return members

    def _extract(
        self,
        zip_file: ZipFile,
        members: list[tuple[ZipInfo, PurePosixPath]],
        root: Path,
    ) -> None:
        """Stream validated members into the temporary extraction root."""
        extracted_bytes = 0

        for info, relative in members:
            destination = root.joinpath(*relative.parts).resolve()

            try:
                destination.relative_to(root)
            except ValueError as error:
                raise ZipIngestionError(
                    f"ZIP member escapes extraction root: {info.filename}"
                ) from error

            destination.parent.mkdir(parents=True, exist_ok=True)

            member_bytes = 0

            try:
                with (
                    zip_file.open(info) as source,
                    destination.open("wb") as target,
                ):
                    while block := source.read(1024 * 1024):
                        member_bytes += len(block)

                        if member_bytes > self.limits.max_file_size:
                            raise ZipIngestionError(
                                f"ZIP member size limit exceeded: {info.filename}"
                            )

                        extracted_bytes += len(block)

                        if extracted_bytes > self.limits.max_extracted_bytes:
                            raise ZipIngestionError("ZIP extracted byte limit exceeded")

                        target.write(block)

            except ZipIngestionError:
                raise
            except OSError as error:
                raise ZipIngestionError(
                    f"unable to extract ZIP member: {info.filename}"
                ) from error

    @staticmethod
    def _normalize_member_name(name: str) -> PurePosixPath | None:
        """Normalize a ZIP member name while rejecting unsafe paths."""
        if not name or "\x00" in name:
            return None

        normalized = name.replace("\\", "/")

        if normalized.startswith("/"):
            return None

        if re.match(r"^[A-Za-z]:/", normalized):
            return None

        parts = PurePosixPath(normalized).parts

        if not parts:
            return None

        if any(part in {"", ".", ".."} for part in parts):
            return None

        return PurePosixPath(*parts)

    @staticmethod
    def _is_symlink(info: ZipInfo) -> bool:
        """Return whether a ZIP member is encoded as a symbolic link."""
        mode = (info.external_attr >> 16) & 0xFFFF
        return stat.S_ISLNK(mode)
