from __future__ import annotations

import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from knowledge_hub.ingestion.code import (
    CodeDiscovery,
    CodeSource,
    Language,
    SecureZipIngestor,
    ZipIngestionError,
    ZipLimits,
)


def make_zip(
    tmp_path: Path,
    members: dict[str, str],
    name: str = "code.zip",
) -> Path:
    archive = tmp_path / name

    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        for member, content in members.items():
            zip_file.writestr(member, content)

    return archive


def test_valid_nested_zip_uses_existing_discovery(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path,
        {
            "src/auth/service.ts": "export class Service {}\n",
            "src/models/user.py": "class User:\n    pass\n",
            "cpp/engine.cpp": "int run() { return 1; }\n",
            "README.md": "not code\n",
            "node_modules/fake.js": "ignored\n",
            ".env": "SECRET=ignored\n",
        },
    )

    result = SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert {file.relative_path for file in result.files} == {
        "src/auth/service.ts",
        "src/models/user.py",
        "cpp/engine.cpp",
    }

    assert {file.language for file in result.files} == {
        Language.TYPESCRIPT,
        Language.PYTHON,
        Language.CPP,
    }

    assert any(issue.path.as_posix() == "README.md" for issue in result.unsupported)

    assert not any(
        file.relative_path == "node_modules/fake.js" for file in result.files
    )


@pytest.mark.parametrize(
    "member",
    [
        "../evil.py",
        "nested/../../evil.py",
        "C:/outside.py",
        "/absolute.py",
        "nested\\..\\evil.py",
    ],
)
def test_zip_traversal_and_absolute_members_are_rejected(
    tmp_path: Path,
    member: str,
) -> None:
    archive = make_zip(
        tmp_path,
        {member: "print('unsafe')\n"},
    )

    with pytest.raises(
        ZipIngestionError,
        match="unsafe ZIP member path",
    ):
        SecureZipIngestor().ingest(CodeSource.zip(archive))


def test_rejected_traversal_does_not_create_external_file(
    tmp_path: Path,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "../escape.py": "print('escaped')\n",
        },
    )

    outside_target = tmp_path.parent / "escape.py"

    if outside_target.exists():
        outside_target.unlink()

    with pytest.raises(ZipIngestionError):
        SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert not outside_target.exists()


def test_zip_limits_reject_file_count_total_bytes_and_member_size(
    tmp_path: Path,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "a.py": "1234",
            "b.py": "5678",
        },
    )

    with pytest.raises(
        ZipIngestionError,
        match="file count",
    ):
        SecureZipIngestor(
            ZipLimits(max_file_count=1),
        ).ingest(CodeSource.zip(archive))

    with pytest.raises(
        ZipIngestionError,
        match="extracted byte",
    ):
        SecureZipIngestor(
            ZipLimits(max_extracted_bytes=5),
        ).ingest(CodeSource.zip(archive))

    with pytest.raises(
        ZipIngestionError,
        match="member size",
    ):
        SecureZipIngestor(
            ZipLimits(max_file_size=3),
        ).ingest(CodeSource.zip(archive))


def test_runtime_member_size_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "large.py": "1234567890",
        },
    )

    original_open = ZipFile.open

    def fake_open(
        self: ZipFile,
        name,
        mode="r",
        pwd=None,
        *,
        force_zip64=False,
    ):
        stream = original_open(
            self,
            name,
            mode=mode,
            pwd=pwd,
            force_zip64=force_zip64,
        )

        class OversizedStream:
            def __enter__(self):
                stream.__enter__()
                return self

            def __exit__(self, *args):
                return stream.__exit__(*args)

            def read(self, size=-1):
                data = stream.read(size)

                if data:
                    return data

                return b"x" * 11

        return OversizedStream()

    monkeypatch.setattr(ZipFile, "open", fake_open)

    with pytest.raises(
        ZipIngestionError,
        match="member size",
    ):
        SecureZipIngestor(
            ZipLimits(
                max_file_size=10,
                max_extracted_bytes=100,
            ),
        ).ingest(CodeSource.zip(archive))


def test_runtime_total_extracted_limit_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "a.py": "1234",
            "b.py": "5678",
        },
    )

    original_open = ZipFile.open

    def fake_open(
        self: ZipFile,
        name,
        mode="r",
        pwd=None,
        *,
        force_zip64=False,
    ):
        stream = original_open(
            self,
            name,
            mode=mode,
            pwd=pwd,
            force_zip64=force_zip64,
        )

        class Stream:
            def __enter__(self):
                stream.__enter__()
                return self

            def __exit__(self, *args):
                return stream.__exit__(*args)

            def read(self, size=-1):
                return stream.read(size)

        return Stream()

    monkeypatch.setattr(ZipFile, "open", fake_open)

    with pytest.raises(
        ZipIngestionError,
        match="extracted byte",
    ):
        SecureZipIngestor(
            ZipLimits(
                max_extracted_bytes=5,
                max_file_size=100,
            ),
        ).ingest(CodeSource.zip(archive))


def test_invalid_zip_raises_controlled_error(tmp_path: Path) -> None:
    archive = tmp_path / "invalid.zip"
    archive.write_bytes(b"not a zip")

    with pytest.raises(
        ZipIngestionError,
        match="invalid ZIP archive",
    ):
        SecureZipIngestor().ingest(CodeSource.zip(archive))


def test_encrypted_zip_member_is_rejected(
    tmp_path: Path,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "secret.py": "print('secret')\n",
        },
        "encrypted.zip",
    )

    raw = bytearray(archive.read_bytes())

    # Set the encryption flag in both the local file header and the
    # central directory entry. ZIP flag_bits bit 0 means encrypted.
    local_header = raw.find(b"PK\x03\x04")
    central_header = raw.find(b"PK\x01\x02")

    assert local_header >= 0
    assert central_header >= 0

    local_flags_offset = local_header + 6
    central_flags_offset = central_header + 8

    raw[local_flags_offset] |= 0x01
    raw[local_flags_offset + 1] |= 0x00

    raw[central_flags_offset] |= 0x01
    raw[central_flags_offset + 1] |= 0x00

    archive.write_bytes(raw)

    with pytest.raises(
        ZipIngestionError,
        match="encrypted ZIP member",
    ):
        SecureZipIngestor().ingest(CodeSource.zip(archive))


def test_extraction_directory_is_cleaned_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "src/main.py": "print('ok')\n",
        },
    )

    created: list[Path] = []
    original = tempfile.TemporaryDirectory

    class TrackingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self._directory = original(*args, **kwargs)

        def __enter__(self):
            path = Path(self._directory.__enter__())
            created.append(path)
            return str(path)

        def __exit__(self, *args):
            return self._directory.__exit__(*args)

    monkeypatch.setattr(
        "tempfile.TemporaryDirectory",
        TrackingTemporaryDirectory,
    )

    SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert created
    assert not created[0].exists()


def test_extraction_directory_is_cleaned_after_discovery_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "src/main.py": "print('ok')\n",
        },
    )

    created: list[Path] = []
    original_temp_directory = tempfile.TemporaryDirectory

    class TrackingTemporaryDirectory:
        def __init__(self, *args, **kwargs):
            self._directory = original_temp_directory(
                *args,
                **kwargs,
            )

        def __enter__(self):
            path = Path(self._directory.__enter__())
            created.append(path)
            return str(path)

        def __exit__(self, *args):
            return self._directory.__exit__(*args)

    monkeypatch.setattr(
        "tempfile.TemporaryDirectory",
        TrackingTemporaryDirectory,
    )

    def fail_discovery(self, source):
        raise RuntimeError("forced discovery failure")

    monkeypatch.setattr(
        CodeDiscovery,
        "discover",
        fail_discovery,
    )

    with pytest.raises(RuntimeError, match="forced discovery failure"):
        SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert created
    assert not created[0].exists()


def test_duplicate_member_names_are_handled_deterministically(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "duplicates.zip"

    with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
        zip_file.writestr("src/main.py", "print('first')\n")
        zip_file.writestr("src/main.py", "print('second')\n")

    result = SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert [file.relative_path for file in result.files] == ["src/main.py"]

    assert result.files[0].content == "print('second')\n"


def test_discovery_exclusions_are_preserved_inside_zip(
    tmp_path: Path,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "src/main.py": "print('ok')\n",
            ".git/config": "ignored\n",
            ".cache/cache.py": "ignored\n",
            "dist/generated.js": "ignored\n",
            "coverage/report.js": "ignored\n",
            "node_modules/fake.ts": "ignored\n",
            ".env": "SECRET=ignored\n",
        },
    )

    result = SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert [file.relative_path for file in result.files] == ["src/main.py"]


def test_unsupported_files_are_reported_by_discovery(
    tmp_path: Path,
) -> None:
    archive = make_zip(
        tmp_path,
        {
            "README.md": "documentation\n",
            "src/main.py": "print('ok')\n",
        },
    )

    result = SecureZipIngestor().ingest(CodeSource.zip(archive))

    assert [file.relative_path for file in result.files] == ["src/main.py"]

    assert any(issue.path.as_posix() == "README.md" for issue in result.unsupported)


def test_non_zip_source_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "source"
    directory.mkdir()

    with pytest.raises(
        ZipIngestionError,
        match="requires a ZIP CodeSource",
    ):
        SecureZipIngestor().ingest(
            CodeSource.directory(directory),
        )
