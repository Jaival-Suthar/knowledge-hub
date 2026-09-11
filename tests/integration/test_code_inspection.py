from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ARCHIVE = Path("data/raw/perfengine.zip")


def run_inspection(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "knowledge_hub.ingestion.code.inspect", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_inspection_cli_prints_real_perfengine_transformation() -> None:
    result = run_inspection(str(ARCHIVE))
    assert result.returncode == 0, result.stderr
    for section in (
        "SOURCE",
        "DISCOVERY",
        "PARSING",
        "CHUNKING",
        "CANONICALIZATION",
        "REPRESENTATIVE FILES",
        "REPRESENTATIVE SYMBOLS",
        "REPRESENTATIVE CHUNKS",
    ):
        assert section in result.stdout
    assert "perfengine/bin/perfengine.js" in result.stdout
    assert "processItems" in result.stdout
    assert "document_id:" in result.stdout
    assert "content:" in result.stdout


def test_inspection_cli_file_option_shows_one_real_file() -> None:
    relative_file = "perfengine/fixtures/node-service/index.js"
    result = run_inspection(str(ARCHIVE), "--file", relative_file)
    assert result.returncode == 0, result.stderr
    assert "parsed files: 1" in result.stdout
    assert relative_file in result.stdout
    assert "processItems" in result.stdout


def test_inspection_cli_invalid_input_is_controlled(tmp_path: Path) -> None:
    result = run_inspection(str(tmp_path / "missing.zip"))
    assert result.returncode != 0
    assert "ZIP source is not a file" in result.stderr
