"""`paths.py`'s run-artefact helpers."""

from __future__ import annotations

from pathlib import Path

import paths


def test_run_dir_creates_the_directory(tmp_path: Path) -> None:
    result = paths.run_dir("abc123", runs_dir=tmp_path / "runs")
    assert result == tmp_path / "runs" / "abc123"
    assert result.is_dir()


def test_span_dump_path_creates_run_dir(tmp_path: Path) -> None:
    result = paths.span_dump_path("abc123", runs_dir=tmp_path / "runs")
    assert result == tmp_path / "runs" / "abc123" / "spans.json"
    assert result.parent.is_dir()
