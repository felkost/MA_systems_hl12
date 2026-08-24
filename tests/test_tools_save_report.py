"""`save_report` confines a model-supplied filename to `output/`, and names
what it writes by when it was written.

The refusal is policy-class: `resolve_report_path` raises `ReportPathError`,
a typed exception, rather than returning a wrong path -- so the `@tool`
wrapper's caught-and-converted `"ERROR: ..."` string is distinguishable, in a
span, from "the disk was full" (`docs/specs/stage-2.md`, "`tools.py` --
guardrails ship with the tools, not after them"). `save_report` also never
overwrites a file that already exists.

The confinement tests call `resolve_report_path` directly: LangChain's
`@tool` decorator turns the function into a `BaseTool` whose `.invoke()`
path adds argument-schema machinery they have no reason to go through to
check one thing. The two naming tests do go through the wrapper, because
the `YYYYMMDD-HHMM` prefix is applied there.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import SecretStr

import tools
from config import Settings
from tools import ReportPathError, resolve_report_path


def test_relative_traversal_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(ReportPathError):
        resolve_report_path("../escape.md", output_dir)


def test_absolute_path_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    outside = tmp_path / "elsewhere" / "report.md"

    with pytest.raises(ReportPathError):
        resolve_report_path(str(outside), output_dir)


def test_symlink_target_outside_output_dir_is_refused(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    outside_dir = tmp_path / "outside"
    output_dir.mkdir()
    outside_dir.mkdir()

    link = output_dir / "escape.md"
    try:
        link.symlink_to(outside_dir / "target.md")
    except OSError:
        pytest.skip("symlink creation requires elevated privileges on this machine")

    with pytest.raises(ReportPathError):
        resolve_report_path("escape.md", output_dir)


def test_plain_filename_resolves_inside_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    resolved = resolve_report_path("my-report.md", output_dir)

    assert resolved.parent == output_dir.resolve()
    assert resolved.name == "my-report.md"


def test_existing_report_is_never_overwritten(tmp_path: Path) -> None:
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = output_dir / "already-there.md"
    existing.write_text("original", encoding="utf-8")

    with pytest.raises(ReportPathError):
        resolve_report_path("already-there.md", output_dir, must_not_exist=True)


def _settings_writing_to(output_dir: Path) -> Settings:
    return Settings(
        openrouter_api_key=SecretStr("test-key"), output_dir=str(output_dir)
    )


def test_saved_filename_starts_with_the_save_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A saved report's name begins with `YYYYMMDD-HHMM`.

    This one goes through the `@tool`-wrapped `save_report`, unlike the
    path-confinement tests above, because the prefix is applied there and
    not in `resolve_report_path`.
    """
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        tools, "load_settings", lambda: _settings_writing_to(output_dir)
    )

    result = tools.save_report.invoke(
        {"filename": "my-report.md", "content": "# Findings\n\nSomething."}
    )

    assert not result.startswith("ERROR:"), result
    written = list(output_dir.glob("*.md"))
    assert len(written) == 1
    assert re.fullmatch(r"\d{8}-\d{4}-my-report\.md", written[0].name)


def test_two_reports_named_alike_do_not_collide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefix carries the minute, so same-minute saves still need
    `must_not_exist=True` to keep them apart -- the second one is refused
    rather than silently overwriting the first."""
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        tools, "load_settings", lambda: _settings_writing_to(output_dir)
    )

    first = tools.save_report.invoke({"filename": "same.md", "content": "first"})
    second = tools.save_report.invoke({"filename": "same.md", "content": "second"})

    assert not first.startswith("ERROR:"), first
    assert second.startswith("ERROR:"), second
    written = list(output_dir.glob("*.md"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8") == "first"
