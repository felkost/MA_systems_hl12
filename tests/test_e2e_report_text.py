"""`report_text_or_closing_message`.

Offline: no live call, a hand-built `SupervisorLiveRun` and a real
`tmp_path` standing in for the per-case `output_dir` `test_golden_dataset`
redirects `save_report` into.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.runner import RunSpans
from tests.live_supervisor import SupervisorLiveRun
from tests.test_e2e import report_text_or_closing_message


def _live(output: str) -> SupervisorLiveRun:
    return SupervisorLiveRun(
        run_id="r1",
        output=output,
        spans=RunSpans(run_id="r1", spans=[]),
        messages=[],
    )


def test_reads_the_saved_report_when_one_was_written(tmp_path: Path) -> None:
    report_path = tmp_path / "20260822-1200-topic.md"
    report_path.write_text(
        "# Full report body\n\nReal findings here.", encoding="utf-8"
    )

    result = report_text_or_closing_message(tmp_path, _live("Report saved to: ..."))

    assert result == "# Full report body\n\nReal findings here."


def test_falls_back_to_the_closing_message_when_nothing_was_saved(
    tmp_path: Path,
) -> None:
    result = report_text_or_closing_message(
        tmp_path, _live("This request is outside my scope.")
    )

    assert result == "This request is outside my scope."


def test_raises_on_more_than_one_saved_report(tmp_path: Path) -> None:
    (tmp_path / "20260822-1200-a.md").write_text("a", encoding="utf-8")
    (tmp_path / "20260822-1201-b.md").write_text("b", encoding="utf-8")

    with pytest.raises(RuntimeError):
        report_text_or_closing_message(tmp_path, _live("..."))
