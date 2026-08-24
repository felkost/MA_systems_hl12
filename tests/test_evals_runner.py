"""`evals/runner.py` -- stage 5's own slice (D5.1, `docs/specs/stage-5.md`):
the span-JSON schema and `load_run()`. `build_llm_test_case()` (turning a
run into a DeepEval `LLMTestCase`) ships at stage 8, against this schema.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals import runner


def _write_spans(tmp_path: Path, run_id: str, spans: list[dict]) -> None:
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "spans.json").write_text(json.dumps(spans), encoding="utf-8")


def _span(**overrides: object) -> dict:
    base: dict[str, object] = {
        "trace_id": "a" * 32,
        "span_id": "b" * 16,
        "parent_span_id": None,
        "name": "repl.question",
        "start_time": 1,
        "end_time": 2,
        "status": "OK",
        "attributes": {"run_id": "run-1"},
    }
    base.update(overrides)
    return base


def test_load_run_parses_a_hand_built_spans_json_fixture(tmp_path: Path) -> None:
    _write_spans(tmp_path, "run-1", [_span()])
    result = runner.load_run("run-1", runs_dir=tmp_path)
    assert result.run_id == "run-1"
    assert len(result.spans) == 1
    assert result.spans[0]["name"] == "repl.question"


def test_load_run_raises_on_a_span_missing_a_required_field(tmp_path: Path) -> None:
    broken = _span()
    del broken["trace_id"]
    _write_spans(tmp_path, "run-2", [broken])
    with pytest.raises(ValueError, match="trace_id"):
        runner.load_run("run-2", runs_dir=tmp_path)


def test_load_run_raises_on_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        runner.load_run("does-not-exist", runs_dir=tmp_path)
