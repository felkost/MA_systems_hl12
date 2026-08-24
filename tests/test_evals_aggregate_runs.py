"""`evals.aggregate_runs` (stage 9e, D9e.2a).

Offline: DeepEval's own `.latest_run_full.json` is never touched -- every
test monkeypatches the module-level `LATEST_FULL_TEST_RUN_FILE_PATH` both
modules read, pointing it at a hand-written fixture file under `tmp_path`
instead. `merge_runs`/`main` are tested against hand-written snapshot files
directly, matching `tests/test_record_case_cost.py`'s own pattern of writing
into a real, throwaway `runs/<uuid>/` and cleaning up after itself.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterator
from uuid import uuid4

import pytest

import paths
from evals import aggregate_runs, summarize_e2e


def _metric(name: str, score: float, threshold: float, success: bool) -> dict:
    return {"name": name, "score": score, "threshold": threshold, "success": success}


def _case_costs_free_run(name: str) -> dict:
    return {
        "testCases": [
            {
                "name": name,
                "metricsData": [
                    _metric("Answer Relevancy", 0.9, 0.7, True),
                    _metric("Correctness [GEval]", 0.8, 0.6, True),
                    _metric("Citation Presence [GEval]", 0.7, 0.6, True),
                ],
            }
        ]
    }


@pytest.fixture
def eval_run_id() -> Iterator[str]:
    run_id = str(uuid4())
    try:
        yield run_id
    finally:
        shutil.rmtree(paths.run_dir(run_id), ignore_errors=True)


def test_snapshot_latest_run_copies_deepevals_own_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_run_id: str
) -> None:
    fake_source = tmp_path / "latest_run_full.json"
    fake_source.write_text(json.dumps(_case_costs_free_run("core-a")), encoding="utf-8")
    monkeypatch.setattr(
        summarize_e2e, "LATEST_FULL_TEST_RUN_FILE_PATH", str(fake_source)
    )
    monkeypatch.setattr(
        aggregate_runs, "LATEST_FULL_TEST_RUN_FILE_PATH", str(fake_source)
    )

    destination = aggregate_runs.snapshot_latest_run(eval_run_id, "e2e")

    assert destination == paths.run_dir(eval_run_id) / "deepeval-run-e2e.json"
    assert json.loads(destination.read_text(encoding="utf-8")) == json.loads(
        fake_source.read_text(encoding="utf-8")
    )


def test_snapshot_latest_run_raises_load_latest_runs_own_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_run_id: str
) -> None:
    missing = tmp_path / "never-written.json"
    monkeypatch.setattr(summarize_e2e, "LATEST_FULL_TEST_RUN_FILE_PATH", str(missing))
    monkeypatch.setattr(aggregate_runs, "LATEST_FULL_TEST_RUN_FILE_PATH", str(missing))

    with pytest.raises(FileNotFoundError, match="deepeval test run"):
        aggregate_runs.snapshot_latest_run(eval_run_id, "e2e")


def test_collect_invocation_artefacts_appends_costs_and_copies_spans(
    eval_run_id: str,
) -> None:
    """Stage 9e phase 1b: bridging two invocations' own `case-costs.jsonl`/
    `spans/` was done by hand at the live checkpoint (`cat >>`, `cp`) --
    this is that step made repeatable, since phase 6's n=3 repeats it six
    times."""
    source_id = str(uuid4())
    try:
        target_dir = paths.run_dir(eval_run_id)
        source_dir = paths.run_dir(source_id)
        (target_dir / "case-costs.jsonl").write_text(
            '{"case_id": "a", "agent_cost_usd": 0.01}\n', encoding="utf-8"
        )
        (source_dir / "case-costs.jsonl").write_text(
            '{"case_id": "b", "agent_cost_usd": 0.02}\n', encoding="utf-8"
        )
        (source_dir / "spans").mkdir()
        (source_dir / "spans" / "b.json").write_text("[]", encoding="utf-8")

        copied = aggregate_runs.collect_invocation_artefacts(eval_run_id, source_id)

        assert copied == 1
        merged_costs = (target_dir / "case-costs.jsonl").read_text(encoding="utf-8")
        assert merged_costs.count("case_id") == 2
        assert (target_dir / "spans" / "b.json").is_file()
    finally:
        shutil.rmtree(paths.run_dir(source_id), ignore_errors=True)


def test_collect_invocation_artefacts_is_a_no_op_for_a_source_with_nothing(
    eval_run_id: str,
) -> None:
    source_id = str(uuid4())
    assert aggregate_runs.collect_invocation_artefacts(eval_run_id, source_id) == 0


def test_merge_runs_concatenates_test_cases_from_every_snapshot(
    tmp_path: Path,
) -> None:
    first = tmp_path / "deepeval-run-e2e.json"
    second = tmp_path / "deepeval-run-components.json"
    first.write_text(json.dumps(_case_costs_free_run("core-a")), encoding="utf-8")
    second.write_text(json.dumps(_case_costs_free_run("core-b")), encoding="utf-8")

    merged = aggregate_runs.merge_runs([first, second])

    assert [case["name"] for case in merged["testCases"]] == ["core-a", "core-b"]


def test_aggregate_main_writes_a_summary_over_the_merged_run(
    eval_run_id: str,
) -> None:
    run_dir = paths.run_dir(eval_run_id)
    (run_dir / "deepeval-run-e2e.json").write_text(
        json.dumps(_case_costs_free_run("core-single-vs-multi-agent")),
        encoding="utf-8",
    )
    (run_dir / "deepeval-run-components.json").write_text(
        json.dumps(_case_costs_free_run("edge-out-of-scope-recipe")),
        encoding="utf-8",
    )

    written_to = aggregate_runs.main(eval_run_id, ["e2e", "components"])

    assert written_to == run_dir
    summary = (run_dir / "summary.md").read_text(encoding="utf-8")
    assert "Overall: 2/2 passed" in summary
