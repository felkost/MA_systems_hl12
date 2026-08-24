"""`evals.repeat_summary` (stage 9e phase 6, D9e.11's own reporting shape).

Offline, hand-written `eval-results.json` fixtures under `tmp_path`/a
throwaway `runs/<uuid>/` -- no live spend, matching
`tests/test_evals_aggregate_runs.py`'s own pattern.
"""

from __future__ import annotations

import json
import shutil
from typing import Iterator
from uuid import uuid4

import pytest

import paths
from evals import repeat_summary


def _metric(name: str, success: bool, score: float | None = None) -> dict:
    return {
        "name": name,
        "success": success,
        "score": score if score is not None else (1.0 if success else 0.0),
    }


def _case(name: str, success: bool, errored: bool = False) -> dict:
    metrics = [_metric("Correctness [GEval]", success)]
    if errored:
        metrics.append({"name": "Answer Relevancy", "success": False, "score": None})
    return {"name": name, "metricsData": metrics}


def _write_eval_results(
    run_id: str, cases: list[dict], component_cases: dict[str, list[dict]]
) -> None:
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval-results.json").write_text(
        json.dumps(
            {"cases": cases, "component_cases": component_cases, "case_costs": []}
        ),
        encoding="utf-8",
    )


@pytest.fixture
def three_run_ids() -> Iterator[list[str]]:
    ids = [str(uuid4()) for _ in range(3)]
    try:
        yield ids
    finally:
        for run_id in ids:
            shutil.rmtree(paths.run_dir(run_id), ignore_errors=True)


def test_load_eval_results_reads_the_persisted_file(three_run_ids: list[str]) -> None:
    run_id = three_run_ids[0]
    _write_eval_results(run_id, [_case("core-a", True)], {})

    result = repeat_summary.load_eval_results(run_id)

    assert result["cases"][0]["name"] == "core-a"


def test_aggregate_repetitions_buckets_stable_pass_flaky_stable_fail(
    three_run_ids: list[str],
) -> None:
    # core-a passes every rep (stable pass), core-b passes only rep 2 (flaky),
    # core-c fails every rep (stable fail).
    _write_eval_results(
        three_run_ids[0],
        [_case("core-a", True), _case("core-b", False), _case("core-c", False)],
        {},
    )
    _write_eval_results(
        three_run_ids[1],
        [_case("core-a", True), _case("core-b", True), _case("core-c", False)],
        {},
    )
    _write_eval_results(
        three_run_ids[2],
        [_case("core-a", True), _case("core-b", False), _case("core-c", False)],
        {},
    )

    agg = repeat_summary.aggregate_repetitions(three_run_ids)

    assert agg["stable_pass"] == ["core-a"]
    assert agg["flaky"] == ["core-b"]
    assert agg["stable_fail"] == ["core-c"]
    assert agg["per_case"]["core-b"]["k"] == 1
    assert agg["per_case"]["core-b"]["n"] == 3


def test_aggregate_repetitions_reports_overall_per_repetition_and_range(
    three_run_ids: list[str],
) -> None:
    _write_eval_results(three_run_ids[0], [_case("a", True), _case("b", True)], {})
    _write_eval_results(three_run_ids[1], [_case("a", True), _case("b", False)], {})
    _write_eval_results(three_run_ids[2], [_case("a", True), _case("b", True)], {})

    agg = repeat_summary.aggregate_repetitions(three_run_ids)

    assert agg["overall_per_rep"] == [(2, 2), (1, 2), (2, 2)]
    assert agg["overall_mean"] == pytest.approx(5 / 3)
    assert agg["overall_range"] == (50.0, 100.0)


def test_aggregate_repetitions_flags_a_varying_denominator_explicitly(
    three_run_ids: list[str],
) -> None:
    _write_eval_results(three_run_ids[0], [_case("a", True), _case("b", True)], {})
    _write_eval_results(three_run_ids[1], [_case("a", True)], {})
    _write_eval_results(three_run_ids[2], [_case("a", True), _case("b", True)], {})

    agg = repeat_summary.aggregate_repetitions(three_run_ids)

    assert agg["denominators"] == [2, 1, 2]
    assert agg["denominator_varies"] is True


def test_aggregate_repetitions_counts_an_errored_metric_as_a_failure_and_flags_it(
    three_run_ids: list[str],
) -> None:
    _write_eval_results(three_run_ids[0], [_case("a", True, errored=True)], {})
    _write_eval_results(three_run_ids[1], [_case("a", True)], {})
    _write_eval_results(three_run_ids[2], [_case("a", True)], {})

    agg = repeat_summary.aggregate_repetitions(three_run_ids)

    # errored case counts as a failure for pass/fail purposes...
    assert agg["per_case"]["a"]["k"] == 2
    # ...but is separately, explicitly flagged, not silently folded in.
    assert agg["errors"] == [
        {"run_id": three_run_ids[0], "case": "a", "metric": "Answer Relevancy"}
    ]


def test_aggregate_repetitions_includes_component_cases(
    three_run_ids: list[str],
) -> None:
    for run_id in three_run_ids:
        _write_eval_results(
            run_id,
            [_case("core-a", True)],
            {"tests/test_tools.py": [_case("test_supervisor_save", True)]},
        )

    agg = repeat_summary.aggregate_repetitions(three_run_ids)

    assert "test_supervisor_save" in agg["per_case"]
    assert agg["stable_pass"] == ["core-a", "test_supervisor_save"]


def test_render_repeat_summary_markdown_matches_the_specs_own_example_shape(
    three_run_ids: list[str],
) -> None:
    _write_eval_results(three_run_ids[0], [_case("a", True), _case("b", True)], {})
    _write_eval_results(three_run_ids[1], [_case("a", True), _case("b", False)], {})
    _write_eval_results(three_run_ids[2], [_case("a", True), _case("b", True)], {})
    agg = repeat_summary.aggregate_repetitions(three_run_ids)

    markdown = repeat_summary.render_repeat_summary_markdown(agg)

    assert "Overall: 2/2, 1/2, 2/2" in markdown
    assert "mean 1.7/2" in markdown
    assert "range 50.0-100.0%" in markdown
    assert "Flaky" in markdown
    assert "b" in markdown
    assert "no confidence intervals" in markdown.lower() or "n=3" in markdown


def test_main_writes_repeat_summary_md(three_run_ids: list[str]) -> None:
    _write_eval_results(three_run_ids[0], [_case("a", True)], {})
    _write_eval_results(three_run_ids[1], [_case("a", True)], {})
    _write_eval_results(three_run_ids[2], [_case("a", True)], {})

    try:
        written_to = repeat_summary.main(three_run_ids)
        assert written_to.is_file()
        assert written_to.name == "repeat-summary.md"
        assert "Overall: 1/1, 1/1, 1/1" in written_to.read_text(encoding="utf-8")
    finally:
        default_path = paths.resolve("runs") / "repeat-summary.md"
        if default_path.is_file():
            default_path.unlink()
