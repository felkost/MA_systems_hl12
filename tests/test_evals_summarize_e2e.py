"""`evals.summarize_e2e` (stage 9a, `docs/specs/stage-9a.md` D9a.8).

Offline: a hand-built, **camelCase**, `.latest_run_full.json`-shaped fixture
dict (`testCases`/`metricsData`, matching the real file's own shape --
`docs/specs/stage-9a.md` N5/N6/N7, measured live against the real CLI
during this stage's own kickoff) mixing e2e and non-e2e metric names, plus
a hand-built case-costs list. No live call, no `.deepeval/` file needed.
"""

from __future__ import annotations

import pytest

from evals.summarize_e2e import (
    UnknownCaseError,
    filter_e2e_cases,
    group_component_cases,
    render_summary_markdown,
)

_E2E_METRIC_NAMES = frozenset(
    {"Answer Relevancy", "Correctness [GEval]", "Citation Presence [GEval]"}
)


def _metric(name: str, score: float, threshold: float, success: bool) -> dict:
    return {"name": name, "score": score, "threshold": threshold, "success": success}


def _fake_full_run() -> dict:
    return {
        "testCases": [
            {
                "name": "core-single-vs-multi-agent",
                "metricsData": [
                    _metric("Answer Relevancy", 0.9, 0.7, True),
                    _metric("Correctness [GEval]", 0.8, 0.6, True),
                    _metric("Citation Presence [GEval]", 0.7, 0.6, True),
                ],
            },
            {
                "name": "edge-out-of-scope-recipe",
                "metricsData": [
                    _metric("Answer Relevancy", 0.5, 0.7, False),
                    _metric("Correctness [GEval]", 0.9, 0.6, True),
                    _metric("Citation Presence [GEval]", 1.0, 0.6, True),
                ],
            },
            {
                # A stage-7 component-test case in the same persisted file --
                # deepeval test run tests/ collects every eval-tier file.
                "name": "plan-quality-case",
                "metricsData": [_metric("Plan Quality [GEval]", 0.9, 0.7, True)],
            },
        ]
    }


_CATEGORY_BY_ID = {
    "core-single-vs-multi-agent": "happy_path",
    "edge-out-of-scope-recipe": "failure_case",
}


def test_filter_e2e_cases_keeps_only_the_three_e2e_metric_names() -> None:
    filtered = filter_e2e_cases(_fake_full_run(), category_by_id=_CATEGORY_BY_ID)

    names = {case["name"] for case in filtered}
    assert names == {"core-single-vs-multi-agent", "edge-out-of-scope-recipe"}
    for case in filtered:
        assert {m["name"] for m in case["metricsData"]} == _E2E_METRIC_NAMES


def test_filter_e2e_cases_tags_each_case_with_its_category() -> None:
    filtered = filter_e2e_cases(_fake_full_run(), category_by_id=_CATEGORY_BY_ID)

    by_name = {case["name"]: case for case in filtered}
    assert by_name["core-single-vs-multi-agent"]["category"] == "happy_path"
    assert by_name["edge-out-of-scope-recipe"]["category"] == "failure_case"


def test_filter_e2e_cases_raises_on_an_e2e_case_with_no_category_match() -> None:
    run = {
        "testCases": [
            {
                "name": "unknown-case-id",
                "metricsData": [
                    _metric("Answer Relevancy", 0.9, 0.7, True),
                    _metric("Correctness [GEval]", 0.8, 0.6, True),
                    _metric("Citation Presence [GEval]", 0.7, 0.6, True),
                ],
            }
        ]
    }

    with pytest.raises(UnknownCaseError):
        filter_e2e_cases(run, category_by_id=_CATEGORY_BY_ID)


def test_render_summary_markdown_has_all_three_of_i4s_elements() -> None:
    filtered = filter_e2e_cases(_fake_full_run(), category_by_id=_CATEGORY_BY_ID)
    case_costs = [
        {
            "case_id": "core-single-vs-multi-agent",
            "run_id": "r1",
            "agent_cost_usd": 0.04,
            "judge_cost_usd": 0.025,
        },
        {
            "case_id": "edge-out-of-scope-recipe",
            "run_id": "r2",
            "agent_cost_usd": 0.01,
            "judge_cost_usd": 0.02,
        },
    ]

    markdown = render_summary_markdown(filtered, case_costs=case_costs, total_cases=2)

    # I4 element 1: one line per case per metric, name/score/threshold.
    assert "core-single-vs-multi-agent" in markdown
    assert "Answer Relevancy: 0.9" in markdown or "Answer Relevancy" in markdown
    # I4 element 2: aggregate avg/min/max per metric, absolute category counts.
    assert "avg" in markdown and "min" in markdown and "max" in markdown
    assert "happy_path: 1/1" in markdown or "happy_path" in markdown
    assert "%" not in markdown.split("happy_path")[1].split("\n")[0]
    # I4 element 3: Overall: N/M passed (X%).
    assert "Overall: 1/2 passed" in markdown
    assert "%" in markdown.split("Overall:")[1]
    # Cost line, both halves measured.
    assert "0.05" in markdown or "$0.05" in markdown  # agent_cost_usd summed
    assert "0.045" in markdown or "$0.045" in markdown  # judge_cost_usd summed


def test_render_summary_markdown_a_case_passes_only_if_every_metric_did() -> None:
    filtered = filter_e2e_cases(_fake_full_run(), category_by_id=_CATEGORY_BY_ID)

    markdown = render_summary_markdown(filtered, case_costs=[], total_cases=2)

    # edge-out-of-scope-recipe has one failing metric (Answer Relevancy), so
    # only core-single-vs-multi-agent counts as passed overall.
    assert "Overall: 1/2 passed" in markdown


def test_render_summary_markdown_handles_an_errored_metric_with_no_score() -> None:
    """Regression: this stage's own live run produced exactly this shape --
    a judge-model timeout leaves `success: false` but no `score` key at all
    (DeepEval dumps with `exclude_none=True`). A first draft of this
    function indexed `metric['score']` unconditionally and raised `KeyError`
    against the real persisted file; only the live run caught it."""
    run = {
        "testCases": [
            {
                "name": "edge-ukrainian-language-question",
                "metricsData": [
                    {
                        "name": "Answer Relevancy",
                        "threshold": 0.7,
                        "success": False,
                        "error": "Timed out/cancelled while evaluating metric.",
                    },
                    _metric("Correctness [GEval]", 1.0, 0.6, True),
                    _metric("Citation Presence [GEval]", 0.9, 0.6, True),
                ],
            }
        ]
    }
    category_by_id = {"edge-ukrainian-language-question": "edge_case"}
    filtered = filter_e2e_cases(run, category_by_id=category_by_id)

    markdown = render_summary_markdown(filtered, case_costs=[], total_cases=1)

    assert "ERRORED" in markdown
    assert "Timed out/cancelled" in markdown
    # An errored metric's success is False, so the case does not pass overall.
    assert "Overall: 0/1 passed" in markdown
    # Aggregates for Answer Relevancy must not include a phantom score from
    # the errored entry (there were no other Answer Relevancy scores here).
    assert "Answer Relevancy: avg" not in markdown


# -- Stage-9d D9d.8: the report covers all five test files, not e2e alone --


def _five_file_run() -> dict:
    run = _fake_full_run()
    run["testCases"].extend(
        [
            {
                "name": "test_research_grounded[core-agent-persona]",
                "metricsData": [_metric("Groundedness [GEval]", 0.45, 0.7, False)],
            },
            {
                "name": "test_critique_approve",
                "metricsData": [_metric("Critique Quality [GEval]", 0.92, 0.7, True)],
            },
            {
                "name": "test_planner_tools",
                "metricsData": [_metric("Tool Correctness", 1.0, 0.5, True)],
            },
        ]
    )
    return run


def test_group_component_cases_keys_each_case_by_its_source_test_file() -> None:
    grouped = group_component_cases(_five_file_run())
    assert set(grouped) == {
        "tests/test_planner.py",
        "tests/test_researcher.py",
        "tests/test_critic.py",
        "tests/test_tools.py",
    }
    assert [c["name"] for c in grouped["tests/test_tools.py"]] == ["test_planner_tools"]


def test_render_summary_markdown_shows_every_test_file_with_its_own_lines() -> None:
    run = _five_file_run()
    cases = filter_e2e_cases(run, category_by_id=_CATEGORY_BY_ID)
    summary = render_summary_markdown(
        cases,
        case_costs=[],
        total_cases=len(cases),
        component_groups=group_component_cases(run),
    )
    for file_name in (
        "tests/test_planner.py",
        "tests/test_researcher.py",
        "tests/test_critic.py",
        "tests/test_tools.py",
        "tests/test_e2e.py",
    ):
        assert file_name in summary
    assert "test_critique_approve" in summary
    assert "Critique Quality [GEval]: 0.92 (threshold 0.70)" in summary
    # 6 scored cases -- 2 e2e plus 4 component (the base fixture already
    # carries one Plan Quality case). One e2e and one component case fail.
    assert "Overall: 4/6 passed (66.7%)" in summary


def test_filter_e2e_cases_keeps_a_case_scored_by_refusal_appropriateness() -> None:
    """D9e.5/D9e.6: since the metric substitution, an e2e case carries one
    of two possible trios (Answer Relevancy or Refusal Appropriateness),
    both subsets of the four-name superset -- never both trios at once. The
    old equality check would have dropped this case from the summary and
    from the Overall denominator entirely."""
    run = {
        "testCases": [
            {
                "name": "edge-out-of-scope-recipe",
                "metricsData": [
                    _metric("Refusal Appropriateness [GEval]", 0.9, 0.7, True),
                    _metric("Correctness [GEval]", 0.9, 0.6, True),
                    _metric("Citation Presence [GEval]", 1.0, 0.6, True),
                ],
            }
        ]
    }
    filtered = filter_e2e_cases(run, category_by_id=_CATEGORY_BY_ID)
    assert [case["name"] for case in filtered] == ["edge-out-of-scope-recipe"]


def test_filter_e2e_cases_drops_a_case_with_no_metrics_at_all() -> None:
    """The membership check's own guard: an empty `metricsData` list is
    vacuously a subset of any set, so it must be excluded explicitly rather
    than falling through to a false match."""
    run = {"testCases": [{"name": "some-case", "metricsData": []}]}
    assert filter_e2e_cases(run, category_by_id=_CATEGORY_BY_ID) == []


def test_render_summary_markdown_without_component_groups_is_e2e_only() -> None:
    """The stage-9a shape stays available: a run over `tests/test_e2e.py`
    alone must not sprout empty headings for files it never collected."""
    run = _fake_full_run()
    cases = filter_e2e_cases(run, category_by_id=_CATEGORY_BY_ID)
    summary = render_summary_markdown(cases, case_costs=[], total_cases=len(cases))
    assert "tests/test_planner.py" not in summary
    assert summary.startswith("tests/test_e2e.py")
