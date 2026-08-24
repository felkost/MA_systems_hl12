"""Turn one `deepeval test run` invocation's own persisted results, plus
this project's own persisted per-case cost log, into `runs/<id>/summary.md`
and `runs/<id>/eval-results.json` (stage 9a, `docs/specs/stage-9a.md` D9a.8).

Run manually, **after** a `deepeval test run` invocation that included
`tests/test_e2e.py` -- `wrap_up_test_run()`, which (re)writes DeepEval's own
`LATEST_FULL_TEST_RUN_FILE_PATH`, is called from exactly one code path this
project's `assert_test`-based usage ever reaches: the `deepeval test run`
CLI command itself, in-process, right after `pytest.main()` returns (stage
9a's own SDK measurement, N6). A bare `pytest tests/test_e2e.py` never
writes it.

Two data sources, written by two different processes, at two different
times:

- DeepEval's own file -- every case's per-metric score/threshold/success.
- `runs/<eval_run_id>/case-costs.jsonl` -- this project's own per-case
  agent/judge cost, written incrementally by `tests/test_e2e.py` itself
  *during* the pytest session, while the data was still in memory (D9a.7's
  judge-cost accumulator does not survive past the pytest process on its
  own; this file is what carries its result across that boundary).

Neither source is ever written by this module -- it only reads and joins.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from deepeval.test_run.test_run import LATEST_FULL_TEST_RUN_FILE_PATH

import paths

_E2E_METRIC_NAMES = frozenset(
    {
        "Answer Relevancy",
        "Correctness [GEval]",
        "Citation Presence [GEval]",
        # D9e.5: substituted for Answer Relevancy on `expects.out_of_scope`
        # cases, so an e2e case now carries one of two possible trios, both
        # subsets of this four-name superset -- never all four at once.
        "Refusal Appropriateness [GEval]",
    }
)
_GOLDEN_DATASET_PATH = paths.PROJECT_ROOT / "tests" / "golden_dataset.json"

# Which test file a persisted case came from, keyed by the metric that
# scored it. DeepEval's own persisted case carries no source-file field
# (measured, stage 9d), so metric identity is the only join available --
# the same basis `filter_e2e_cases` already uses, extended to the four
# non-e2e files so one `deepeval test run tests/` renders the brief's whole
# five-file report (D9d.8).
_FILE_BY_METRIC: dict[str, str] = {
    "Plan Quality [GEval]": "tests/test_planner.py",
    "Groundedness [GEval]": "tests/test_researcher.py",
    "Critique Quality [GEval]": "tests/test_critic.py",
    "Tool Correctness": "tests/test_tools.py",
}
_E2E_FILE = "tests/test_e2e.py"


class UnknownCaseError(RuntimeError):
    """An e2e-scoped test case's name has no match in the golden dataset --
    fails loudly rather than silently dropping it from the summary (D8.5's
    "fail loudly on an unrecognised mapping" discipline)."""


def load_latest_run(path: Path | None = None) -> dict[str, Any]:
    """Read and JSON-parse a persisted DeepEval full test-run file.

    Parameters
    ----------
    path : Path, optional
        Defaults to DeepEval's own `LATEST_FULL_TEST_RUN_FILE_PATH` -- the
        existing no-argument caller is byte-for-byte unaffected. Stage 9e's
        `evals.aggregate_runs` passes an explicit snapshot path instead
        (D9e.2a): the **second** `deepeval test run` invocation overwrites
        that same file, so anything needing more than one invocation's
        result must read from a snapshot taken before the next run.

    Raises
    ------
    FileNotFoundError
        No such file -- named with the exact command needed, not a bare
        "file not found".
    """
    resolved = (
        path if path is not None else paths.resolve(LATEST_FULL_TEST_RUN_FILE_PATH)
    )
    if not resolved.is_file():
        raise FileNotFoundError(
            f"{resolved} does not exist -- run `deepeval test run tests/` "
            "(with tests/test_e2e.py included) before summarize_e2e"
        )
    return json.loads(resolved.read_text(encoding="utf-8"))


def _category_by_id() -> dict[str, str]:
    cases = json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    return {case["id"]: case["category"] for case in cases}


def filter_e2e_cases(
    run: dict[str, Any], *, category_by_id: dict[str, str]
) -> list[dict[str, Any]]:
    """Keep only `testCases` entries scored by this stage's own three
    metrics, tagged with their golden-dataset category.

    Filters on metric identity (names this stage itself owns), not on the
    case name's own shape -- robust regardless of how many other eval-tier
    files (stage 7/8's component and tool-correctness tests) were collected
    in the same `deepeval test run` invocation.

    Membership, not equality (D9e.6): since D9e.5, an e2e case carries one
    of *two* possible three-metric trios (Answer Relevancy or Refusal
    Appropriateness, never both), both subsets of `_E2E_METRIC_NAMES`'
    four-name superset. An equality check against one fixed set would drop
    every substituted-trio case from the summary **and the Overall
    denominator** -- silently, the same failure shape D9e.2a's snapshot
    mechanism exists to prevent elsewhere. Guarded against an empty
    `metricsData` list, which is vacuously a subset of anything.

    Raises
    ------
    UnknownCaseError
        An e2e-scoped case's `name` has no entry in `category_by_id`.
    """
    filtered: list[dict[str, Any]] = []
    for case in run.get("testCases", []):
        metric_names = {metric["name"] for metric in case.get("metricsData", [])}
        if not metric_names or not metric_names <= _E2E_METRIC_NAMES:
            continue
        category = category_by_id.get(case["name"])
        if category is None:
            raise UnknownCaseError(
                f"e2e test case {case['name']!r} has no matching id in "
                "tests/golden_dataset.json -- the mapping, not the run, is "
                "what failed here"
            )
        filtered.append({**case, "category": category})
    return filtered


def _case_passed(case: dict[str, Any]) -> bool:
    return all(metric["success"] for metric in case["metricsData"])


def group_component_cases(run: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Persisted cases from the four non-e2e test files, keyed by file.

    A case whose metrics match none of `_FILE_BY_METRIC` and are not the
    e2e trio is skipped rather than guessed at -- an unrecognised metric
    means a new test file this renderer has not been taught about, and
    inventing a heading for it would be worse than omitting it.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in run.get("testCases", []):
        metric_names = {metric["name"] for metric in case.get("metricsData", [])}
        if metric_names == _E2E_METRIC_NAMES:
            continue
        files = {
            _FILE_BY_METRIC[name] for name in metric_names if name in _FILE_BY_METRIC
        }
        if len(files) != 1:
            continue
        grouped.setdefault(files.pop(), []).append(case)
    return grouped


def _metric_line(metric: dict[str, Any]) -> str:
    # A metric that errored (e.g. a judge-model timeout) carries no "score"
    # at all -- DeepEval's own file dumps with `exclude_none=True`, so the
    # key is absent, not `null` (measured against a real run, not assumed).
    score = metric.get("score")
    score_text = f"{score:.2f}" if score is not None else "ERRORED"
    error = metric.get("error")
    suffix = f" -- {error}" if error else ""
    return (
        f"     {metric['name']}: {score_text} "
        f"(threshold {metric['threshold']:.2f}){suffix}"
    )


def render_component_sections(grouped: dict[str, list[dict[str, Any]]]) -> list[str]:
    """The brief's own per-file blocks for the four non-e2e test files."""
    lines: list[str] = []
    for file_name in sorted(grouped):
        lines.append(file_name)
        for case in sorted(grouped[file_name], key=lambda c: str(c.get("name") or "")):
            icon = "✅" if _case_passed(case) else "❌"
            lines.append(f"  {icon} {case.get('name') or 'unnamed case'}")
            lines.extend(_metric_line(metric) for metric in case["metricsData"])
        lines.append("")
    return lines


def render_summary_markdown(
    cases: list[dict[str, Any]],
    *,
    case_costs: list[dict[str, Any]],
    total_cases: int,
    component_groups: dict[str, list[dict[str, Any]]] | None = None,
) -> str:
    """The brief's own mockup shape, all three of I4's elements.

    Parameters
    ----------
    cases : list of dict
        `filter_e2e_cases`'s own output.
    case_costs : list of dict
        `record_case_cost`'s own persisted records
        (`case_id`/`run_id`/`agent_cost_usd`/`judge_cost_usd`).
    total_cases : int
        The denominator for `Overall: N/<total_cases> passed` -- ordinarily
        `len(cases)`, passed explicitly rather than re-derived so a caller
        that already knows how many cases were *scored* (as opposed to
        skipped, e.g. the poisoned-knowledge-base case, D9a.12) states it
        plainly instead of this function guessing.
    component_groups : dict, optional
        `group_component_cases`'s output. Supplied when the invocation
        covered the whole suite, so the report shows all five test files
        rather than the e2e one alone (D9d.8).
    """
    lines: list[str] = []
    if component_groups:
        lines.extend(render_component_sections(component_groups))

    lines.append(_E2E_FILE)
    for case in sorted(cases, key=lambda c: str(c["name"])):
        icon = "✅" if _case_passed(case) else "❌"
        lines.append(f"  {icon} test_golden_dataset[{case['name']}]")
        lines.extend(_metric_line(metric) for metric in case["metricsData"])
    lines.append("")

    lines.append("Aggregates:")
    for metric_name in sorted(_E2E_METRIC_NAMES):
        all_metrics = [
            metric
            for case in cases
            for metric in case["metricsData"]
            if metric["name"] == metric_name
        ]
        scores = [m["score"] for m in all_metrics if m.get("score") is not None]
        errored = sum(1 for m in all_metrics if m.get("score") is None)
        if scores:
            errored_note = f", {errored} errored" if errored else ""
            lines.append(
                f"  {metric_name}: avg {sum(scores) / len(scores):.2f}, "
                f"min {min(scores):.2f}, max {max(scores):.2f}{errored_note}"
            )
    lines.append("")

    lines.append("Category breakdown (absolute counts):")
    categories = sorted({case["category"] for case in cases})
    for category in categories:
        in_category = [case for case in cases if case["category"] == category]
        passed = sum(1 for case in in_category if _case_passed(case))
        lines.append(f"  {category}: {passed}/{len(in_category)}")
    lines.append("")

    agent_total = sum(record["agent_cost_usd"] for record in case_costs)
    judge_total = sum(record["judge_cost_usd"] for record in case_costs)
    lines.append(
        f"Cost (both measured, neither estimated): agent ${agent_total:.4f}, "
        f"judge ${judge_total:.4f}"
    )
    lines.append("")

    component_cases = [
        case for group in (component_groups or {}).values() for case in group
    ]
    passed_overall = sum(1 for case in [*cases, *component_cases] if _case_passed(case))
    scored_overall = total_cases + len(component_cases)
    pass_rate = (passed_overall / scored_overall * 100) if scored_overall else 0.0
    lines.append(
        f"Overall: {passed_overall}/{scored_overall} passed ({pass_rate:.1f}%)"
    )

    return "\n".join(lines) + "\n"


def load_case_costs(eval_run_id: str) -> list[dict[str, Any]]:
    """Read back `runs/<eval_run_id>/case-costs.jsonl`
    (`tests.conftest.record_case_cost`'s own output)."""
    path = paths.resolve("runs") / eval_run_id / "case-costs.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _discover_latest_eval_run_id() -> str:
    runs_dir = paths.resolve("runs")
    candidates = sorted(
        runs_dir.glob("*/case-costs.jsonl"), key=lambda p: p.stat().st_mtime
    )
    if not candidates:
        raise FileNotFoundError(
            f"no runs/*/case-costs.jsonl under {runs_dir} -- run `deepeval "
            "test run tests/test_e2e.py` first, or pass eval_run_id explicitly"
        )
    return candidates[-1].parent.name


def main(eval_run_id: str | None = None, *, run: dict[str, Any] | None = None) -> Path:
    """Read both sources, join them, write `eval-results.json` and
    `summary.md` under `runs/<eval_run_id>/`, and return that directory.

    Parameters
    ----------
    eval_run_id : str, optional
        Defaults to the most recently modified `runs/*/case-costs.jsonl`'s
        own directory.
    run : dict, optional
        An already-parsed DeepEval run (stage 9e, D9e.2a) -- when supplied,
        `load_latest_run()` is not called at all. `evals.aggregate_runs`
        passes the in-memory merge of several invocations' own snapshots
        here; the default (`None`) keeps this function's original,
        single-invocation behaviour unchanged.
    """
    if eval_run_id is None:
        eval_run_id = _discover_latest_eval_run_id()

    if run is None:
        run = load_latest_run()
    cases = filter_e2e_cases(run, category_by_id=_category_by_id())
    component_groups = group_component_cases(run)
    case_costs = load_case_costs(eval_run_id)

    run_dir = paths.run_dir(eval_run_id)
    (run_dir / "eval-results.json").write_text(
        json.dumps(
            {
                "cases": cases,
                "component_cases": component_groups,
                "case_costs": case_costs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary = render_summary_markdown(
        cases,
        case_costs=case_costs,
        total_cases=len(cases),
        component_groups=component_groups,
    )
    (run_dir / "summary.md").write_text(summary, encoding="utf-8")
    return run_dir


if __name__ == "__main__":
    import sys

    written_to = main(sys.argv[1] if len(sys.argv) > 1 else None)
    print(f"wrote {written_to / 'summary.md'}")
