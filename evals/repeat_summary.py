"""Stage 9e phase 6's n=3 report: per-case pass frequency, three buckets,
and Overall as a range -- never one number (`docs/specs/stage-9e.md`,
D9e.11, go/no-go 7).

Reads `n` repetitions' own already-written `runs/<eval_run_id>/eval-results.json`
(`evals.summarize_e2e.main`'s own output, one per repetition of the same
two-invocation checkpoint) and joins them by case name. A single run's own
`Overall: N/M passed` (`evals.summarize_e2e.render_summary_markdown`) is the
right report for *one* run; averaging three of those into one number would
hide exactly the run-to-run variance this phase exists to measure, per this
project's own "never report the best of n runs" rule.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import paths


def load_eval_results(eval_run_id: str) -> dict[str, Any]:
    """Read `runs/<eval_run_id>/eval-results.json`
    (`evals.summarize_e2e.main`'s own persisted output).

    Raises
    ------
    FileNotFoundError
        No such file -- named with the exact command needed.
    """
    path = paths.run_dir(eval_run_id) / "eval-results.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist -- run `python -m evals.summarize_e2e "
            f"{eval_run_id}` (or `evals.aggregate_runs.main`) first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _case_passed(case: dict[str, Any]) -> bool:
    """A case passes only if every one of its metrics succeeded. An ERRORed
    metric (`score` absent, per `evals.summarize_e2e`'s own convention) is
    never `success: True` in DeepEval's own persisted case, so it already
    counts as a failure here -- callers additionally surface it separately
    via `aggregate_repetitions`'s own `errors` list, per D9e.11's "an
    ERRORED metric counted as a failure and reported separately"."""
    return all(metric["success"] for metric in case["metricsData"])


def _named_cases(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Every case in one repetition's `eval-results.json`, e2e and
    component alike, keyed by its own `name` -- e2e case ids
    (`core-agent-persona`) and component test names
    (`test_plan_quality[core-agent-persona]`) never collide."""
    named: dict[str, dict[str, Any]] = {}
    for case in result.get("cases", []):
        named[case["name"]] = case
    for cases in result.get("component_cases", {}).values():
        for case in cases:
            named[case["name"]] = case
    return named


def aggregate_repetitions(eval_run_ids: list[str]) -> dict[str, Any]:
    """Join `n` repetitions' own `eval-results.json` into the n=3 report's
    own shape.

    Parameters
    ----------
    eval_run_ids : list of str
        In repetition order -- `overall_per_rep`/`denominators` preserve
        that order.

    Returns
    -------
    dict
        `n`: repetition count.
        `per_case`: `{name: {"k": passes, "n": repetitions it appeared in}}`
          -- `n` here is **per case**, not the repetition count, since a
          case missing from one repetition (a crashed invocation, an
          INCONCLUSIVE skip) must not be silently treated as a failure for
          a repetition it was never scored in.
        `stable_pass` / `flaky` / `stable_fail`: case names, sorted.
        `overall_per_rep`: `[(passed, total), ...]`, one pair per repetition.
        `overall_mean`: mean passed count across repetitions (not a rate).
        `overall_range`: `(min_pct, max_pct)` pass-rate percentages.
        `denominators`: each repetition's own total scored-case count.
        `denominator_varies`: `True` when `denominators` are not all equal
          -- D9e.11's own "a varying denominator (26 or 27) handled
          explicitly", never assumed constant.
        `errors`: `[{"run_id", "case", "metric"}, ...]` for every ERRORed
          metric found, in repetition order.
    """
    per_case: dict[str, dict[str, int]] = {}
    overall_per_rep: list[tuple[int, int]] = []
    denominators: list[int] = []
    errors: list[dict[str, str]] = []

    for run_id in eval_run_ids:
        result = load_eval_results(run_id)
        named = _named_cases(result)
        denominators.append(len(named))

        passed_this_rep = 0
        for name, case in named.items():
            entry = per_case.setdefault(name, {"k": 0, "n": 0})
            entry["n"] += 1
            if _case_passed(case):
                entry["k"] += 1
                passed_this_rep += 1
            for metric in case["metricsData"]:
                if metric.get("score") is None:
                    errors.append(
                        {"run_id": run_id, "case": name, "metric": metric["name"]}
                    )
        overall_per_rep.append((passed_this_rep, len(named)))

    stable_pass = sorted(name for name, e in per_case.items() if e["k"] == e["n"])
    stable_fail = sorted(name for name, e in per_case.items() if e["k"] == 0)
    flaky = sorted(name for name, e in per_case.items() if 0 < e["k"] < e["n"])

    pass_rates = [
        (passed / total * 100) if total else 0.0 for passed, total in overall_per_rep
    ]
    overall_mean = (
        sum(passed for passed, _ in overall_per_rep) / len(overall_per_rep)
        if overall_per_rep
        else 0.0
    )

    return {
        "n": len(eval_run_ids),
        "per_case": per_case,
        "stable_pass": stable_pass,
        "flaky": flaky,
        "stable_fail": stable_fail,
        "overall_per_rep": overall_per_rep,
        "overall_mean": overall_mean,
        "overall_range": (
            (min(pass_rates), max(pass_rates)) if pass_rates else (0.0, 0.0)
        ),
        "denominators": denominators,
        "denominator_varies": len(set(denominators)) > 1,
        "errors": errors,
    }


def render_repeat_summary_markdown(agg: dict[str, Any]) -> str:
    """`docs/specs/stage-9e.md`'s own worked example shape, verbatim:
    `Overall: 24/26, 25/26, 25/26 — mean 24.7/26 (94.9%), range 92.3-96.2%`.
    """
    lines: list[str] = []
    lines.append(f"# n={agg['n']} repeat summary")
    lines.append("")

    overall_parts = ", ".join(f"{p}/{t}" for p, t in agg["overall_per_rep"])
    mean_total = agg["denominators"][0] if agg["denominators"] else 0
    mean_pct = (agg["overall_mean"] / mean_total * 100) if mean_total else 0.0
    low, high = agg["overall_range"]
    lines.append(
        f"Overall: {overall_parts} — mean {agg['overall_mean']:.1f}/{mean_total} "
        f"({mean_pct:.1f}%), range {low:.1f}-{high:.1f}%"
    )
    lines.append(
        "No confidence intervals at n=3 — range only, per this project's own "
        "standing rule against treating a 3-point bootstrap as meaningful."
    )
    lines.append("")

    if agg["denominator_varies"]:
        lines.append(
            f"**Denominator varies across repetitions**: {agg['denominators']} "
            "-- not assumed constant; see the per-case table for which case(s) "
            "this affects."
        )
        lines.append("")

    lines.append(f"## Stable pass ({len(agg['stable_pass'])})")
    lines.extend(f"- {name}" for name in agg["stable_pass"])
    lines.append("")

    lines.append(f"## Flaky ({len(agg['flaky'])})")
    for name in agg["flaky"]:
        entry = agg["per_case"][name]
        lines.append(f"- {name}: {entry['k']}/{entry['n']}")
    lines.append("")

    lines.append(f"## Stable fail ({len(agg['stable_fail'])})")
    lines.extend(f"- {name}" for name in agg["stable_fail"])
    lines.append("")

    lines.append(f"## Errors ({len(agg['errors'])})")
    lines.append(
        "Counted as a failure in the buckets above; listed here separately "
        "so an infrastructure error never hides inside a quality number."
    )
    for error in agg["errors"]:
        lines.append(f"- {error['case']} ({error['metric']}), run {error['run_id']}")
    lines.append("")

    return "\n".join(lines) + "\n"


def main(eval_run_ids: list[str], *, out_path: str | Path | None = None) -> Path:
    """Aggregate `eval_run_ids` and write `repeat-summary.md`.

    Parameters
    ----------
    eval_run_ids : list of str
    out_path : str or Path, optional
        Defaults to `runs/repeat-summary.md` -- this artefact aggregates
        several run directories at once, so it is not naturally owned by
        any single one of them.

    Returns
    -------
    Path
        Where `repeat-summary.md` was written.
    """
    agg = aggregate_repetitions(eval_run_ids)
    markdown = render_repeat_summary_markdown(agg)
    destination = (
        Path(out_path)
        if out_path is not None
        else paths.resolve("runs") / "repeat-summary.md"
    )
    destination.write_text(markdown, encoding="utf-8")
    return destination


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        raise SystemExit(
            "usage: python -m evals.repeat_summary <eval_run_id_1> <eval_run_id_2> "
            "<eval_run_id_3> [...]"
        )
    written_to = main(sys.argv[1:])
    print(f"wrote {written_to}")
