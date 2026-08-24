"""Snapshot and merge DeepEval's own persisted run file across more than one
`deepeval test run` invocation (stage 9e, D9e.2a).

The second `deepeval test run` **overwrites**
`.deepeval/.latest_run_full.json` -- stage 9d merged three such files by
hand into one report, and no code in this repository did it. Without this
module the `Overall: N/26` line does not exist at all after the second
invocation, and six of stage 9e's own nine go/no-go criteria assume it does.

No intermediate merged-JSON schema is invented: `merge_runs` concatenates
`testCases` **in memory** and hands the result straight to
`summarize_e2e.main(run=...)`. A second on-disk shape would need its own
doc entry and its own not-found error for no benefit over the two things
that already exist -- `summarize_e2e.load_latest_run` and DeepEval's own
persisted file shape.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from deepeval.test_run.test_run import LATEST_FULL_TEST_RUN_FILE_PATH

import paths
from evals.summarize_e2e import load_latest_run, main as summarize_main


def snapshot_latest_run(eval_run_id: str, label: str) -> Path:
    """Copy DeepEval's own currently-persisted full test-run file to
    `runs/<eval_run_id>/deepeval-run-<label>.json`.

    Run this once, right after each `deepeval test run` invocation, before
    the next invocation overwrites `.deepeval/.latest_run_full.json`.

    Parameters
    ----------
    eval_run_id : str
    label : str
        Names this invocation in the snapshot's own filename, e.g. `"e2e"`
        or `"components"` -- the two invocations D9e.11/D9e.12 name for one
        repetition.

    Returns
    -------
    Path
        The snapshot's own path.

    Raises
    ------
    FileNotFoundError
        `load_latest_run`'s own error -- reused rather than re-implemented,
        so both callers name the same missing-file remedy.
    """
    load_latest_run()  # raises with the right message if nothing was ever run
    source = paths.resolve(LATEST_FULL_TEST_RUN_FILE_PATH)
    destination = paths.run_dir(eval_run_id) / f"deepeval-run-{label}.json"
    shutil.copyfile(source, destination)
    return destination


def collect_invocation_artefacts(
    target_eval_run_id: str, source_eval_run_id: str
) -> int:
    """Fold one invocation's own `case-costs.jsonl` and `spans/` into
    `target_eval_run_id`'s directory (stage 9e phase 1b).

    `snapshot_latest_run` handles DeepEval's own run file, which is the only
    artefact that gets *overwritten* by the next invocation. These two are a
    different problem with the same root: `eval_run_id` is a session-scoped
    pytest fixture minting a fresh `uuid4()` per invocation, with no CLI
    override, so a second `deepeval test run` writes its costs and spans
    under a directory of its own rather than beside the first's. Phase 1b
    bridged that by hand (`cat >>`, `cp`); at n=3 that is six manual steps,
    each one a chance to publish a cost total missing half its rows.

    Idempotent on the spans (a re-copy overwrites the same filenames), but
    **not** on the costs: `case-costs.jsonl` is appended to, so calling this
    twice for the same source double-counts. Callers pass each source id
    exactly once.

    Parameters
    ----------
    target_eval_run_id : str
        Where the merged report is being assembled -- ordinarily the first
        invocation's own id.
    source_eval_run_id : str
        The later invocation's id, whose artefacts are folded in.

    Returns
    -------
    int
        How many span files were copied -- returned rather than logged so a
        caller can assert it against the case count it expected, instead of
        trusting a silent copy.
    """
    target = paths.run_dir(target_eval_run_id)
    source = paths.resolve("runs") / source_eval_run_id

    source_costs = source / "case-costs.jsonl"
    if source_costs.is_file():
        with (target / "case-costs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(source_costs.read_text(encoding="utf-8"))

    copied = 0
    source_spans = source / "spans"
    if source_spans.is_dir():
        target_spans = target / "spans"
        target_spans.mkdir(parents=True, exist_ok=True)
        for span_file in sorted(source_spans.glob("*.json")):
            shutil.copyfile(span_file, target_spans / span_file.name)
            copied += 1
    return copied


def merge_runs(paths_: list[Path]) -> dict[str, Any]:
    """Concatenate `testCases` from several snapshot files into one run.

    Parameters
    ----------
    paths_ : list of Path
        Snapshot files, as written by `snapshot_latest_run`.

    Returns
    -------
    dict
        `{"testCases": [...]}` -- every field `summarize_e2e` reads, from
        every snapshot, in the order given.
    """
    merged: list[dict[str, Any]] = []
    for path in paths_:
        merged.extend(load_latest_run(path).get("testCases", []))
    return {"testCases": merged}


def main(eval_run_id: str, labels: list[str]) -> Path:
    """Merge `runs/<eval_run_id>/deepeval-run-<label>.json` for each label
    in `labels` and render the combined summary via
    `summarize_e2e.main(run=...)`.

    Parameters
    ----------
    eval_run_id : str
    labels : list of str
        Every label previously passed to `snapshot_latest_run` for this
        `eval_run_id`, in invocation order.

    Returns
    -------
    Path
        `summarize_e2e.main`'s own return value -- the run directory
        `summary.md`/`eval-results.json` were written into.
    """
    run_dir = paths.run_dir(eval_run_id)
    snapshot_paths = [run_dir / f"deepeval-run-{label}.json" for label in labels]
    merged = merge_runs(snapshot_paths)
    return summarize_main(eval_run_id, run=merged)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit(
            "usage: python -m evals.aggregate_runs <eval_run_id> <label> "
            "[<label> ...]"
        )
    written_to = main(sys.argv[1], sys.argv[2:])
    print(f"wrote {written_to / 'summary.md'}")
