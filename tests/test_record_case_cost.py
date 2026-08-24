"""`record_case_cost`/`load_case_costs` round trip (stage 9a, D9a.8).

Offline: writes into a real `runs/<eval_run_id>/` (via `paths.run_dir`),
using a throwaway uuid so it never collides with a real eval run, and
cleans up after itself -- `runs/` is gitignored, but a stray directory left
behind by a test is exactly the "runs/ pollution" mistake insights.md
already records once (stage 5).
"""

from __future__ import annotations

import json
import shutil
from uuid import uuid4

import paths
from evals.runner import RunSpans
from tests.conftest import load_case_costs, persist_case_spans, record_case_cost


def test_records_round_trip_in_write_order() -> None:
    eval_run_id = str(uuid4())
    try:
        record_case_cost(
            eval_run_id,
            case_id="core-a",
            run_id="run-1",
            agent_cost_usd=0.01,
            judge_cost_usd=0.02,
        )
        record_case_cost(
            eval_run_id,
            case_id="core-b",
            run_id="run-2",
            agent_cost_usd=0.03,
            judge_cost_usd=0.04,
        )

        records = load_case_costs(eval_run_id)

        assert records == [
            {
                "case_id": "core-a",
                "run_id": "run-1",
                "agent_cost_usd": 0.01,
                "judge_cost_usd": 0.02,
            },
            {
                "case_id": "core-b",
                "run_id": "run-2",
                "agent_cost_usd": 0.03,
                "judge_cost_usd": 0.04,
            },
        ]
    finally:
        shutil.rmtree(paths.run_dir(eval_run_id), ignore_errors=True)


def test_load_case_costs_returns_empty_list_for_an_unknown_run() -> None:
    assert load_case_costs(str(uuid4())) == []


def test_persist_case_spans_writes_the_case_scoped_dump() -> None:
    """D9e.2: three prior stages each lost a diagnosis to the
    `tmp_path_factory` cleanup that ends every live-eval session -- this is
    what keeps one case's own spans past that point."""
    eval_run_id = str(uuid4())
    spans = RunSpans(run_id="run-1", spans=[{"name": "model.researcher"}])
    try:
        persist_case_spans(eval_run_id, case_id="core-a", spans=spans)

        path = paths.run_dir(eval_run_id) / "spans" / "core-a.json"
        assert json.loads(path.read_text(encoding="utf-8")) == spans.spans
    finally:
        shutil.rmtree(paths.run_dir(eval_run_id), ignore_errors=True)
