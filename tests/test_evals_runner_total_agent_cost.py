"""`total_agent_cost` -- sums `gen_ai.usage.cost_usd` across a run's own
spans.

Hand-built span trees, matching `tests/test_evals_runner_retrieval_context.py`'s
own pattern -- no live call, no real dump needed.
"""

from __future__ import annotations

from evals.runner import RunSpans, total_agent_cost


def _span(span_id: str, *, cost_usd: float | None = None) -> dict[str, object]:
    attributes: dict[str, object] = {}
    if cost_usd is not None:
        attributes["gen_ai.usage.cost_usd"] = cost_usd
    return {
        "trace_id": "t1",
        "span_id": span_id,
        "parent_span_id": None,
        "name": "model.planner",
        "start_time": 0,
        "end_time": 1,
        "status": "OK",
        "attributes": attributes,
    }


def test_sums_cost_across_every_span_that_carries_it() -> None:
    run = RunSpans(
        run_id="r1",
        spans=[
            _span("s1", cost_usd=0.001),
            _span("s2", cost_usd=0.002),
            _span("s3", cost_usd=0.0005),
        ],
    )

    assert total_agent_cost(run) == 0.0035


def test_treats_a_missing_attribute_as_zero_rather_than_raising() -> None:
    run = RunSpans(
        run_id="r1",
        spans=[
            _span("s1", cost_usd=0.001),
            _span("s2"),  # a tool span, no gen_ai.usage.cost_usd at all
        ],
    )

    assert total_agent_cost(run) == 0.001


def test_empty_run_costs_nothing() -> None:
    assert total_agent_cost(RunSpans(run_id="r1", spans=[])) == 0.0
