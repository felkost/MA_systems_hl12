"""Planner component tests -- GEval `Plan Quality` (R2a, `docs/task-hl11.md`).

`_plan_quality_metric` is a factory, not a module-level metric: DeepEval
4.1.10 raises `DeepEvalError` at **construction** for any metric with no
`model=`, which would break gate-tier collection of this file if the metric
were built at import time (measured, `docs/specs/stage-7.md` M1/D7.3).
`tests/test_component_metric_definitions.py` calls this factory offline,
with a fake model, to validate the definition without spending anything.

Live inputs are golden `happy_path` cases (`docs/specs/stage-7.md`, D7.13),
sent to the Planner exactly as production does -- the raw `input` string,
no template (the Supervisor's own `plan` tool forwards the user's message
unmodified, `supervisor.py`'s `_make_plan_tool`).
"""

from __future__ import annotations

import pytest
from deepeval.metrics import GEval
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import Settings
from evals.deepeval_model import OpenRouterModel
from evals.runner import total_agent_cost
from tests.conftest import (
    golden_input,
    persist_case_spans,
    record_case_cost,
    run_judged_case,
    skip_without_index,
)
from tests.live_agents import run_agent_live

pytestmark = pytest.mark.eval


def _plan_quality_metric(model: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Plan Quality",
        evaluation_steps=[
            "Check that the plan contains specific search queries (not vague)",
            "Check that sources_to_check includes relevant sources for the topic",
            "Check that the output_format matches what the user asked for",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=model,
        threshold=0.7,
    )


def _run_case(
    case_id: str,
    live_settings: Settings,
    *,
    test_name: str,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    skip_without_index()
    agent_input = golden_input(case_id)
    live = run_agent_live("planner", agent_input, settings=live_settings)
    # `name` is what `evals/summarize_e2e.py` prints as this line's test id
    # (D9d.8); without it DeepEval's persisted file carries no way back to
    # the test that produced the case.
    case = LLMTestCase(
        name=f"{test_name}[{case_id}]", input=agent_input, actual_output=live.output
    )
    assert case.actual_output, f"{case_id}: planner produced no output"
    metrics: list[BaseMetric] = [_plan_quality_metric(component_judge_model)]
    run_judged_case(
        eval_run_id,
        case_id=f"{test_name}[{case_id}]",
        test_case=case,
        metrics=metrics,
        judge=component_judge_model,
        spans=live.spans,
    )


@pytest.mark.parametrize(
    "case_id", ["core-single-vs-multi-agent", "core-agent-persona"]
)
def test_plan_quality(
    case_id: str,
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    _run_case(
        case_id,
        live_settings,
        test_name="test_plan_quality",
        eval_run_id=eval_run_id,
        component_judge_model=component_judge_model,
    )


@pytest.mark.parametrize(
    "case_id", ["core-tool-calling-role", "core-agent-vs-rag-boundary"]
)
def test_plan_has_queries(
    case_id: str,
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    _run_case(
        case_id,
        live_settings,
        test_name="test_plan_has_queries",
        eval_run_id=eval_run_id,
        component_judge_model=component_judge_model,
    )


def test_planner_recognizes_incoherent_nonsense_as_out_of_scope(
    live_settings: Settings, eval_run_id: str
) -> None:
    """D9e.8 (`p2`) -- deterministic, no judge: `failure-nonsense-query`'s
    own input has no coherent research question inside it, and `p2`'s new
    incoherence criterion says that is out of scope the same way a personal
    request is, not a topic for the Planner to invent a plan around.

    `assert plan.in_scope is False` is the whole check -- no GEval, no
    threshold, at the live agent's own cost alone (~$0.005 per the spec's
    own estimate). Cost and spans are still persisted (`record_case_cost`
    with `judge_cost_usd=0.0`), matching this project's own "no number is
    published without saying what it rests on" for every live call, judged
    or not.
    """
    skip_without_index()
    agent_input = golden_input("failure-nonsense-query")
    live = run_agent_live("planner", agent_input, settings=live_settings)
    try:
        assert live.plan is not None, "failure-nonsense-query: planner returned no plan"
        assert live.plan.in_scope is False, (
            "failure-nonsense-query: p2's incoherence criterion should have "
            f"set in_scope to False; got {live.plan!r}"
        )
    finally:
        record_case_cost(
            eval_run_id,
            case_id="test_planner_recognizes_incoherent_nonsense_as_out_of_scope",
            run_id=live.run_id,
            agent_cost_usd=total_agent_cost(live.spans),
            judge_cost_usd=0.0,
        )
        persist_case_spans(
            eval_run_id,
            case_id="test_planner_recognizes_incoherent_nonsense_as_out_of_scope",
            spans=live.spans,
        )
