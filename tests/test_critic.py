"""Critic component tests -- GEval `Critique Quality` (R2b,
`docs/task-hl11.md`).

Inputs are **committed fixtures**
(`evals/fixtures/findings_strong.md`/`findings_thin.md`), not a live
Researcher run chained into a live Critic run: chaining two live agents
would make a Critic score depend on Researcher variance, and a red result
would not localise to either agent. The fixtures differ in exactly the
properties the brief's own Critique Quality steps ask about -- specific,
cited findings vs. vague, uncited ones -- so the pair exercises both the
APPROVE-leaning and REVISE-leaning paths. The **verdict itself is never
asserted**; only the quality of the critique is scored (`docs/specs/stage-7.md`,
D7.9).

`Settings.critic_prompt_version` defaults to `"c2"`, which is *stricter*
than this exact metric measures: `c2` forbids `APPROVE` alongside any gap
at all, while the brief's own step 3 permits `APPROVE` with "empty or …
only minor" gaps (`insights.md`, stage 3). A systematically lower score on
`findings_strong.md` than an `APPROVE`-permissive prompt would produce is
a plausible, already-predicted outcome of that choice -- reported, not
tuned away.
"""

from __future__ import annotations

import pytest
from deepeval.metrics import GEval
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams

from config import Settings
from evals.deepeval_model import OpenRouterModel
from schemas import RESEARCH_INPUT_TEMPLATE
from tests.conftest import fixture_text, run_judged_case, skip_without_index
from tests.live_agents import run_agent_live

pytestmark = pytest.mark.eval

_ORIGINAL_REQUEST = (
    "What distinguishes single-agent from multi-agent AI architectures, "
    "and when does each tend to work better?"
)


def _critique_quality_metric(model: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Critique Quality",
        evaluation_steps=[
            "Check that the critique identifies specific issues, not vague complaints",
            "Check that revision_requests are actionable (researcher can act on them)",
            "If verdict is APPROVE, gaps list should be empty or contain only minor "
            "items",
            "If verdict is REVISE, there must be at least one revision_request",
        ],
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=model,
        threshold=0.7,
    )


def _run_case(
    fixture_name: str,
    live_settings: Settings,
    *,
    test_name: str,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    skip_without_index()
    findings = fixture_text(fixture_name)
    rendered_input = RESEARCH_INPUT_TEMPLATE.format(
        request=_ORIGINAL_REQUEST, task=findings
    )
    live = run_agent_live("critic", rendered_input, settings=live_settings)
    case = LLMTestCase(name=test_name, input=findings, actual_output=live.output)
    assert case.actual_output, f"{fixture_name}: critic produced no output"
    metrics: list[BaseMetric] = [_critique_quality_metric(component_judge_model)]
    run_judged_case(
        eval_run_id,
        case_id=test_name,
        test_case=case,
        metrics=metrics,
        judge=component_judge_model,
        spans=live.spans,
    )


def test_critique_approve(
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    _run_case(
        "findings_strong.md",
        live_settings,
        test_name="test_critique_approve",
        eval_run_id=eval_run_id,
        component_judge_model=component_judge_model,
    )


def test_critique_revise(
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    _run_case(
        "findings_thin.md",
        live_settings,
        test_name="test_critique_revise",
        eval_run_id=eval_run_id,
        component_judge_model=component_judge_model,
    )
