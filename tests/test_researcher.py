"""Researcher component tests -- GEval `Groundedness` (R2c,
`docs/task-hl11.md`).

**This metric measures corpus-groundedness, not groundedness in general**
(`docs/specs/stage-7.md`, D7.14): `retrieval.chunks` is set at exactly one
site, `tools.py`'s `knowledge_search` (`tools.py:277-283`). The Researcher
also has `web_search` and `read_url`; nothing captures what those return in
a form this metric can see. A claim drawn from the web is therefore counted
ungrounded **by construction**, per the brief's own step 3 ("Claims not
present in retrieval context count as ungrounded, even if true"). The
baseline this stage publishes is reported with that sentence attached, not
as unqualified "Groundedness".

`retrieval_context` comes from `evals.runner.retrieval_context_for_agent`,
ancestor-scoped to `agent.researcher` -- a flat filter on
`tool.knowledge_search` would mix in chunks the Planner or Critic retrieved
in the same run (measured on two real runs, `docs/specs/stage-7.md` D7.2).

Live input is rendered through `schemas.RESEARCH_INPUT_TEMPLATE`, the same
template both coordinators use before invoking the Researcher
(`supervisor.py:188`, `orchestrator.py`) -- not the raw golden `input`,
because that template is itself a project invariant (CLAUDE.md: "the
original user request is forwarded by code, not by prompt").
"""

from __future__ import annotations

import pytest
from deepeval.metrics import GEval
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.test_case.llm_test_case import RetrievedContextData

from config import Settings
from evals.deepeval_model import OpenRouterModel
from evals.runner import retrieval_context_for_agent
from schemas import RESEARCH_INPUT_TEMPLATE
from tests.conftest import golden_input, run_judged_case, skip_without_index
from tests.live_agents import AGENT_SPAN_NAME, run_agent_live

pytestmark = pytest.mark.eval


def _groundedness_metric(model: DeepEvalBaseLLM) -> GEval:
    return GEval(
        name="Groundedness",
        evaluation_steps=[
            "Extract every factual claim from 'actual output'",
            "For each claim, check if it can be directly supported by 'retrieval "
            "context'",
            "Claims not present in retrieval context count as ungrounded, even if true",
            "Score = number of grounded claims / total claims",
        ],
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
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
    request = golden_input(case_id)
    rendered_input = RESEARCH_INPUT_TEMPLATE.format(request=request, task=request)
    live = run_agent_live("researcher", rendered_input, settings=live_settings)

    retrieval_context = retrieval_context_for_agent(
        live.spans, AGENT_SPAN_NAME["researcher"]
    )
    assert retrieval_context, (
        f"{case_id}: researcher's run produced no retrieval_context -- either "
        "the model never called knowledge_search, or the corpus does not "
        "cover this question. Groundedness is unmeasurable without it "
        "(docs/specs/stage-7.md, Known risks)."
    )

    # `input` is required by LLMTestCase but not in Groundedness's own
    # evaluation_params (measured: brief's own sample omits INPUT for this
    # metric) -- left empty rather than filled with a value the metric
    # never reads. `retrieval_context`'s declared type is invariant in its
    # element type (mypy), so it is rebound through an explicitly-typed
    # local rather than passed as the narrower `list[str]` directly.
    typed_retrieval_context: list[str | RetrievedContextData] = list(retrieval_context)
    case = LLMTestCase(
        name=f"{test_name}[{case_id}]",
        input="",
        actual_output=live.output,
        retrieval_context=typed_retrieval_context,
    )
    assert case.actual_output, f"{case_id}: researcher produced no output"
    metrics: list[BaseMetric] = [_groundedness_metric(component_judge_model)]
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
def test_research_grounded(
    case_id: str,
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    _run_case(
        case_id,
        live_settings,
        test_name="test_research_grounded",
        eval_run_id=eval_run_id,
        component_judge_model=component_judge_model,
    )


@pytest.mark.parametrize("case_id", ["edge-narrow-memory-question"])
def test_research_edge_case(
    case_id: str,
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    _run_case(
        case_id,
        live_settings,
        test_name="test_research_edge_case",
        eval_run_id=eval_run_id,
        component_judge_model=component_judge_model,
    )
