"""Tool-correctness tests -- R3a/R3b/R3c (`docs/task-hl11.md`, section 3).

`tools_called` is read from the run's own offline span dump, never from a
mock. This needs one caveat for `save_report` specifically: `TracingMiddleware`
sits **outside** `SaveReportGuardMiddleware`/`SaveReportVerdictGuardMiddleware`
in the middleware stack (`supervisor.py`'s `_supervisor_middleware`), so a
`tool.save_report` span is created whether the call is later refused by a
guard or actually executes -- measured live this stage: a real run recorded
two `tool.save_report` spans, the first `ERROR: save_report call refused --
the verdict is not APPROVE...`, the second `Report saved to: ...`. A span's
presence alone is therefore not proof the file was written; `test_supervisor_save`
does not rely on it being one -- it separately confirms the *message
history*'s last critique was APPROVE (`_assert_approved_before_saving`)
before trusting that a subsequent `save_report` call was the real one.

`ToolCorrectnessMetric` is deterministic here -- no `available_tools` is
ever passed, so it makes no judge call at all and the whole cost of this
file is the live agent runs themselves. Its exact scoring is pinned offline
by `tests/test_tool_correctness_offline.py`; this file spends money only on
producing real runs to score.

The `threshold=0.5` below is the brief's own literal, and for
`test_planner_tools` it is load-bearing semantics rather than a tuning
knob: the score is the matched fraction of `expected_tools`, so 0.5 against
a two-tool expectation is exactly the brief's "web_search та/або
knowledge_search". **It must not be raised.** The project's rule that
thresholds move upward from a measured baseline applies to the judged
metrics, not to this number -- raising it here would silently turn an "or"
into an "and".
"""

from __future__ import annotations

import pytest
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.metrics.base_metric import BaseMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import ToolCall
from langchain_core.messages import ToolMessage

import tools
from config import Settings
from evals.deepeval_model import OpenRouterModel
from evals.runner import build_llm_test_case
from schemas import RESEARCH_INPUT_TEMPLATE
from tests.conftest import golden_input, run_judged_case, skip_without_index
from tests.live_agents import AGENT_SPAN_NAME, run_agent_live
from tests.live_supervisor import run_supervisor_live

pytestmark = pytest.mark.eval

_APPROVED_VERDICT_PREFIX = "**Verdict:** APPROVE"


def _expect(name: str) -> ToolCall:
    """One expected tool, by name alone -- no case here checks arguments.

    `input_parameters` carries a default, but pydantic's mypy plugin treats
    an aliased field as required, so it is passed explicitly.
    """
    return ToolCall(name=name, input_parameters=None)


def _tool_correctness_metric(
    model: DeepEvalBaseLLM, *, ordering: bool = False
) -> ToolCorrectnessMetric:
    """Built inside a test body, never at module scope: every deepeval
    metric raises at construction without an explicit `model=`, which would
    abort gate-tier collection of this file."""
    return ToolCorrectnessMetric(
        threshold=0.5,
        model=model,
        should_consider_ordering=ordering,
    )


def test_planner_tools(
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    """R3a -- a Planner given a research question reaches for search."""
    skip_without_index()
    case_id = "core-single-vs-multi-agent"
    agent_input = golden_input(case_id)
    live = run_agent_live("planner", agent_input, settings=live_settings)

    case = build_llm_test_case(
        live.spans,
        input=agent_input,
        actual_output=live.output,
        agent_span_name=AGENT_SPAN_NAME["planner"],
        expected_tools=[_expect("web_search"), _expect("knowledge_search")],
        name="test_planner_tools",
    )
    metrics: list[BaseMetric] = [_tool_correctness_metric(component_judge_model)]
    run_judged_case(
        eval_run_id,
        case_id="test_planner_tools",
        test_case=case,
        metrics=metrics,
        judge=component_judge_model,
        spans=live.spans,
    )


def _expected_tools_from_sources(sources: list[str]) -> list[ToolCall]:
    """Map a plan's own `sources_to_check` onto the tools that serve them.

    `sources_to_check` is free text by design -- its description invites the
    model to answer "both" -- so this is a substring mapping, and it
    **raises** when nothing matches rather than returning an empty list. An
    empty expectation scores 0.0, i.e. a red test reading "the Researcher
    called the wrong tools", when the truth would be "the Planner named its
    sources in words this mapping does not recognise". Failing here keeps
    the diagnosis attached to the actual defect.
    """
    expected: list[ToolCall] = []
    joined = " ".join(sources).lower()
    if "web" in joined:
        expected.append(_expect("web_search"))
    if "knowledge" in joined:
        expected.append(_expect("knowledge_search"))
    if not expected:
        raise AssertionError(
            f"no source in {sources!r} maps to a known tool -- the mapping, "
            "not the Researcher, is what failed here"
        )
    return expected


def test_researcher_tools(
    live_settings: Settings,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    """R3b -- a Researcher uses the tools its plan's `sources_to_check` names.

    The plan is produced live rather than hand-written, so what this
    measures is agreement between two real agents, not agreement between
    one real agent and a fixture's guess about the other.
    """
    skip_without_index()
    case_id = "edge-mixed-corpus-and-web"
    request = golden_input(case_id)

    plan_run = run_agent_live("planner", request, settings=live_settings)
    plan = plan_run.plan
    assert plan is not None, f"{case_id}: planner returned no structured plan"
    expected_tools = _expected_tools_from_sources(plan.sources_to_check)

    rendered_input = RESEARCH_INPUT_TEMPLATE.format(
        request=request, task=plan_run.output
    )
    live = run_agent_live("researcher", rendered_input, settings=live_settings)

    case = build_llm_test_case(
        live.spans,
        input=rendered_input,
        actual_output=live.output,
        agent_span_name=AGENT_SPAN_NAME["researcher"],
        expected_tools=expected_tools,
        name="test_researcher_tools",
    )
    metrics: list[BaseMetric] = [_tool_correctness_metric(component_judge_model)]
    run_judged_case(
        eval_run_id,
        case_id="test_researcher_tools",
        test_case=case,
        metrics=metrics,
        judge=component_judge_model,
        spans=live.spans,
    )


def _assert_approved_before_saving(messages: list[ToolMessage | object]) -> None:
    """The Critic really approved, rather than the revision budget running out.

    `SaveReportVerdictGuardMiddleware` deliberately lets `save_report`
    through once no revision rounds remain, even without an APPROVE -- the
    escape valve that stops a stuck loop. A test that only checked "the tool
    ran" would pass on that path while claiming to show the brief's
    scenario, so the verdict itself is checked.

    Read from the message history, not the span dump: `critique` returns a
    `Command`, and the tracing middleware records `tool.result` only for a
    `ToolMessage`, so no `tool.critique` span carries the verdict at all.
    """
    critiques = [
        message
        for message in messages
        if isinstance(message, ToolMessage) and message.name == "critique"
    ]
    assert critiques, (
        "no critique ToolMessage in the run -- save_report cannot have been "
        "reached via an APPROVE verdict, whatever the tool-correctness score says"
    )
    verdict_text = str(critiques[-1].content)
    assert verdict_text.startswith(_APPROVED_VERDICT_PREFIX), (
        "save_report ran without an APPROVE verdict -- the revision-budget "
        "escape valve fired, not the mechanism R3c is about. Last critique "
        f"began: {verdict_text[:80]!r}"
    )


def test_supervisor_save(
    live_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
    eval_run_id: str,
    component_judge_model: OpenRouterModel,
) -> None:
    """R3c -- an approved run ends by calling `save_report`, in that order.

    `save_report` resolves its own settings at call time, so redirecting its
    output directory means monkeypatching `tools.load_settings` -- without
    it this live run writes a real report into the project's tracked
    `output/`.
    """
    skip_without_index()
    output_dir = tmp_path_factory.mktemp("eval-output")
    redirected = live_settings.model_copy(update={"output_dir": str(output_dir)})
    monkeypatch.setattr(tools, "load_settings", lambda: redirected)

    request = golden_input("core-single-vs-multi-agent")
    live = run_supervisor_live(request, settings=live_settings)

    _assert_approved_before_saving(list(live.messages))

    case = build_llm_test_case(
        live.spans,
        input=request,
        actual_output=live.output,
        expected_tools=[_expect("critique"), _expect("save_report")],
        name="test_supervisor_save",
    )
    metrics: list[BaseMetric] = [
        _tool_correctness_metric(component_judge_model, ordering=True)
    ]
    run_judged_case(
        eval_run_id,
        case_id="test_supervisor_save",
        test_case=case,
        metrics=metrics,
        judge=component_judge_model,
        spans=live.spans,
    )
