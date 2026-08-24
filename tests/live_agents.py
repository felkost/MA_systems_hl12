"""Runs one real sub-agent, live, for the eval-tier component tests
(stage 7, `docs/specs/stage-7.md`).

Not a test module itself (no `test_*` name, so pytest never collects it) --
a shared helper `tests/test_planner.py`/`test_researcher.py`/`test_critic.py`
call from inside `@pytest.mark.eval` test bodies.

`AGENT_SPAN_NAME`'s values must match the literal `agent.<role>` spans
`supervisor.py`/`orchestrator.py` actually open -- pinned against drift by
`tests/test_agent_span_names.py`. This module opens the same span itself
(D7.2): `agent.*` spans exist nowhere inside `agents/*.py`, so a helper that
built a sub-agent and invoked it directly, without opening this span, would
produce a dump `retrieval_context_for_agent` can never match anything
against -- indistinguishable from "the model never called
knowledge_search" unless this is done deliberately.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator, cast
from uuid import uuid4

from langchain_core.messages import HumanMessage
from opentelemetry import baggage, context, trace

import observability
from agents.critic import create_critic_agent
from agents.planner import PLANNER_TOOL_CALL_LIMIT, create_planner_agent
from agents.research import create_research_agent
from config import Settings
from evals.runner import RunSpans, load_run
from middleware import (
    CriticVerificationMiddleware,
    ReadUrlCapMiddleware,
    agent_middleware,
)
from models import build_chat_model
from schemas import CritiqueResult, ResearchPlan, render_critique, render_plan
from tools import CRITIC_TOOLS, PLANNER_TOOLS, RESEARCHER_TOOLS

AGENT_SPAN_NAME: dict[str, str] = {
    "planner": "agent.planner",
    "researcher": "agent.researcher",
    "critic": "agent.critic",
}


@dataclass(frozen=True)
class LiveRun:
    """One live sub-agent invocation's rendered output and full span dump.

    `plan` is set only for the planner role. A caller that needs the plan's
    own fields -- stage 8's R3b derives its expected tool set from
    `sources_to_check` -- would otherwise have to parse them back out of the
    markdown `render_plan` produced, which would make a rendering change
    silently break an unrelated test.
    """

    run_id: str
    output: str
    spans: RunSpans
    plan: ResearchPlan | None = None


@contextmanager
def configured_for_eval(settings: Settings) -> Iterator[None]:
    """Configure observability exactly once for this process.

    `configure_observability` raises `RuntimeError` on a second call
    (`observability.py`'s `_CONFIGURED` sentinel) -- callers (the
    `live_settings` fixture in each eval-tier test module) must ensure this
    context manager wraps the whole test session, not one call each.
    """
    handle = observability.configure_observability(settings)
    try:
        yield None
    finally:
        handle.shutdown()


def eval_settings(*, runs_dir: str, settings: Settings) -> Settings:
    """Return `settings` with its span dump redirected to `runs_dir`.

    Never writes into the project's real `runs/` -- the exact pollution
    `insights.md` records as a caught stage-5 gate-test mistake, avoided
    here for a live eval-tier run the same way.
    """
    return settings.model_copy(update={"span_dump_dir": runs_dir})


def run_agent_live(role: str, agent_input: str, *, settings: Settings) -> LiveRun:
    """Build and invoke one real sub-agent, under a fresh `run_id`.

    Parameters
    ----------
    role : str
        One of `AGENT_SPAN_NAME` -- `"planner"`, `"researcher"`, `"critic"`.
    agent_input : str
        The exact text sent as the sub-agent's one `HumanMessage`. Callers
        render it through `schemas.RESEARCH_INPUT_TEMPLATE` first when they
        want the production-shaped input (D7.13) -- this function sends
        whatever string it is given, unmodified.
    settings : Settings
        Must already have `span_dump_dir` pointed off the project's real
        `runs/` (see `eval_settings`) and observability already configured
        for this process (see `configured_for_eval`).

    Returns
    -------
    LiveRun
    """
    model = build_chat_model(settings, role)
    run_id = str(uuid4())
    plan: ResearchPlan | None = None
    token = context.attach(baggage.set_baggage("run_id", run_id))
    try:
        with trace.get_tracer(__name__).start_as_current_span(
            AGENT_SPAN_NAME[role], attributes={"run_id": run_id}
        ):
            graph: Any
            if role == "planner":
                graph = create_planner_agent(
                    settings,
                    PLANNER_TOOLS,
                    model=model,
                    middleware=agent_middleware(
                        tool_call_limit=PLANNER_TOOL_CALL_LIMIT, role="planner"
                    ),
                )
                result = graph.invoke({"messages": [HumanMessage(content=agent_input)]})
                plan = cast(ResearchPlan, result["structured_response"])
                output = render_plan(plan)

            elif role == "researcher":
                graph = create_research_agent(
                    settings,
                    RESEARCHER_TOOLS,
                    model=model,
                    middleware=[
                        *agent_middleware(
                            tool_call_limit=settings.researcher_max_tool_calls,
                            role="researcher",
                        ),
                        ReadUrlCapMiddleware(settings.max_read_url_per_search),
                    ],
                )
                result = graph.invoke({"messages": [HumanMessage(content=agent_input)]})
                output = str(result["messages"][-1].content)
            elif role == "critic":
                graph = create_critic_agent(
                    settings,
                    CRITIC_TOOLS,
                    model=model,
                    middleware=[
                        *agent_middleware(
                            tool_call_limit=settings.critic_max_tool_calls,
                            role="critic",
                        ),
                        CriticVerificationMiddleware(),
                    ],
                )
                result = graph.invoke({"messages": [HumanMessage(content=agent_input)]})
                critique = cast(CritiqueResult, result["structured_response"])
                output = render_critique(critique)
            else:
                raise ValueError(
                    f"unknown role {role!r}, expected one of {AGENT_SPAN_NAME}"
                )
    finally:
        context.detach(token)

    spans = load_run(run_id, runs_dir=settings.span_dump_dir or "runs")
    return LiveRun(run_id=run_id, output=output, spans=spans, plan=plan)
