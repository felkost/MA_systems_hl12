"""`orchestrator.py`: the explicit `StateGraph` coordination path
(`docs/specs/stage-4.md`).

Nodes: `planner` -> **scope gate** -> `researcher` -> `critic` ->
(revision loop back to `researcher`, or) `composer` -> `hitl_gate` -> `save`.

**D4.4 -- two nodes the agent-as-tool path gets for free that this path
must build explicitly.** The scope gate exists because this path has no
model reading the Planner's prompt-level out-of-scope rule (`_S1` rule 1a)
-- without it, an out-of-scope request would be researched and a report
saved about it. The composer exists because nobody here writes the final
Markdown otherwise: on the agent-as-tool path the Supervisor model composes
it; this path's Critic node ends the loop with a verdict, not a report.

**D4.4 -- `revision_round` is incremented inside the critic node's return
value, never inside the router.** `add_conditional_edges`'s `path` callable
returns a destination name and cannot write state (installed
`langgraph.graph.state`) -- the increment has to happen in the node that
runs before the router reads it.

**D4.5/D4.16 -- the HITL gate is manual, not `HumanInTheLoopMiddleware`.**
This path never touches `create_agent`'s middleware system, so `hitl_gate`
calls `langgraph.types.interrupt` directly, on a payload built by
`hitl.build_interrupt_request` -- the same shape, same
`allowed_decisions=["approve", "reject"]`, that `HumanInTheLoopMiddleware`
itself builds for `save_report` on the supervisor path (D4.1), so one REPL
loop (`main.py`) can resolve either path's interrupt without knowing which
path raised it.

**D4.20 -- the fixed Planner -> Researcher -> Critic edges are a declared
exception to CLAUDE.md's "no hardcoded tool-call order" ban.** That ban is
scoped to an agent factory, where the model decides the sequence; this
module is not one, and its determinism is exactly the property being
compared against the agent-as-tool path.

**D4.5 -- the revision cap, independent of the supervisor path's.**
`revision_round` starts at 0, increments only on the REVISE back-edge, and
the router's guard is `revision_round < max_revisions + 1` --
`revision_round` counts *completed* REVISE calls, so after N of them it
equals N, and a further (N+1)-th call is the same boundary case the
supervisor path allows: `max_revisions + 1` total critique calls
(`middleware.RevisionCapMiddleware`'s emitted count). `< max_revisions`
under-counts by exactly one call; caught by a scripted test, not by
re-reading the arithmetic a second time. Enforced by an entirely separate
counter from the supervisor path's, on purpose (CLAUDE.md's invariant).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command, interrupt
from opentelemetry import trace

import hitl
import tools
from agents.critic import create_critic_agent
from agents.planner import PLANNER_TOOL_CALL_LIMIT, create_planner_agent
from agents.research import create_research_agent
from config import Settings
from grounding import UnsupportedClaimMiddleware
from middleware import (
    CriticVerificationMiddleware,
    ReadUrlCapMiddleware,
    TracingMiddleware,
    agent_middleware,
)
from prompts import build_composer_prompt
from schemas import (
    RESEARCH_INPUT_TEMPLATE,
    CritiqueResult,
    ReportDraft,
    ResearchPlan,
    render_critique,
    render_plan,
)


class OrchestratorState(TypedDict):
    """This path's own state -- `revision_round` lives here, not on
    `SupervisorState` (D4.2's own note on why not)."""

    messages: Annotated[list[BaseMessage], add_messages]
    revision_round: NotRequired[int]
    plan: NotRequired[ResearchPlan | None]
    latest_findings: NotRequired[str | None]
    latest_critique: NotRequired[CritiqueResult | None]
    report_draft: NotRequired[ReportDraft | None]


def _original_request(messages: Sequence[Any]) -> str:
    """The literal text of the first `HumanMessage` -- same helper shape as
    `supervisor.py`'s, kept as a separate copy rather than a shared import:
    the two coordination paths must never import each other or a module
    that would couple them (`tests/test_layering.py`'s `FORBIDDEN_PAIRS`).
    """
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("no HumanMessage in state -- main.py always seeds one")


def create_orchestrator_graph(
    settings: Settings,
    *,
    role_models: Mapping[str, BaseChatModel],
    base_tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[OrchestratorState, None, Any, Any]:
    """Build the graph-path coordinator.

    Parameters
    ----------
    settings : Settings
    role_models : Mapping of str to BaseChatModel
        Same shape as `supervisor.create_supervisor`'s -- the composer node
        reuses the `"supervisor"` role's model (`Settings.ROLES` has no
        `"composer"` entry; the composer replaces work the Supervisor model
        does on the other path).
    base_tools : Sequence of BaseTool, optional
        Defaults to `tools.SUPERVISOR_TOOLS`; `save_report` is looked up by
        name so a test can inject a fake one.
    checkpointer : BaseCheckpointSaver, optional
        Only this compiled graph gets one (D3.6, D4.10); every sub-agent
        stays uncheckpointed.

    Returns
    -------
    CompiledStateGraph
    """
    if base_tools is None:
        base_tools = tools.SUPERVISOR_TOOLS
    save_report_tool = next(t for t in base_tools if t.name == "save_report")

    planner_graph = create_planner_agent(
        settings,
        tools.PLANNER_TOOLS,
        model=role_models["planner"],
        middleware=agent_middleware(
            tool_call_limit=PLANNER_TOOL_CALL_LIMIT, role="planner"
        ),
    )
    research_graph = create_research_agent(
        settings,
        tools.RESEARCHER_TOOLS,
        model=role_models["researcher"],
        middleware=[
            *agent_middleware(
                tool_call_limit=settings.researcher_max_tool_calls, role="researcher"
            ),
            ReadUrlCapMiddleware(settings.max_read_url_per_search),
            UnsupportedClaimMiddleware(),
        ],
    )
    critic_graph = create_critic_agent(
        settings,
        tools.CRITIC_TOOLS,
        model=role_models["critic"],
        middleware=[
            *agent_middleware(
                tool_call_limit=settings.critic_max_tool_calls, role="critic"
            ),
            CriticVerificationMiddleware(),
        ],
    )
    composer_graph = create_agent(
        model=role_models["supervisor"],
        tools=[],
        system_prompt=build_composer_prompt(settings.composer_prompt_version),
        response_format=ProviderStrategy(ReportDraft, strict=True),
        # No agent_middleware() stack -- the composer has no tools, so the
        # tool-call-limit/retry machinery has nothing to guard. TracingMiddleware
        # alone still gets its model-call span (D5.9/D5.10's own gap, found
        # during implementation: composer_graph previously had no middleware
        # list at all, so its model calls would have been invisible to the
        # "every agent call, not just the top-level" requirement the plan's
        # own Langfuse-rules table states).
        middleware=[TracingMiddleware(role="composer")],
    )

    def planner_node(state: OrchestratorState) -> dict[str, Any]:
        original = _original_request(state["messages"])
        with trace.get_tracer(__name__).start_as_current_span("agent.planner"):
            result = planner_graph.invoke(
                {"messages": [HumanMessage(content=original)]}
            )
        return {
            "plan": cast(ResearchPlan, result["structured_response"]),
            "revision_round": 0,
        }

    def route_after_planner(
        state: OrchestratorState,
    ) -> Literal["researcher", "out_of_scope"]:
        plan = state["plan"]
        assert plan is not None
        return "researcher" if plan.in_scope else "out_of_scope"

    def out_of_scope_node(state: OrchestratorState) -> dict[str, Any]:
        plan = state["plan"]
        assert plan is not None
        return {"messages": [AIMessage(content=render_plan(plan))]}

    def researcher_node(state: OrchestratorState) -> dict[str, Any]:
        critique = state.get("latest_critique")
        if critique is None:
            plan = state["plan"]
            assert plan is not None
            task = render_plan(plan)
        else:
            task = "Revision feedback: " + "; ".join(critique.revision_requests)
        rendered_input = RESEARCH_INPUT_TEMPLATE.format(
            request=_original_request(state["messages"]), task=task
        )
        with trace.get_tracer(__name__).start_as_current_span("agent.researcher"):
            result = research_graph.invoke(
                {"messages": [HumanMessage(content=rendered_input)]}
            )
        return {"latest_findings": str(result["messages"][-1].content)}

    def critic_node(state: OrchestratorState) -> dict[str, Any]:
        findings = state["latest_findings"]
        assert findings is not None
        rendered_input = RESEARCH_INPUT_TEMPLATE.format(
            request=_original_request(state["messages"]), task=findings
        )
        with trace.get_tracer(__name__).start_as_current_span("agent.critic"):
            result = critic_graph.invoke(
                {"messages": [HumanMessage(content=rendered_input)]}
            )
        critique = cast(CritiqueResult, result["structured_response"])
        update: dict[str, Any] = {"latest_critique": critique}
        if critique.verdict != "APPROVE":
            update["revision_round"] = state.get("revision_round", 0) + 1
        return update

    def route_after_critique(
        state: OrchestratorState,
    ) -> Literal["researcher", "composer"]:
        critique = state["latest_critique"]
        assert critique is not None
        if critique.verdict == "APPROVE":
            return "composer"
        # `revision_round` counts completed REVISE calls, so after N of them
        # it equals N; a further (N+1)-th call is the boundary case the
        # supervisor path's cap also allows (`max_revisions + 1` total
        # critique calls). `< max_revisions + 1` is the correct guard --
        # `< max_revisions` under-counts by exactly one call, found by a
        # scripted test, not by re-reading this arithmetic.
        if state.get("revision_round", 0) < settings.max_revisions + 1:
            return "researcher"
        # Cap reached with a standing REVISE: still composer, not a dead
        # end -- the run must produce something (D4.18's supervisor-path
        # twin: a cap-exhausted run may still save).
        return "composer"

    def composer_node(state: OrchestratorState) -> dict[str, Any]:
        findings = state["latest_findings"] or ""
        critique = state["latest_critique"]
        assert critique is not None
        cap_exhausted_on_revise = (
            critique.verdict != "APPROVE"
            and state.get("revision_round", 0) >= settings.max_revisions
        )
        task_lines = [
            f"Original request: {_original_request(state['messages'])}",
            f"Findings:\n{findings}",
            f"Critic verdict:\n{render_critique(critique)}",
        ]
        if cap_exhausted_on_revise:
            task_lines.append(
                "Note: the revision cap was reached before an APPROVE verdict. "
                "Compose the report from what exists and note this in one "
                "sentence at the end, followed by the Critic's standing gaps."
            )
        with trace.get_tracer(__name__).start_as_current_span("agent.composer"):
            result = composer_graph.invoke(
                {"messages": [HumanMessage(content="\n\n".join(task_lines))]}
            )
        draft = cast(ReportDraft, result["structured_response"])
        return {"report_draft": draft}

    def hitl_gate_node(state: OrchestratorState) -> Command[Any]:
        draft = state["report_draft"]
        assert draft is not None
        request = hitl.build_interrupt_request(
            hitl.action_requests_from_tool_calls(
                [
                    {
                        "name": "save_report",
                        "args": {"filename": draft.filename, "content": draft.content},
                    }
                ]
            )
        )
        response = interrupt(request)
        decision = response["decisions"][0]
        if decision["type"] == "approve":
            return Command(goto="save")
        return Command(
            goto=END,
            update={
                "messages": [
                    AIMessage(content="[system] No report was saved this turn.")
                ]
            },
        )

    def save_node(state: OrchestratorState) -> dict[str, Any]:
        draft = state["report_draft"]
        assert draft is not None
        with trace.get_tracer(__name__).start_as_current_span("tool.save_report"):
            result = save_report_tool.invoke(
                {"filename": draft.filename, "content": draft.content}
            )
        return {"messages": [AIMessage(content=str(result))]}

    graph = StateGraph(OrchestratorState)
    graph.add_node("planner", planner_node)
    graph.add_node("out_of_scope", out_of_scope_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("critic", critic_node)
    graph.add_node("composer", composer_node)
    graph.add_node("hitl_gate", hitl_gate_node, destinations=("save", END))
    graph.add_node("save", save_node)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"researcher": "researcher", "out_of_scope": "out_of_scope"},
    )
    graph.add_edge("out_of_scope", END)
    graph.add_edge("researcher", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critique,
        {"researcher": "researcher", "composer": "composer"},
    )
    graph.add_edge("composer", "hitl_gate")
    graph.add_edge("save", END)

    return graph.compile(checkpointer=checkpointer)
