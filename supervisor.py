"""The Supervisor: agent-as-tool coordination over Planner, Researcher and
Critic, with `save_report` gated by a human (`docs/specs/stage-4.md`).

**D4.3 -- every wrapper forwards the original request in code.** `plan`,
`research` and `critique` all read the first `HumanMessage` out of state and
render it through `RESEARCH_INPUT_TEMPLATE`, never leaving that to the
Supervisor's own paraphrase -- CLAUDE.md's invariant, measured in hl8 as a
coordinator dropping context in five of six paraphrases when this was
prompt-level instead. `plan` and `critique` return a `Command` carrying both
a rendered `ToolMessage` and a state write (D4.2); `research` returns its
sub-agent's final message content directly, since it carries no state to
write.

**D4.3 -- a fresh `thread_id` per sub-agent call, not a stripped config.**
Forwarding the ambient `RunnableConfig` verbatim hands each sub-agent the
Supervisor's own checkpointer; stripping `configurable` does not prevent
this (measured against the installed langgraph -- an empty `configurable`
does not trigger `ensure_config`'s ambient-config drop). A fresh
`configurable={"thread_id": uuid4()}` is what actually keeps D3.6 true at
runtime.

**D4.15 -- the middleware order.** `HumanInTheLoopMiddleware` is first in
the list, `agent_middleware(..., tool_exit_behavior="end")` next, then the
two Supervisor-only middlewares from stage 3, then this stage's revision cap
and verdict guard. See `middleware.agent_middleware`'s own docstring for why
`"end"` and this order, both measured rather than assumed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any, NotRequired, cast
from uuid import uuid4

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.agents.middleware.human_in_the_loop import InterruptOnConfig
from langchain.agents.middleware.types import (
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from opentelemetry import trace

import tools
from agents.critic import create_critic_agent
from agents.planner import PLANNER_TOOL_CALL_LIMIT, create_planner_agent
from agents.research import create_research_agent
from config import Settings
from grounding import UnsupportedClaimMiddleware
from hitl import ALLOWED_DECISIONS
from middleware import (
    CriticVerificationMiddleware,
    ReadUrlCapMiddleware,
    RevisionCapMiddleware,
    RoundStabilityMiddleware,
    SaveReportGuardMiddleware,
    SaveReportVerdictGuardMiddleware,
    agent_middleware,
)
from prompts import build_supervisor_prompt
from schemas import (
    RESEARCH_INPUT_TEMPLATE,
    CritiqueResult,
    ResearchPlan,
    render_critique,
    render_plan,
)


class SupervisorState(AgentState[None]):
    """`AgentState` plus what D4.2/D4.5/D4.18 read and write.

    No `revision_round` -- that field belongs to `orchestrator.py`'s state
    only. This path counts rounds from `messages`
    (`middleware.emitted_critique_count`), so a second counter here would
    be two lifecycles for the same thing on one path.
    """

    plan: NotRequired[ResearchPlan | None]
    critic_gaps: NotRequired[list[str] | None]
    previous_critic_gaps: NotRequired[list[str] | None]
    verdict: NotRequired[str | None]


def _original_request(messages: Sequence[Any]) -> str:
    """The literal text of the first `HumanMessage` in state.

    CLAUDE.md's code-not-prompt invariant: every wrapper forwards this in
    code, never relies on the Supervisor's own retelling of it.
    """
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError(
        "no HumanMessage in state -- the Supervisor's own graph invariant "
        "(main.py always seeds the first turn with one)"
    )


def _sanitized_sub_agent_config(config: RunnableConfig) -> RunnableConfig:
    """A `RunnableConfig` safe to hand a sub-agent: same run's callbacks and
    metadata, a fresh thread coordinate.

    Carries `callbacks`/`metadata`/`tags`/`run_name` so the sub-agent's own
    spans still nest under the parent run (measured unaffected by the
    checkpointer change below). `configurable` is replaced, not stripped,
    with a fresh `thread_id`: an *empty* `configurable` does not stop
    langgraph's `ensure_config` from carrying the ambient one forward
    (`docs/specs/stage-4.md`, D4.3) -- only a config that itself supplies a
    checkpoint coordinate does, and only a *different* one keeps the
    Supervisor's checkpointer from being read by the sub-agent's compiled
    graph (which has none of its own, D3.6).
    """
    sanitized: RunnableConfig = {"configurable": {"thread_id": str(uuid4())}}
    if config.get("callbacks") is not None:
        sanitized["callbacks"] = config["callbacks"]
    if config.get("metadata") is not None:
        sanitized["metadata"] = config["metadata"]
    if config.get("tags") is not None:
        sanitized["tags"] = config["tags"]
    if config.get("run_name") is not None:
        sanitized["run_name"] = config["run_name"]
    return sanitized


def _make_plan_tool(
    planner_graph: CompiledStateGraph[Any, Any, Any, Any],
) -> BaseTool:
    @tool
    def plan(
        task: str,
        state: Annotated[SupervisorState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig,
    ) -> Command[Any]:
        """Turn the user's request into a structured research plan.

        Call this first, before research or critique, with the user's
        request restated as the task.
        """
        rendered_input = RESEARCH_INPUT_TEMPLATE.format(
            request=_original_request(state["messages"]), task=task
        )
        with trace.get_tracer(__name__).start_as_current_span("agent.planner"):
            result = planner_graph.invoke(
                {"messages": [HumanMessage(content=rendered_input)]},
                config=_sanitized_sub_agent_config(config),
            )
        research_plan = cast(ResearchPlan, result["structured_response"])
        return Command(
            update={
                "plan": research_plan,
                "messages": [
                    ToolMessage(
                        content=render_plan(research_plan),
                        tool_call_id=tool_call_id,
                        name="plan",
                    )
                ],
            }
        )

    return plan


def _make_research_tool(
    research_graph: CompiledStateGraph[Any, Any, Any, Any],
) -> BaseTool:
    @tool
    def research(
        task: str,
        state: Annotated[SupervisorState, InjectedState],
        config: RunnableConfig,
    ) -> str:
        """Execute research and return findings as Markdown with citations.

        Call with the plan's own text on the first round, or with the
        Critic's revision_requests as the task on a revision round.
        """
        rendered_input = RESEARCH_INPUT_TEMPLATE.format(
            request=_original_request(state["messages"]), task=task
        )
        with trace.get_tracer(__name__).start_as_current_span("agent.researcher"):
            result = research_graph.invoke(
                {"messages": [HumanMessage(content=rendered_input)]},
                config=_sanitized_sub_agent_config(config),
            )
        return str(result["messages"][-1].content)

    return research


def _make_critique_tool(
    critic_graph: CompiledStateGraph[Any, Any, Any, Any],
) -> BaseTool:
    @tool
    def critique(
        task: str,
        state: Annotated[SupervisorState, InjectedState],
        tool_call_id: Annotated[str, InjectedToolCallId],
        config: RunnableConfig,
    ) -> Command[Any]:
        """Get an independent verdict on the current findings.

        Call with the findings text as the task.
        """
        rendered_input = RESEARCH_INPUT_TEMPLATE.format(
            request=_original_request(state["messages"]), task=task
        )
        with trace.get_tracer(__name__).start_as_current_span("agent.critic"):
            result = critic_graph.invoke(
                {"messages": [HumanMessage(content=rendered_input)]},
                config=_sanitized_sub_agent_config(config),
            )
        critique_result = cast(CritiqueResult, result["structured_response"])
        return Command(
            update={
                "verdict": critique_result.verdict,
                "previous_critic_gaps": state.get("critic_gaps"),
                "critic_gaps": critique_result.gaps,
                "messages": [
                    ToolMessage(
                        content=render_critique(critique_result),
                        tool_call_id=tool_call_id,
                        name="critique",
                    )
                ],
            }
        )

    return critique


def _supervisor_middleware(settings: Settings) -> list[Any]:
    """The Supervisor's own middleware list, factored out of
    `create_supervisor` so its assembled order (D4.15) is testable without
    a full `create_agent` build.

    `HumanInTheLoopMiddleware` first, `agent_middleware(...,
    tool_exit_behavior="end")` next, then the two stage-3 Supervisor-only
    middlewares, then this stage's revision cap and verdict guard -- see
    this module's own docstring and `middleware.agent_middleware`'s for why
    this order and why `"end"`, both measured rather than assumed.
    """
    return [
        HumanInTheLoopMiddleware(
            interrupt_on={
                "save_report": InterruptOnConfig(
                    allowed_decisions=list(ALLOWED_DECISIONS)
                ),
            }
        ),
        *agent_middleware(
            tool_call_limit=settings.resolved_supervisor_max_tool_calls(),
            tool_exit_behavior="end",
            role="supervisor",
        ),
        RoundStabilityMiddleware(),
        SaveReportGuardMiddleware(max_revisions=settings.max_revisions),
        RevisionCapMiddleware(max_revisions=settings.max_revisions),
        SaveReportVerdictGuardMiddleware(max_revisions=settings.max_revisions),
    ]


def create_supervisor(
    settings: Settings,
    *,
    role_models: Mapping[str, BaseChatModel],
    base_tools: Sequence[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[
    AgentState[None], None, InputAgentState, OutputAgentState[None]
]:
    """Build the Supervisor.

    Parameters
    ----------
    settings : Settings
    role_models : Mapping of str to BaseChatModel
        Keyed by `Settings.ROLES`, one model per role including
        `"supervisor"` itself -- built by the caller (`main.py`) via
        `models.build_chat_model(settings, role)`. Named `role_models`, not
        `models`, and `base_tools`, not `tools`, so this function's own body
        can still name the `models`/`tools` modules (D4.19's assembly reads
        `tools.SUPERVISOR_TOOLS`, which a parameter named `tools` would
        shadow).
    base_tools : Sequence of BaseTool, optional
        The Supervisor's own non-delegation tools. Defaults to
        `tools.SUPERVISOR_TOOLS` (`[save_report]`); overridable so a test
        can inject a fake `save_report` that never touches disk.
    checkpointer : BaseCheckpointSaver, optional
        Only the Supervisor gets one (D3.6, D4.10 -- typically a
        `MemorySaver`); every sub-agent stays uncheckpointed.

    Returns
    -------
    CompiledStateGraph
        `state_schema=SupervisorState`; tools are
        `[*base_tools, plan, research, critique]` (D4.19), assembled here
        rather than by extending `tools.SUPERVISOR_TOOLS` itself, which is
        infra and cannot hold application-layer wrappers
        (`tests/test_layering.py`).
    """
    if base_tools is None:
        base_tools = tools.SUPERVISOR_TOOLS

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

    supervisor_tools: list[BaseTool] = [
        *base_tools,
        _make_plan_tool(planner_graph),
        _make_research_tool(research_graph),
        _make_critique_tool(critic_graph),
    ]

    graph = create_agent(
        model=role_models["supervisor"],
        tools=supervisor_tools,
        system_prompt=build_supervisor_prompt(settings.supervisor_prompt_version),
        middleware=_supervisor_middleware(settings),
        state_schema=SupervisorState,
        checkpointer=checkpointer,
    )
    return graph.with_config(
        metadata={"agent": "supervisor"}, tags=["agent:supervisor"]
    )
