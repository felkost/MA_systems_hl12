"""Planner sub-agent: turns a user request into a structured `ResearchPlan`.

Bound to `web_search` and `knowledge_search` only -- reconnaissance to
understand the domain, not a deep read of any one source. Stateless per
invocation: no checkpointer (D3.6), one human message in, one `ResearchPlan`
out.

Ported from `MA_systems_hl10`, finishing the dependency inversion hl10
started: `model`, `tools` and `middleware` all arrive as parameters, and
none of the three is imported from `models.py`/`tools.py`/`middleware.py` --
`agents.*` is `DOMAIN` and all three are `INFRA`
(`docs/specs/stage-3.md`, D3.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    InputAgentState,
    OutputAgentState,
)
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agents._allowlist import assert_allowlist
from config import Settings
from prompts import build_planner_prompt
from schemas import ResearchPlan

PLANNER_ALLOWLIST: tuple[str, ...] = ("web_search", "knowledge_search")

# A small, fixed reconnaissance budget -- not a Settings field, since the
# knobs this project varies tune the Researcher and Critic, not how much the
# Planner may look around before it decomposes. Exported so the caller that
# assembles `middleware.agent_middleware(tool_call_limit=...)` uses the same
# number, without a second copy of it.
PLANNER_TOOL_CALL_LIMIT = 4

# Passing the bare schema lets the framework auto-detect a strategy and
# build it *without* `strict`, which leaves the provider free to treat the
# schema's `required` list as a hint and return a plan missing a field.
# Naming the strategy explicitly is what puts `"strict": true` on the wire.
PLANNER_RESPONSE_FORMAT = ProviderStrategy(ResearchPlan, strict=True)


def create_planner_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]],
) -> CompiledStateGraph[
    AgentState[ResearchPlan], None, InputAgentState, OutputAgentState[ResearchPlan]
]:
    """Build the Planner sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Must name exactly `PLANNER_ALLOWLIST` -- checked before any model
        call (D3.5).
    model : BaseChatModel
        Built by the caller, e.g. `models.build_chat_model(settings,
        "planner")`. Tests inject a scripted fake instead.
    middleware : Sequence of AgentMiddleware
        The full stack, e.g. `middleware.agent_middleware(tool_call_limit=
        PLANNER_TOOL_CALL_LIMIT)`. This factory never assembles its own --
        that would import `middleware.py`, which is infra.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the parsed `ResearchPlan` in
        `result["structured_response"]`. Never constructed with a
        checkpointer (D3.6) -- only the Supervisor gets one.
    """
    assert_allowlist(tools, PLANNER_ALLOWLIST, "planner")

    graph = create_agent(
        model=model,
        tools=list(tools),
        system_prompt=build_planner_prompt(settings.planner_prompt_version),
        response_format=PLANNER_RESPONSE_FORMAT,
        middleware=list(middleware),
    )
    # Read by telemetry off on_tool_start's own metadata (stage 5) --
    # confirmed by hl8 to survive nested invocation, so every tool call this
    # graph's own ReAct loop makes attributes to "planner" regardless of
    # which module invokes it.
    return graph.with_config(metadata={"agent": "planner"}, tags=["agent:planner"])
