"""Planner sub-agent: turns a user request into a structured `ResearchPlan`.

Bound to `web_search` and `knowledge_search` only -- reconnaissance to
understand the domain, not a deep read of any one source. Stateless per
invocation: no checkpointer, one human message in, one `ResearchPlan`
out.

`model`, `tools` and `middleware` all arrive as parameters, and
none of the three is imported from `models.py`/`tools.py`/`middleware.py` --
`agents.*` is domain and all three are infra.

`system_prompt` is likewise an injected parameter: this factory never
imports `prompts.build_planner_prompt` -- `prompts.py` holds no prompt text
at all, only a name/label registry (`prompts.PROMPT_NAMES`) that
`supervisor.py`/`orchestrator.py` resolve through `prompt_store.PromptStore`
before calling this factory. Same rule as `model`/`tools`/`middleware`,
just one more infra-sourced value the caller supplies.
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
    system_prompt: str,
) -> CompiledStateGraph[
    AgentState[ResearchPlan], None, InputAgentState, OutputAgentState[ResearchPlan]
]:
    """Build the Planner sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Must name exactly `PLANNER_ALLOWLIST` -- checked before any model
        call.
    model : BaseChatModel
        Built by the caller, e.g. `models.build_chat_model(settings,
        "planner")`. Tests inject a scripted fake instead.
    middleware : Sequence of AgentMiddleware
        The full stack, e.g. `middleware.agent_middleware(tool_call_limit=
        PLANNER_TOOL_CALL_LIMIT)`. This factory never assembles its own --
        that would import `middleware.py`, which is infra.
    system_prompt : str
        Resolved by the caller via `prompt_store.PromptStore.get(
        prompts.PROMPT_NAMES["planner"], label=...)` -- this
        factory never imports `prompts.py` or `prompt_store.py` itself.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the parsed `ResearchPlan` in
        `result["structured_response"]`. Never constructed with a
        checkpointer -- only the Supervisor gets one.
    """
    assert_allowlist(tools, PLANNER_ALLOWLIST, "planner")

    graph = create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        response_format=PLANNER_RESPONSE_FORMAT,
        middleware=list(middleware),
    )
    # Read by telemetry off on_tool_start's own metadata -- confirmed to
    # survive nested invocation, so every tool call this graph's own ReAct
    # loop makes attributes to "planner" regardless of which module invokes
    # it.
    return graph.with_config(metadata={"agent": "planner"}, tags=["agent:planner"])
