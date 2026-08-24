"""Research sub-agent: executes a plan with web, page-read and
knowledge-base tools.

Bound to `web_search`, `read_url` and `knowledge_search`. Returns free-text
findings as structured Markdown with inline citations -- it does not save
anything; only the Supervisor's `save_report` tool, gated by human approval,
writes to disk. Stateless per invocation: no checkpointer (D3.6), one human
message in, one findings message out.

Ported from `MA_systems_hl10`; see `agents/planner.py` for the shared D3.2
dependency-inversion rationale.
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
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph.state import CompiledStateGraph

from agents._allowlist import assert_allowlist
from config import Settings
from prompts import build_researcher_prompt

RESEARCHER_ALLOWLIST: tuple[str, ...] = ("web_search", "read_url", "knowledge_search")


def create_research_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]],
) -> CompiledStateGraph[
    AgentState[None], None, InputAgentState, OutputAgentState[None]
]:
    """Build the Research sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Must name exactly `RESEARCHER_ALLOWLIST` (D3.5).
    model : BaseChatModel
        Built by the caller. Tests inject a scripted fake instead.
    middleware : Sequence of AgentMiddleware
        The full stack, e.g. `middleware.agent_middleware(...)` with
        `ReadUrlCapMiddleware` appended by the caller.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the findings as free-text
        Markdown in the last message -- there is no `response_format`.
        Never constructed with a checkpointer (D3.6).
    """
    assert_allowlist(tools, RESEARCHER_ALLOWLIST, "researcher")

    graph = create_agent(
        model=model,
        tools=list(tools),
        system_prompt=build_researcher_prompt(settings.researcher_prompt_version),
        middleware=list(middleware),
    )
    return graph.with_config(
        metadata={"agent": "researcher"}, tags=["agent:researcher"]
    )
