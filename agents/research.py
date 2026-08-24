"""Research sub-agent: executes a plan with web, page-read and
knowledge-base tools.

Bound to `web_search`, `read_url` and `knowledge_search`. Returns free-text
findings as structured Markdown with inline citations -- it does not save
anything; only the Supervisor's `save_report` tool, gated by human approval,
writes to disk. Stateless per invocation: no checkpointer, one human
message in, one findings message out.

See `agents/planner.py` for the shared dependency-inversion rationale that
also governs `system_prompt` here.
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

RESEARCHER_ALLOWLIST: tuple[str, ...] = ("web_search", "read_url", "knowledge_search")


def create_research_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]],
    system_prompt: str,
) -> CompiledStateGraph[
    AgentState[None], None, InputAgentState, OutputAgentState[None]
]:
    """Build the Research sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Must name exactly `RESEARCHER_ALLOWLIST`.
    model : BaseChatModel
        Built by the caller. Tests inject a scripted fake instead.
    middleware : Sequence of AgentMiddleware
        The full stack, e.g. `middleware.agent_middleware(...)` with
        `ReadUrlCapMiddleware` appended by the caller.
    system_prompt : str
        Resolved by the caller via `prompt_store.PromptStore.get(
        prompts.PROMPT_NAMES["researcher"], label=...)`.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the findings as free-text
        Markdown in the last message -- there is no `response_format`.
        Never constructed with a checkpointer.
    """
    assert_allowlist(tools, RESEARCHER_ALLOWLIST, "researcher")

    graph = create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        middleware=list(middleware),
    )
    return graph.with_config(
        metadata={"agent": "researcher"}, tags=["agent:researcher"]
    )
