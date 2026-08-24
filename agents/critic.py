"""Critic sub-agent: independently verifies research findings and returns a
structured verdict.

Bound to `web_search`, `read_url`, `knowledge_search` -- the same three
tools as the Researcher, so the Critic checks facts through the same
sources rather than only reading the Researcher's own text. Response format
is `CritiqueResult`; the caller's `CriticVerificationMiddleware` forces at
least one verification call before a verdict is accepted. Stateless per
invocation: no checkpointer, one human message in, one
`CritiqueResult` out.

See `agents/planner.py` for the shared dependency-inversion rationale that
also governs `system_prompt` here.

`system_prompt` arrives here already compiled with `today` -- this factory
never calls `date.today()` itself: the caller resolves
`prompts.PROMPT_NAMES["critic"]` through `prompt_store.PromptStore.get(...,
variables={"today": ...})`, so the date the Critic sees is a value the
caller supplies, not the system clock read a second time inside a factory
that has no reason to own it.
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
from schemas import CritiqueResult

CRITIC_ALLOWLIST: tuple[str, ...] = ("web_search", "read_url", "knowledge_search")

# Passing the bare schema lets the framework auto-detect a strategy and
# build it *without* strict, which leaves the provider free to omit
# `verdict` -- the one field the whole revision loop reads. Naming the
# strategy explicitly is what puts "strict": true on the wire.
CRITIC_RESPONSE_FORMAT = ProviderStrategy(CritiqueResult, strict=True)


def create_critic_agent(
    settings: Settings,
    tools: Sequence[BaseTool],
    *,
    model: BaseChatModel,
    middleware: Sequence[AgentMiddleware[Any, Any, Any]],
    system_prompt: str,
) -> CompiledStateGraph[
    AgentState[CritiqueResult], None, InputAgentState, OutputAgentState[CritiqueResult]
]:
    """Build the Critic sub-agent.

    Parameters
    ----------
    settings : Settings
    tools : Sequence of BaseTool
        Must name exactly `CRITIC_ALLOWLIST`.
    model : BaseChatModel
        Built by the caller. Tests inject a scripted fake instead.
    middleware : Sequence of AgentMiddleware
        The full stack, e.g. `middleware.agent_middleware(...)` with
        `CriticVerificationMiddleware` appended by the caller.
    system_prompt : str
        Resolved by the caller via `prompt_store.PromptStore.get(
        prompts.PROMPT_NAMES["critic"], label=..., variables={"today":
        date.today().isoformat()})` -- already compiled, `{{today}}` and
        all.

    Returns
    -------
    CompiledStateGraph
        A graph whose `invoke` result carries the parsed `CritiqueResult` in
        `result["structured_response"]`. Never constructed with a
        checkpointer.
    """
    assert_allowlist(tools, CRITIC_ALLOWLIST, "critic")

    graph = create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        response_format=CRITIC_RESPONSE_FORMAT,
        middleware=list(middleware),
    )
    return graph.with_config(metadata={"agent": "critic"}, tags=["agent:critic"])
