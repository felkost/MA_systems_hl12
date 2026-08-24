"""The three sub-agent factories, under D3.2's dependency-inversion design.

`docs/specs/stage-3.md`: `create_planner_agent`/`create_research_agent`/
`create_critic_agent` take `model`, `tools` and `middleware` as parameters
and import nothing from `tools.py`/`models.py`/`middleware.py` themselves --
`tests/test_layering.py::test_module_imports_respect_its_layer` is the
standing guard for that; this file covers what it cannot: that the factory
actually *validates* what it is handed (D3.5) and never wires in a
checkpointer (D3.6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.prebuilt.tool_node import ToolNode

import tools
from agents._allowlist import ToolAllowlistError
from agents.critic import create_critic_agent
from agents.planner import create_planner_agent
from agents.research import create_research_agent
from config import Settings
from schemas import CritiqueResult, ResearchPlan
from tests.fakes import FakeToolCallingModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _settings(**overrides: Any) -> Settings:
    from pydantic import SecretStr

    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def _fake_model(response: AIMessage) -> FakeToolCallingModel:
    return FakeToolCallingModel(responses=[response])


# -- Test 3: each agent builds and invokes with a fake model, no network --


def test_planner_builds_and_returns_a_structured_plan() -> None:
    plan_args = {
        "goal": "Compare A and B",
        "in_scope": True,
        "search_queries": ["A vs B"],
        "sources_to_check": ["web"],
        "output_format": "narrative",
    }
    model = _fake_model(AIMessage(content=json.dumps(plan_args)))
    graph = create_planner_agent(
        _settings(), tools.PLANNER_TOOLS, model=model, middleware=[]
    )
    result = graph.invoke({"messages": [HumanMessage("Compare A and B")]})
    structured = result["structured_response"]
    assert isinstance(structured, ResearchPlan)
    assert structured.goal == plan_args["goal"]


def test_research_builds_and_returns_free_text() -> None:
    model = _fake_model(AIMessage(content="Findings: A beats B on cost."))
    graph = create_research_agent(
        _settings(), tools.RESEARCHER_TOOLS, model=model, middleware=[]
    )
    result = graph.invoke({"messages": [HumanMessage("Execute the plan")]})
    assert "Findings" in result["messages"][-1].content


def test_critic_builds_and_returns_a_structured_critique() -> None:
    critique_args = {
        "verdict": "APPROVE",
        "is_fresh": True,
        "is_complete": True,
        "is_well_structured": True,
        "strengths": ["clear citations"],
        "gaps": [],
        "revision_requests": [],
    }
    model = _fake_model(AIMessage(content=json.dumps(critique_args)))
    graph = create_critic_agent(
        _settings(), tools.CRITIC_TOOLS, model=model, middleware=[]
    )
    result = graph.invoke({"messages": [HumanMessage("Verify these findings")]})
    structured = result["structured_response"]
    assert isinstance(structured, CritiqueResult)
    assert structured.verdict == "APPROVE"


# -- Test 1: allowlist refusal at the factory level (D3.5) --


def test_planner_factory_refuses_a_tool_outside_its_allowlist() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    with pytest.raises(ToolAllowlistError):
        create_planner_agent(
            _settings(), tools.RESEARCHER_TOOLS, model=model, middleware=[]
        )


def test_research_factory_refuses_a_missing_expected_tool() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    with pytest.raises(ToolAllowlistError):
        create_research_agent(
            _settings(), tools.PLANNER_TOOLS, model=model, middleware=[]
        )


# -- Test 2: the agent-level allowlist constants match tools.py exactly --


def test_planner_allowlist_matches_planner_tools() -> None:
    from agents.planner import PLANNER_ALLOWLIST

    assert set(PLANNER_ALLOWLIST) == {t.name for t in tools.PLANNER_TOOLS}


def test_researcher_allowlist_matches_researcher_tools() -> None:
    from agents.research import RESEARCHER_ALLOWLIST

    assert set(RESEARCHER_ALLOWLIST) == {t.name for t in tools.RESEARCHER_TOOLS}


def test_critic_allowlist_matches_critic_tools() -> None:
    from agents.critic import CRITIC_ALLOWLIST

    assert set(CRITIC_ALLOWLIST) == {t.name for t in tools.CRITIC_TOOLS}


# -- Planner is bound only to search tools, no read_url --


def test_planner_graph_has_no_read_url() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    graph = create_planner_agent(
        _settings(), tools.PLANNER_TOOLS, model=model, middleware=[]
    )
    tools_node = cast(ToolNode, graph.nodes["tools"].bound)
    assert set(tools_node.tools_by_name.keys()) == {"web_search", "knowledge_search"}


# -- Test 13: no sub-agent carries a checkpointer (D3.6) --


@pytest.mark.parametrize(
    "path", ["agents/planner.py", "agents/research.py", "agents/critic.py"]
)
def test_agent_source_never_passes_a_checkpointer(path: str) -> None:
    """The word `checkpointer` may appear in a docstring explaining D3.6;
    it must never appear as a keyword argument to `create_agent`."""
    source = (PROJECT_ROOT / path).read_text(encoding="utf-8")
    assert "checkpointer=" not in source


def test_planner_graph_has_no_checkpointer() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    graph = create_planner_agent(
        _settings(), tools.PLANNER_TOOLS, model=model, middleware=[]
    )
    assert graph.checkpointer is None


def test_research_graph_has_no_checkpointer() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    graph = create_research_agent(
        _settings(), tools.RESEARCHER_TOOLS, model=model, middleware=[]
    )
    assert graph.checkpointer is None


def test_critic_graph_has_no_checkpointer() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    graph = create_critic_agent(
        _settings(), tools.CRITIC_TOOLS, model=model, middleware=[]
    )
    assert graph.checkpointer is None


# -- Each factory is tagged for span attribution --


def test_planner_graph_is_tagged_with_its_own_agent_name() -> None:
    model = _fake_model(AIMessage(content="", tool_calls=[]))
    graph = create_planner_agent(
        _settings(), tools.PLANNER_TOOLS, model=model, middleware=[]
    )
    assert graph.config is not None
    assert graph.config.get("metadata", {}).get("agent") == "planner"
