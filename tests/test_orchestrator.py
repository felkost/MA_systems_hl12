"""`orchestrator.py`: the explicit `StateGraph` coordination path
(`docs/specs/stage-4.md`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from pydantic import SecretStr

from config import Settings
from orchestrator import create_orchestrator_graph
from tests.fakes import FakeToolCallingModel

_PLAN_ARGS = {
    "goal": "g",
    "in_scope": True,
    "search_queries": ["q"],
    "sources_to_check": ["web"],
    "output_format": "narrative",
}
_OUT_OF_SCOPE_PLAN_ARGS = {
    "goal": "This is not a research question.",
    "in_scope": False,
    "search_queries": [],
    "sources_to_check": [],
    "output_format": "",
}
_APPROVE_ARGS = {
    "verdict": "APPROVE",
    "is_fresh": True,
    "is_complete": True,
    "is_well_structured": True,
    "strengths": ["s"],
    "gaps": [],
    "revision_requests": [],
}
_DRAFT_ARGS = {"filename": "report", "content": "# Report\n\nBody text"}


def _settings(**overrides: Any) -> Settings:
    return Settings(openrouter_api_key=SecretStr("test-key"), **overrides)


def _revise_args(gap: str) -> dict[str, Any]:
    return {
        "verdict": "REVISE",
        "is_fresh": False,
        "is_complete": False,
        "is_well_structured": False,
        "strengths": [],
        "gaps": [gap],
        "revision_requests": [f"fix {gap}"],
    }


def _critic_responses(*gaps_then_approve: str | None) -> list[BaseMessage]:
    """Doubled per call: the critic sub-agent carries
    `CriticVerificationMiddleware` here too, which retries once whenever no
    verification tool was called -- see `tests/test_supervisor.py`'s
    `_critic_responses` for the full rationale."""
    responses: list[BaseMessage] = []
    for gap in gaps_then_approve:
        content = (
            json.dumps(_APPROVE_ARGS) if gap is None else json.dumps(_revise_args(gap))
        )
        message = AIMessage(content=content)
        responses.extend([message, message])
    return responses


def _fake_save_report() -> tuple[BaseTool, list[tuple[str, str]]]:
    calls: list[tuple[str, str]] = []

    @tool("save_report")
    def fake_save_report(filename: str, content: str) -> str:
        """Save a report."""
        calls.append((filename, content))
        return f"Report saved to: {filename}"

    return fake_save_report, calls


def _role_models(
    *,
    plan_args: dict[str, Any] | None = None,
    critic_responses: Sequence[BaseMessage] | None = None,
    findings: str = "findings text",
) -> dict[str, FakeToolCallingModel]:
    return {
        "planner": FakeToolCallingModel(
            responses=[AIMessage(content=json.dumps(plan_args or _PLAN_ARGS))]
        ),
        "researcher": FakeToolCallingModel(responses=[AIMessage(content=findings)]),
        "critic": FakeToolCallingModel(
            responses=(
                list(critic_responses)
                if critic_responses is not None
                else _critic_responses(None)
            )
        ),
        "supervisor": FakeToolCallingModel(
            responses=[AIMessage(content=json.dumps(_DRAFT_ARGS))]
        ),
    }


# -- D4.4: the scope gate --


def test_out_of_scope_request_never_reaches_research_or_saves() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _role_models(plan_args=_OUT_OF_SCOPE_PLAN_ARGS)
    graph = create_orchestrator_graph(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="how do I make borscht")]})
    assert result.get("__interrupt__") is None
    assert calls == []
    assert "This is not a research question." in str(result["messages"][-1].content)


def test_in_scope_request_reaches_the_hitl_gate() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _role_models()
    graph = create_orchestrator_graph(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    assert result.get("__interrupt__")
    assert calls == []


# -- D4.4: the composer produces both save_report args --


def test_composer_produces_a_filename_and_content() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _role_models()
    graph = create_orchestrator_graph(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    draft = result["report_draft"]
    assert draft.filename == "report"
    assert draft.content == "# Report\n\nBody text"


# -- D4.4/D4.16: the HITL gate, approve and reject --


def test_save_executes_exactly_once_after_approve() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _role_models()
    graph = create_orchestrator_graph(
        _settings(),
        role_models=role_models,
        base_tools=[fake_save],
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "o1"}}
    graph.invoke({"messages": [HumanMessage(content="q")]}, config=config)
    graph.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config=config)
    assert calls == [("report", "# Report\n\nBody text")]


def test_save_never_executes_on_reject() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _role_models()
    graph = create_orchestrator_graph(
        _settings(),
        role_models=role_models,
        base_tools=[fake_save],
        checkpointer=MemorySaver(),
    )
    config: RunnableConfig = {"configurable": {"thread_id": "o1"}}
    graph.invoke({"messages": [HumanMessage(content="q")]}, config=config)
    result = graph.invoke(
        Command(resume={"decisions": [{"type": "reject", "message": "no"}]}),
        config=config,
    )
    assert calls == []
    assert "No report was saved" in str(result["messages"][-1].content)


def test_hitl_gate_uses_the_same_allowed_decisions_as_the_supervisor_path() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _role_models()
    graph = create_orchestrator_graph(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    interrupt_value = result["__interrupt__"][0].value
    assert interrupt_value["review_configs"][0]["allowed_decisions"] == [
        "approve",
        "reject",
    ]


# -- D4.5: the revision cap / router's three branches --


def test_revise_under_the_cap_loops_back_to_research() -> None:
    fake_save, calls = _fake_save_report()
    role_models = _role_models(critic_responses=_critic_responses("A", None))
    graph = create_orchestrator_graph(
        _settings(max_revisions=1), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    assert result["revision_round"] == 1
    assert result["latest_critique"].verdict == "APPROVE"
    assert result.get("__interrupt__")  # reached the composer -> gate


def test_revise_at_the_cap_still_reaches_the_composer() -> None:
    """A cap-exhausted REVISE run must still produce a report (D4.4's twin
    of the supervisor path's D4.18 guard) -- not a dead end. With
    max_revisions=1 the cap allows max_revisions+1=2 total critique calls,
    matching the supervisor path's arithmetic exactly."""
    fake_save, _ = _fake_save_report()
    role_models = _role_models(critic_responses=_critic_responses("A", "B"))
    graph = create_orchestrator_graph(
        _settings(max_revisions=1), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    assert result["revision_round"] == 2  # both REVISE calls counted
    assert result["latest_critique"].verdict == "REVISE"
    assert result.get("__interrupt__")  # still reached the gate
    assert result["report_draft"] is not None


def test_revision_cap_boundary_matches_the_supervisor_path_arithmetic() -> None:
    """`docs/specs/stage-4.md` D4.5: for max_revisions in {1,2,3}, total
    critique calls = {2,3,4} on both paths. Pinned here for the graph path
    with an always-REVISE critic, stopping only once the cap is exhausted."""
    for max_revisions, expected_calls in ((1, 2), (2, 3), (3, 4)):
        fake_save, _ = _fake_save_report()
        gaps = [f"gap{i}" for i in range(expected_calls)]
        role_models = _role_models(critic_responses=_critic_responses(*gaps))
        graph = create_orchestrator_graph(
            _settings(max_revisions=max_revisions),
            role_models=role_models,
            base_tools=[fake_save],
        )
        result = graph.invoke({"messages": [HumanMessage(content="q")]})
        assert result["revision_round"] == expected_calls, (
            f"max_revisions={max_revisions}: expected {expected_calls} "
            f"critique calls, got {result['revision_round']}"
        )
        assert result["latest_critique"].verdict == "REVISE"
        assert result.get("__interrupt__")  # still reached the composer/gate


def test_revision_round_starts_at_zero_and_increments_only_on_revise() -> None:
    fake_save, _ = _fake_save_report()
    role_models = _role_models()  # single APPROVE
    graph = create_orchestrator_graph(
        _settings(max_revisions=2), role_models=role_models, base_tools=[fake_save]
    )
    result = graph.invoke({"messages": [HumanMessage(content="q")]})
    assert result["revision_round"] == 0


# -- D4.3 (path 2): the original request is forwarded by code --


def test_original_request_is_forwarded_to_the_researcher() -> None:
    seen: dict[str, str] = {}

    class RecordingModel(FakeToolCallingModel):
        def _generate(  # type: ignore[override]
            self, messages: list[Any], **kwargs: Any
        ) -> Any:
            human = next(m for m in messages if isinstance(m, HumanMessage))
            seen["researcher"] = str(human.content)
            return super()._generate(messages, **kwargs)

    fake_save, _ = _fake_save_report()
    role_models = _role_models()
    role_models["researcher"] = RecordingModel(
        responses=[AIMessage(content="findings")]
    )
    graph = create_orchestrator_graph(
        _settings(), role_models=role_models, base_tools=[fake_save]
    )
    graph.invoke({"messages": [HumanMessage(content="What is the capital of France?")]})
    assert "Original request: What is the capital of France?" in seen["researcher"]
